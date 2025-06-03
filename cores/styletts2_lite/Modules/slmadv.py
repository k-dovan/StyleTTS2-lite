import torch
import numpy as np
import torch.nn.functional as F

class SLMAdversarialLoss(torch.nn.Module):

    def __init__(self, model, wl, min_len, max_len, batch_percentage=0.5, skip_update=10, sig=1.5):
        super(SLMAdversarialLoss, self).__init__()
        self.model = model
        self.wl = wl
        
        self.min_len = min_len
        self.max_len = max_len
        self.batch_percentage = batch_percentage
        
        self.sig = sig
        self.skip_update = skip_update
        
    def forward(self, iters, y_rec_gt, y_rec_gt_pred, waves, mel_input_length, ref_text, ref_lengths, ref_s):
        text_mask = length_to_mask(ref_lengths).to(ref_text.device)
        t_en = self.model.text_encoder(ref_text, ref_lengths, text_mask)
            
        s_dur = ref_s[:, 128:]
        #s = ref_s[:, :128] #Not used
        
        d, _ = self.model.predictor(t_en, s_dur, 
                                                ref_lengths, 
                                                torch.randn(ref_lengths.shape[0], ref_lengths.max(), 2).to(ref_text.device), 
                                                text_mask)
        
        bib = 0

        output_lengths = []
        attn_preds = []
        
        # differentiable duration modeling
        for _s2s_pred, _text_length in zip(d, ref_lengths):

            _s2s_pred_org = _s2s_pred[:_text_length, :]

            _s2s_pred = torch.sigmoid(_s2s_pred_org)
            _dur_pred = _s2s_pred.sum(axis=-1)

            l = int(torch.round(_s2s_pred.sum()).item())
            t = torch.arange(0, l).expand(l)

            t = torch.arange(0, l).unsqueeze(0).expand((len(_s2s_pred), l)).to(ref_text.device)
            loc = torch.cumsum(_dur_pred, dim=0) - _dur_pred / 2

            h = torch.exp(-0.5 * torch.square(t - (l - loc.unsqueeze(-1))) / (self.sig)**2)

            out = torch.nn.functional.conv1d(_s2s_pred_org.unsqueeze(0), 
                                         h.unsqueeze(1), 
                                         padding=h.shape[-1] - 1, groups=int(_text_length))[..., :l]
            attn_preds.append(F.softmax(out.squeeze(), dim=0))

            output_lengths.append(l)

        max_len = max(output_lengths)
        
        with torch.no_grad():
            t_en = self.model.text_encoder(ref_text, ref_lengths, text_mask)
            
        s2s_attn = torch.zeros(len(ref_lengths), int(ref_lengths.max()), max_len).to(ref_text.device)
        for bib in range(len(output_lengths)):
            s2s_attn[bib, :ref_lengths[bib], :output_lengths[bib]] = attn_preds[bib]

        asr_pred = t_en @ s2s_attn

        _, p_pred = self.model.predictor(t_en, s_dur, 
                                                ref_lengths, 
                                                s2s_attn, 
                                                text_mask)
        
        mel_len = max(int(min(output_lengths) / 2 - 1), self.min_len // 2)
        mel_len = min(mel_len, self.max_len // 2)
        
        # get clips
        
        en = []
        p_en = []
        sp = []
        
        F0_fakes = []
        N_fakes = []
        
        wav = []

        for bib in range(len(output_lengths)):
            mel_length_pred = output_lengths[bib]
            mel_length_gt = int(mel_input_length[bib].item() / 2)
            if mel_length_gt <= mel_len or mel_length_pred <= mel_len:
                continue

            sp.append(ref_s[bib])

            random_start = np.random.randint(0, mel_length_pred - mel_len)
            en.append(asr_pred[bib, :, random_start:random_start+mel_len])
            p_en.append(p_pred[bib, :, random_start:random_start+mel_len])

            # get ground truth clips
            random_start = np.random.randint(0, mel_length_gt - mel_len)
            y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
            wav.append(torch.from_numpy(y).to(ref_text.device))
            
            if len(wav) >= self.batch_percentage * len(waves): # prevent OOM due to longer lengths
                break

        if len(sp) <= 1:
            return None
            
        sp = torch.stack(sp)
        wav = torch.stack(wav).float()
        en = torch.stack(en)
        p_en = torch.stack(p_en)
        
        F0_fake, N_fake = self.model.predictor.F0Ntrain(p_en, sp[:, 128:])
        y_pred = self.model.decoder(en, F0_fake, N_fake, sp[:, :128])
        
        # discriminator loss
        if (iters + 1) % self.skip_update == 0:
            if np.random.randint(0, 2) == 0:
                wav = y_rec_gt_pred
                use_rec = True
            else:
                use_rec = False

            crop_size = min(wav.size(-1), y_pred.size(-1))
            if use_rec: # use reconstructed (shorter lengths), do length invariant regularization
                if wav.size(-1) > y_pred.size(-1):
                    real_GP = wav[:, : , :crop_size]
                    out_crop = self.wl.discriminator_forward(real_GP.detach().squeeze())
                    out_org = self.wl.discriminator_forward(wav.detach().squeeze())
                    loss_reg = F.l1_loss(out_crop, out_org[..., :out_crop.size(-1)])

                    if np.random.randint(0, 2) == 0:
                        d_loss = self.wl.discriminator(real_GP.detach().squeeze(), y_pred.detach().squeeze()).mean()
                    else:
                        d_loss = self.wl.discriminator(wav.detach().squeeze(), y_pred.detach().squeeze()).mean()
                else:
                    real_GP = y_pred[:, : , :crop_size]
                    out_crop = self.wl.discriminator_forward(real_GP.detach().squeeze())
                    out_org = self.wl.discriminator_forward(y_pred.detach().squeeze())
                    loss_reg = F.l1_loss(out_crop, out_org[..., :out_crop.size(-1)])

                    if np.random.randint(0, 2) == 0:
                        d_loss = self.wl.discriminator(wav.detach().squeeze(), real_GP.detach().squeeze()).mean()
                    else:
                        d_loss = self.wl.discriminator(wav.detach().squeeze(), y_pred.detach().squeeze()).mean()
                
                # regularization (ignore length variation)
                d_loss += loss_reg

                out_gt = self.wl.discriminator_forward(y_rec_gt.detach().squeeze())
                out_rec = self.wl.discriminator_forward(y_rec_gt_pred.detach().squeeze())

                # regularization (ignore reconstruction artifacts)
                d_loss += F.l1_loss(out_gt, out_rec)

            else:
                d_loss = self.wl.discriminator(wav.detach().squeeze(), y_pred.detach().squeeze()).mean()
        else:
            d_loss = 0
            
        # generator loss
        gen_loss = self.wl.generator(y_pred.squeeze())
        
        gen_loss = gen_loss.mean()
        
        return d_loss, gen_loss, y_pred.detach().cpu().numpy()
    
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask
