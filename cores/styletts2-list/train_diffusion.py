# load packages
import os
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
import torch.nn.functional as F
import click
import shutil
import warnings
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader

from models import *
from losses import *
from utils import *

from optimizers import build_optimizer

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yaml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    
    log_dir = config['log_dir']
    if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    debug = config.get('debug', True)
    epochs = config.get('epochs', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    data_params = config.get('data_params', None)
    train_path = data_params['train_data']
    root_path = data_params['root_path']
    max_len = config.get('max_len', 200)

    try:
        symbols = (
                        list(config['symbol']['pad']) +
                        list(config['symbol']['punctuation']) +
                        list(config['symbol']['letters']) +
                        list(config['symbol']['letters_ipa']) +
                        list(config['symbol']['extend'])
                    )
        symbol_dict = {}
        for i in range(len((symbols))):
            symbol_dict[symbols[i]] = i

        n_token = len(symbol_dict) + 1
        print("\nFound:", n_token, "symbols")
    except Exception as e:
        print(f"\nERROR: Cannot find {e} in config file!\nYour config file is likely outdated, please download updated version from the repository.")
        raise SystemExit(1)
    
    loss_params = Munch(config['loss_params'])
    optimizer_params = Munch(config['optimizer_params'])

    train_list, _ = get_data_path_list(train_path, train_path)
    device = 'cuda'

    print("\n")
    print("Initializing train_dataloader")
    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        symbol_dict,
                                        batch_size=batch_size,
                                        num_workers=3,
                                        dataset_config={"debug": debug},
                                        device=device)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    model_params['n_token'] = n_token
    #model = build_model(model_params)
    enabled_modules = [
      "decoder",
      "predictor",
      "text_encoder",
      "style_encoder",
      "text_aligner",
      "pitch_extractor",
      "mpd",
      "msd",
      'diffusion'
    ]
    model = build_model_custom(model_params, enabled_modules)
    _ = [model[key].to(device) for key in model]

    # DP
    for key in model:
        if key != "mpd" and key != "msd":
            model[key] = MyDataParallel(model[key])

    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != ''

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                          scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    # load models if there is a model
    if load_pretrained:
        print("\nLoading main model...")
        try:
            training_strats = config['training_strats']
        except Exception as e:
            print("\nNo training_strats found in config. Proceeding with default settings...")
            training_strats = {}
            training_strats['ignore_modules'] = ''
            training_strats['freeze_modules'] = ''

        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, 
                                                               config['pretrained_model'], 
                                                               load_only_params=config.get('load_only_params', True),
                                                               ignore_modules=training_strats['ignore_modules'],
                                                               freeze_modules=training_strats['freeze_modules'])
    else:
            raise Exception('Must have a pretrained!')
        
    iters = 0
    torch.cuda.empty_cache()
    print('\ndecoder', optimizer.optimizers['decoder'])
    
############################################## TRAIN ##############################################

    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        _ = [model[key].eval() for key in model]

        model.diffusion.train()

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, mels, mel_input_length = batch
            
            with torch.no_grad():
                text_mask = length_to_mask(input_lengths).to(texts.device)
        
            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # cut
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            gt = []
            wav = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

            gt = torch.stack(gt).detach()
            wav = torch.stack(wav).float().detach()

            s = model.style_encoder(gt.unsqueeze(1))
            
            num_steps = np.random.randint(3, 5)
            s_preds = sampler(  noise = torch.randn_like(s).unsqueeze(1).to(device), 
                                embedding=t_en.transpose(-1, -2),
                                embedding_scale=1,
                                embedding_mask_proba=0.1,
                                num_steps=num_steps).squeeze(1)
            
            optimizer.zero_grad()
            loss_diff = model.diffusion(s.unsqueeze(1), embedding=t_en.transpose(-1, -2)).mean() # EDM loss
            loss_sty = F.l1_loss(s_preds, s.detach()) # style reconstruction loss
            style_loss = loss_params.lambda_diff* loss_diff   +\
                     loss_params.lambda_sty * loss_sty
            style_loss.backward()
            optimizer.step('diffusion')

            iters = iters + 1
    
            if (i+1)%log_interval == 0:
                logger.info(
                    f'Epoch [{epoch+1}/{epochs}], '
                    f'Step [{i+1}/{len(train_list)//batch_size}], '
                    f'Diff Loss: {loss_diff:.5f}, '
                    f'Sty Loss: {loss_sty:.5f}'
                )

                writer.add_scalar('train/sty_loss', float(f"{loss_sty:.5f}"), iters)
                writer.add_scalar('train/diff_loss', float(f"{loss_diff:.5f}"), iters)
                
                print('Time elasped:', time.time()-start_time)

            if iters % 2000 == 0: # Save to current_model every 2000 iters
                state = {
                    'net':  {key: model[key].state_dict() for key in model}, 
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': 0,
                    'epoch': epoch,
                }
                save_path = os.path.join(log_dir, 'current_model.pth')
                torch.save(state, save_path)  

        if (epoch + 1) % save_freq == 0 :
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': 0,
                'epoch': epoch,
            }
            save_path = os.path.join(log_dir, 'epoch_%05d.pth' % epoch)
            torch.save(state, save_path)  


############################################## EVAL ##############################################


                            
if __name__=="__main__":
    main()