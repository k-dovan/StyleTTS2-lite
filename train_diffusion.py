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

from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules.diffusion.diffusion import AudioDiffusionConditional
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule, KDiffusion, LogNormalDistribution

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
    model = build_model(model_params)
    _ = [model[key].to(device) for key in model]

    # DP
    for key in model:
        if key != "mpd" and key != "msd":
            model[key] = MyDataParallel(model[key])

    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != ''

    # DEFINE DIFFUSION MODEL
    
    diffusion_params = {
        "dist": {"estimate_sigma_data": True, "mean": -3.0, "sigma_data": 0.19926648961191362, "std": 1.0},
        "transformer": { "head_features": 64, "multiplier": 2, "num_heads": 8, "num_layers": 3 },
        "embedding_mask_proba": 0.1
    }
    style_dim = config['model_params']['style_dim']
    hidden_dim = config['model_params']['hidden_dim']
    transformer = StyleTransformer1d(channels=style_dim, 
                                    context_embedding_features=hidden_dim,
                                    context_features=style_dim, 
                                    **diffusion_params['transformer'])
    
    model_diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=512,
        embedding_features=hidden_dim,
        embedding_mask_proba=diffusion_params["embedding_mask_proba"], # Conditional dropout of batch elements,
        channels=style_dim,
        context_features=style_dim,
    )
    
    model_diffusion.diffusion = KDiffusion(
        net=model_diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = diffusion_params["dist"]["mean"], std = diffusion_params["dist"]["std"]),
        sigma_data=diffusion_params["dist"]["sigma_data"], # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0 
    )
    model_diffusion.diffusion.net = transformer
    model_diffusion.unet = transformer

    sampler = DiffusionSampler(
        model_diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )

    model_diffusion =   MyDataParallel(model_diffusion) 

    # END DIFFUSION
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    scheduler_params_dict= {"diffusion": scheduler_params.copy()}
    
    diffusion_optimizer = build_optimizer({'diffusion': model_diffusion.parameters()},
                                        scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
        
        
    # load models if there is a model
    if load_pretrained:
        print("Loading main model...")
        try:
            training_strats = config['training_strats']
        except Exception as e:
            print("\nNo training_strats found in config. Proceeding with default settings...")
            training_strats = {}
            training_strats['ignore_modules'] = ''
            training_strats['freeze_modules'] = ''

        model, _, start_epoch, iters = load_checkpoint(model,  None, 
                                                        config['pretrained_model'], 
                                                        load_only_params= True,
                                                        ignore_modules=training_strats['ignore_modules'],
                                                        freeze_modules=training_strats['freeze_modules'])
    else:
            raise Exception('Must have a pretrained!')
    
    # check in the same dir exist diffusion checkpoint
    from pathlib import Path
    parent_dir = Path(config['pretrained_model']).parent
    diff_model_path = parent_dir / "current_diffusion.pth"

    if diff_model_path.is_file():
        print("Found 'current_diffusion.pth'.")
        print("Loading diffusion module...")
        model_diffusion, diffusion_optimizer, start_epoch, iters = load_checkpoint(model_diffusion,  diffusion_optimizer, 
                                                        config['pretrained_model'], 
                                                        load_only_params= False,
                                                        ignore_modules=training_strats['ignore_modules'],
                                                        freeze_modules=training_strats['freeze_modules'])
    else:
        print("'current_diffusion.pth' not found! Initializing a new one...")
        
        
    n_down = model.text_aligner.n_down

    iters = 0
    
    torch.cuda.empty_cache()
    
    print('\diffusion', diffusion_optimizer.optimizers['diffusion'])
    
############################################## TRAIN ##############################################

    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        _ = [model[key].eval() for key in model]

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, mels, mel_input_length = batch
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # compute the style of the entire utterance
            s = model.style_encoder(mels.unsqueeze(1))
            
            num_steps = np.random.randint(3, 5)
            s_preds = sampler(  noise = torch.randn_like(s).unsqueeze(1).to(device), 
                                embedding=t_en.transpose(-1, -2),
                                embedding_scale=1,
                                features=s, # reference from the same speaker as the embedding
                                embedding_mask_proba=0.1,
                                num_steps=num_steps).squeeze(1)
            loss_diff = model_diffusion.diffusion(s.unsqueeze(1), embedding=t_en.transpose(-1, -2), features=s).mean() # EDM loss
            loss_sty = F.l1_loss(s_preds, s.detach()) # style reconstruction loss

            diffusion_optimizer.step('diffusion')
    
            if (i+1)%log_interval == 0:
                logger.info (f'Epoch [{epoch+1}/{epochs}], \
                             Step [{i+1}/{len(train_list)//batch_size}], \
                             Diff Loss: {loss_diff}, \
                             Sty Loss: {loss_sty}')
                
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/diff_loss', loss_diff, iters)
                
                print('Time elasped:', time.time()-start_time)

            if iters % 100 == 0: # Save to current_model every 2000 iters
                state = {
                    'net':  {'diffusion': model_diffusion.state_dict()}, 
                    'optimizer': diffusion_optimizer.state_dict(),
                    'iters': iters,
                    'epoch': epoch,
                }
                save_path = os.path.join(log_dir, 'current_diffusion.pth')
                torch.save(state, save_path)  


############################################## EVAL ##############################################


                            
if __name__=="__main__":
    main()