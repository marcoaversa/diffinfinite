import os
import sys
import fire
import yaml
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T

from dataset import set_global_seed

from dm import Unet, GaussianDiffusion, Trainer
from random_diffusion import RandomDiffusion, save_tensor_as_png
from random_diffusion_masks import RandomDiffusionMasks

import importlib
import dm_masks
importlib.reload(dm_masks)

from dm_masks import Unet as MaskUnet
from dm_masks import GaussianDiffusion as MaskGD
from dm_masks import Trainer as MaskTrainer
from utils.mask_modules import CleanMask

'''Example usage: CUDA_VISIBLE_DEVICES=1 nohup python sample.py --sample_label=0 --cond_scale=0.0 >/dev/null 2>&1 &'''

'''Load Masks Model'''

milestone=5
config_file='./config/mask_gen_sample5.yaml'

with open(config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)
for key in config.keys():
    globals().update(config[key])
    
maskunet = MaskUnet(
        dim=dim,
        num_classes=num_classes,
        dim_mults=dim_mults,
        channels=channels,
        resnet_block_groups = resnet_block_groups,
        block_per_layer=block_per_layer,
    )

maskmodel = MaskGD(
        maskunet,
        image_size=mask_size//8,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2')

masktrainer = MaskTrainer(
        maskmodel,
        train_batch_size=batch_size,
        train_lr=lr,
        train_num_steps=train_num_steps,
        save_and_sample_every=save_sample_every,
        gradient_accumulate_every=gradient_accumulate_every,
        save_loss_every=save_loss_every,
        num_samples=num_samples,
        num_workers=num_workers,
        results_folder=results_folder)

masktrainer.load(milestone)
masktrainer.ema.cuda()
masktrainer.ema=masktrainer.ema.eval()

'''Load Images Model'''

milestone=10
config_file='./config/image_gen_sample5.yaml'

with open(config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)
for key in config.keys():
    globals().update(config[key])

unet = Unet(
        dim=dim,
        num_classes=num_classes,
        dim_mults=dim_mults,
        channels=channels,
        resnet_block_groups = resnet_block_groups,
        block_per_layer=block_per_layer,
    )

model = GaussianDiffusion(
        unet,
        image_size=image_size//8,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2')

trainer = Trainer(
        model,
        train_batch_size=batch_size,
        train_lr=lr,
        train_num_steps=train_num_steps,
        save_and_sample_every=save_sample_every,
        gradient_accumulate_every=gradient_accumulate_every,
        save_loss_every=save_loss_every,
        num_samples=num_samples,
        num_workers=num_workers,
        results_folder=results_folder)

trainer.load(milestone)
trainer.ema.cuda()
trainer.ema=trainer.ema.eval()

@torch.no_grad()
def sample(masks, cond_scale=3.0):
    z = torch.ones((masks.shape[0],
                    4,512//8,512//8), device='cuda:0')
    z = trainer.ema.ema_model.sample(z,masks, cond_scale=cond_scale+1)*50
    return torch.clip(trainer.vae.decode(z).sample,0,1)

def main(
        n_imgs=2500,
        batch_size=16,
        cond_scale=0.0,
        sample_label=0,
        imgs_path='./results/patches/'
         ):
    
    imgs_path=imgs_path+f'omega{cond_scale:.1f}/'
    os.makedirs(imgs_path, exist_ok=True)


    N=n_imgs//batch_size
    n_batches=1
    
    for _ in range(N):
        
        '''Mask Generation'''

        masks = masktrainer.sample_loop(n_batches, batch_size=batch_size, sample_label=sample_label, save_sample=False)

        *_, h,w = masks.shape
        masks = masks.reshape(-1, 1, h, w)

        '''Image Generation'''

        imgs=sample(masks.cuda(), cond_scale=cond_scale)

        '''Save Images'''

        for j in range(len(imgs)):
            n=len([file for file in os.listdir(imgs_path) if file.endswith('png')])
            save_tensor_as_png(imgs[j], os.path.join(imgs_path, f'sample_label{sample_label}_{n:04d}.jpg'))

            image = Image.fromarray(masks[j,0].int().numpy())
            image.save(os.path.join(imgs_path, f'sample_label{sample_label}_{n:04d}_mask.png'))
            
if __name__=='__main__':
    fire.Fire(main)