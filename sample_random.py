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


'''Example usage: CUDA_VISIBLE_DEVICES=1 python sample.py --cond_scale=0.0'''


'''Mask Model'''

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

'''Images Model'''

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

def main(
        n_images=512,
        batch_size=4,
        label=1,
        image_size=2048,
        cond_scale=3.0,
        sampling_steps=250,
        imgs_path='./results/large'):

    imgs_path=imgs_path+f'omega{cond_scale:.1f}/'
    os.makedirs(imgs_path, exist_ok=True)
    
    with torch.no_grad():
        for _ in range(n_images//batch_size):
            num_classes=5
            labels=[label for i in range(batch_size)]

            masks=torch.cat([torch.ones((1,1,image_size//4,image_size//4))*label for label in labels], 0)

            random_sample_masks = RandomDiffusionMasks(
                                    masktrainer,
                                    patch_size=128, 
                                    sampling_steps=sampling_steps, 
                                    cond_scale=0.0)

            random_sample = RandomDiffusion(
                                trainer,
                                patch_size=512, 
                                sampling_steps=sampling_steps, 
                                cond_scale=cond_scale)


            zs,_ = random_sample_masks(masks.cuda())

            masks=[]
            for z in zs:
                mask = random_sample_masks.hann_tile_overlap(z[None])
                mask = torch.round((num_classes - 1) * mask[0])
                mask = torch.round(torch.mean(mask, axis=0, keepdim=True)) / (num_classes - 1)
                mask_upsampled = CleanMask(num_labels=num_classes,ups_mask_size=image_size)(mask)
                mask = T.ToTensor()(mask_upsampled)[None]
                masks.append(mask)
            masks=torch.cat(masks,0)

            zs,_ = random_sample(masks.cuda())

            imgs=[]
            for z in zs:
                imgs.append(random_sample.hann_tile_overlap(z[None]))
            imgs=torch.cat(imgs,0)


            """ Save Images """
            for j in range(len(masks)):
                n=len([file for file in os.listdir(imgs_path) if file.endswith('png')])
                save_tensor_as_png(imgs[j], os.path.join(imgs_path, f'sample{n:04d}.jpg'))

                image = Image.fromarray(masks[j,0].numpy())
                image.save(os.path.join(imgs_path, f'sample{n:04d}_mask.png'))
            
if __name__=='__main__':
    fire.Fire(main)