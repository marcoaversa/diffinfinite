from fire import Fire
import yaml
from dm_masks import Unet, GaussianDiffusion, Trainer

def main(
        config_file: str = None,
        dim: int = 64,
        num_classes: int = 2,
        num_labels: int = 5,
        dim_mults: str = '1 2 4 8',
        channels: int = 4,
        resnet_block_groups: int = 2,
        block_per_layer: int = 2,
        timesteps: int = 1000,
        sampling_timesteps: int = 250,
        batch_size: int = 32,
        lr: float = 1e-4,
        train_num_steps: int = 100000,
        save_sample_every: int = 20000,
        gradient_accumulate_every: int = 1,
        save_loss_every: int = 100,
        num_samples: int = 4,
        num_workers: int = 32,
        small: bool = True,
        results_folder: str = './results',
        milestone: int = None,
        mask_size: int = 128,
        ups_mask_size: int = 512,
):

    dim_mults=[int(mult) for mult in dim_mults.split(' ')]

    if config_file:
        with open(config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)
        for key in config.keys():
            locals().update(config[key])
    
    z_size=mask_size//8
    
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
            image_size=z_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type='l2')

    trainer = Trainer(
            model,
            mask_size = mask_size,
            ups_mask_size=ups_mask_size,
            train_batch_size=batch_size,
            train_lr=lr,
            train_num_steps=train_num_steps,
            save_and_sample_every=save_sample_every,
            gradient_accumulate_every=gradient_accumulate_every,
            save_loss_every=save_loss_every,
            num_samples=num_samples,
            num_workers=num_workers,
            results_folder=results_folder
            )

    if milestone:
        trainer.load(milestone)
        
    trainer.train()

if __name__=='__main__':
    Fire(main)
