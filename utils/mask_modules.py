import numpy as np
import torch

from torch import nn
from torchvision.transforms import ToPILImage, ToTensor
from skimage.segmentation import find_boundaries
from PIL import Image

class CleanMask(nn.Module):
    def __init__(self,num_labels=5,ups_mask_size=512):
        super().__init__()

        self.num_labels = num_labels
        self.ups_mask_size = ups_mask_size
        self.maxpool = nn.MaxPool2d(kernel_size=6, stride=1)
        self.ten2pil = ToPILImage()
        self.pil2ten = ToTensor()

    def forward(self,mask):
        with torch.no_grad():
            # transform torch.tensor to PIL image with values in {0,1,...,num_values}
            mask = self.ten2pil(mask)
            mask = np.array(mask)
            boundary = np.asarray(find_boundaries(mask, connectivity=1, mode='thick', background=0))
            mask[boundary] = 0
            mask = torch.from_numpy(mask).float().repeat(3, 1, 1)
            mask = -self.maxpool(-mask)
            mask = torch.mean(mask, axis=0)
            mask = mask.numpy()
            mask = (self.num_labels - 1) * np.array(mask, dtype=np.float32) / 255
            mask = np.round(mask).astype(np.int32)
            mask = Image.fromarray(mask)
            mask = mask.resize((self.ups_mask_size, self.ups_mask_size), Image.NEAREST)
        return mask

class LatentMaskNormalize(nn.Module):

    def __init__(self, small=True, num_labels=5, max_norm=True):
        super().__init__()

        self.small = small
        self.num_labels = num_labels
        self.max_norm = max_norm

        if max_norm:
            if self.small:
                if num_labels == 5:
                    latent_mean = [0.0, 0.0, 0.0, 0.0]
                    latent_std = [40.0, 36.0, 48.0, 32.0]
                elif num_labels == 10:
                    latent_mean = [0.0, 0.0, 0.0, 0.0]
                    latent_std = [46.0, 36.0, 55.0, 36.0]
                else:
                    assert False, f'num_labels={num_labels} is not valid'
            else:
                if num_labels == 5:
                    latent_mean = [0.0, 0.0, 0.0, 0.0]
                    latent_std = [24.0, 25.0, 30.0, 21.0]
                elif num_labels == 10:
                    latent_mean = [0.0, 0.0, 0.0, 0.0]
                    latent_std = [21.0, 25.0, 24.0, 27.0]
                else:
                    assert False, f'number_values={num_labels} is not valid'
        else:
            if self.small:
                if num_labels == 5:
                    latent_mean = [0.2561, -5.2205, 3.2811, 1.9930]
                    latent_std = [6.5330, 7.5497, 3.3809, 4.4666]
                elif num_labels == 10:
                    latent_mean = [-0.8345, -6.7988, 3.7122, 2.8775]
                    latent_std = [5.8512, 6.8660, 3.2907, 4.0929]
                else:
                    assert False, f'number_values={num_labels} is not valid'
            else:
                if num_labels == 5:
                    latent_mean = [-4.0167, -11.0401, 5.0172, 5.4594]
                    latent_std = [3.5608, 5.2368, 2.9128, 3.0306]
                elif num_labels == 10:
                    latent_mean = [-4.2810, -11.5054, 5.1709, 5.7229]
                    latent_std = [3.1850, 4.6366, 2.7675, 2.7103]
                else:
                    assert False, f'number_values={num_labels} is not valid'

        self.latent_mean = torch.nn.parameter.Parameter(torch.tensor(latent_mean)[None, :, None, None],
                                                        requires_grad=False)
        self.latent_std = torch.nn.parameter.Parameter(torch.tensor(latent_std)[None, :, None, None],
                                                       requires_grad=False)

    def un_normalize(self, x):
        return (self.latent_std) * x + self.latent_mean

    def forward(self, x):
        return (x - self.latent_mean) / self.latent_std
