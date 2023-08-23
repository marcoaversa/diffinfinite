import math
import copy
import os
import time as random_time
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import random as py_random
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torchvision.transforms as T

from torchvision.transforms import ToPILImage,ToTensor
from torch.nn import MaxPool2d

from torch.optim import Adam, lr_scheduler
from ema_pytorch import EMA

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import DiffusionPipeline

from utils.mask_modules import CleanMask, LatentMaskNormalize
from dataset import ComposeState, RandomRotate90
from dataset_masks import import_small_mask_dataset
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img# * 2 - 1

def unnormalize_to_zero_to_one(t):
    return t#(t + 1) * 0.5

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
#         self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        block_per_layer=2,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
    ):
        super().__init__()

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            blocks=[]
            for i in range(block_per_layer):
                blocks+=[block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),]
            
            blocks+=[Residual(PreNorm(dim_in, LinearAttention(dim_in))),]
            blocks+=[Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1),]
            
            self.downs.append(nn.ModuleList(blocks))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            blocks=[]
            for i in range(block_per_layer):
                blocks+=[block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),]
            
            blocks+=[Residual(PreNorm(dim_out, LinearAttention(dim_out))),]
            blocks+=[Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)]
            
            self.ups.append(nn.ModuleList(blocks))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        classes,
        cond_drop_prob = None
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance        

        classes_emb = self.classes_emb(classes)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for *blocks, attn, downsample in self.downs:
            for i, block in enumerate(blocks):
                x = block(x, t, c)
                if i < len(blocks)-1:
                    h.append(x)
                
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for *blocks, attn, upsample in self.ups:
            for block in blocks:
                x = torch.cat((x, h.pop()), dim = 1)
                x = block(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, cond_scale = 3., clip_x_start = False):
        model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, cond_scale)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 3., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 3., verbose=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        if verbose:
            iterator = tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)
        else:
            iterator = reversed(range(0, self.num_timesteps))
            
        for t in iterator:
            img, x_start = self.p_sample(img, t, classes, cond_scale)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 3., clip_denoised = True, verbose=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        x_start = None

        if verbose:
            iterator = tqdm(time_pairs, desc = 'sampling loop time step')
        else:
            iterator = time_pairs
        for time, time_next in iterator:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, cond_scale = 3., verbose=False):
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes=classes, shape=(batch_size, channels, image_size, image_size),  cond_scale=cond_scale, verbose=verbose)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, *, classes, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step

        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
            
        model_out=torch.nan_to_num(model_out)
        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            *,
            train_batch_size=32,
            cond_scale=3.0,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            save_loss_every=100,
            num_workers=0,
            num_samples=4,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            format='grayscale',
            not_zero_frac=0.3,
            num_labels=5,
            max_norm=True,
            mask_size=128,
            ups_mask_size=512,
            data_path=None,
            augmentation=True,
            small=True,
            save_folder='samples/'
    ):
        super().__init__()

        # torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))

        self.latent_normalize = LatentMaskNormalize(small=small, num_labels=num_labels, max_norm=max_norm)
        self.clean_mask = CleanMask(num_labels=num_labels,ups_mask_size=ups_mask_size)

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            gradient_accumulation_steps=gradient_accumulate_every
        )

        self.latent_normalize, self.clean_mask = self.accelerator.prepare(self.latent_normalize,self.clean_mask)

        self.processes = [i + 1 for i in range(self.accelerator.state.num_processes)]
        self.accelerator.native_amp = amp
        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.save_loss_every = save_loss_every

        self.batch_size = train_batch_size
        self.num_workers = num_workers
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.cond_scale = cond_scale

        self.mask_size = mask_size
        self.ups_mask_size = ups_mask_size
        self.format = format
        self.not_zero_frac = not_zero_frac
        self.num_labels = num_labels
        self.augmentation = augmentation
        self.small = small
        self.save_folder = save_folder

        if data_path:
            if self.augmentation:
                transform = ComposeState([
                    T.PILToTensor(),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    RandomRotate90(),
                    T.Lambda(lambda x: 2 * (x / (self.num_labels - 1)) - 1),
                    T.Lambda(lambda x: x.repeat(3, 1, 1)),
                ])
            else:
                transform = ComposeState([
                    T.PILToTensor(),
                    T.Lambda(lambda x: 2 * (x / (self.num_labels - 1)) - 1),
                    T.Lambda(lambda x: x.repeat(3, 1, 1)),
                ])

            self.data_path = data_path

            if self.small:
                train_loader, test_loader = import_small_mask_dataset(
                    data_path=self.data_path,
                    num_labels=self.num_labels,
                    use_split=True,
                    size=self.mask_size,
                    transform=transform,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    small=True)
            else:
                train_loader, test_loader = import_small_mask_dataset(
                    data_path=self.data_path,
                    num_labels=self.num_labels,
                    use_split=False,
                    size=self.mask_size,
                    transform=transform,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    small=False)

            train_loader = self.accelerator.prepare(train_loader)
            self.dl = cycle(train_loader)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        #         if self.accelerator.is_main_process:
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0
        self.running_loss = []
        self.running_lr = []

        # prepare model, dataloader, optimizer with accelerator

        self.scheduler = lr_scheduler.OneCycleLR(self.opt, max_lr=train_lr, total_steps=train_num_steps + 1)
        self.model, self.opt, self.ema, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.ema,
                                                                                  self.scheduler)
        self.latent_normalize = self.accelerator.prepare(self.latent_normalize)

        repo_id = "stabilityai/stable-diffusion-2-base"
        self.vae = DiffusionPipeline.from_pretrained(repo_id).vae
        #   PATH = './vae/stable_dif_2_base_weights.pth'
        #   self.vae = torch.load(PATH)
        self.vae = self.accelerator.prepare(self.vae)

        # to clean masks
        self.maxpool = MaxPool2d(kernel_size=6, stride=1)
        self.ten2pil = ToPILImage()
        self.pil2ten = ToTensor()
        self.maxpool, self.ten2pil, self.pil2ten = self.accelerator.prepare(self.maxpool, self.ten2pil,
                                                                            self.pil2ten)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'loss': self.running_loss,
            'lr': self.running_lr,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.accelerator.get_state_dict(self.ema),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone,show_train=False):

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.accelerator.device)

        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.ema = self.accelerator.unwrap_model(self.ema)
        self.ema.load_state_dict(data['ema'])
        self.running_loss = data['loss']
        self.running_lr = data['lr']

        if exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        self.model, self.opt, self.ema, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.ema,
                                                                                  self.scheduler)
        if show_train:
            plt.plot(self.running_loss)
            plt.show()

            plt.plot(self.running_lr)
            plt.show()

    def train_loop(self, imgs, labels):
        with torch.no_grad():
            imgs = self.vae.module.encode(imgs).latent_dist.sample()

        with self.accelerator.autocast():
            imgs = self.latent_normalize(imgs)
            loss = self.model(img=imgs, classes=labels)

        self.accelerator.backward(loss)

        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

        self.opt.step()
        self.opt.zero_grad()
        self.scheduler.step()

        return loss

    def sample_loop(self, n_batches, batch_size=1, sample_label=None, return_sample=True, a=0.0, b=1.0, save_sample=True):

        with torch.no_grad():
            self.ema.to(self.accelerator.device)
#             if torch.cuda.is_available():
#                 self.ema.module.update()
#                 self.ema.module.ema_model.eval()
#             else:
            self.ema.update()
            self.ema.ema_model.eval()

            stamp = str(random_time.time() - int(random_time.time()))[2:7]
            self.accelerator.print(
                f'generate {n_batches * batch_size} masks using device={self.accelerator.device}')
            rand_int = int(str(random_time.time()).split('.')[-1])
            random_seed = (self.accelerator.local_process_index + 1) * rand_int
            set_seed(random_seed)
            gen_n_samples = 0
            masks = torch.zeros((n_batches, batch_size, 1, self.ups_mask_size, self.ups_mask_size))
            for n in range(n_batches):
                if sample_label == None:
                    label = 0 if n % 2 == 0 else 1
                else:
                    label = sample_label

#                 if torch.cuda.is_available():
#                     image_classes = torch.tensor([label] * batch_size).to(self.accelerator.device)
#                     z = self.ema.module.ema_model.sample(image_classes, cond_scale=self.cond_scale)
#                     z = self.latent_normalize.un_normalize(z)
#                     y = self.vae.module.decode(z).sample
#                     x = torch.clip((y + 1) / 2, 0, 1)
#                 else:
                image_classes = torch.tensor([label] * batch_size).to(torch.int64).to(self.accelerator.device)
                z = self.ema.ema_model.sample(image_classes, cond_scale=self.cond_scale)
                z = self.latent_normalize.un_normalize(z)
                y = self.vae.decode(z).sample
                x = torch.clip((y + 1) / 2, 0, 1)

                x = torch.round((self.num_labels - 1) * x)
                x = torch.round(torch.mean(x, axis=1, keepdim=True)) / (self.num_labels - 1)
                for m in range(batch_size):
                    mask = x[m]
                    mask = self.clean_mask(mask)
                    if save_sample:
                        save_path = os.path.join(self.results_folder, self.save_folder)
                        Path(save_path).mkdir(parents=True, exist_ok=True)
                        mask_path = save_path + f'{label}_label_{n + 1}_batch_{m + 1}_sample_{stamp}.png'
                        print(f'save at {mask_path}')
                        mask.save(mask_path)
                    gen_n_samples += 1
                    if return_sample:
                        masks[n,m] = self.pil2ten(mask)
                    if gen_n_samples % 1000 == 0:
                        self.accelerator.print(f'generated so far {gen_n_samples} samples.')

        self.accelerator.print(f'Sampling complete.')
        return masks if return_sample else None

    def eval_loop(self):
        if self.accelerator.is_main_process:

            self.ema.to(self.accelerator.device)
            self.ema.module.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema.module.ema_model.eval()

                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every

                    #     test_images, labels = next(self.test_loader)
                    #     utils.save_image((test_images[:self.num_samples] + 1) / 2,
                    #                      str(self.results_folder / f'images-{milestone}.png'),
                    #                      nrow=int(math.sqrt(self.num_samples)))

                    image_classes = torch.tensor([0, 0, 1, 1]).cuda()

                    z = self.ema.module.ema_model.sample(image_classes, cond_scale=self.cond_scale)
                    z = self.latent_normalize.un_normalize(z)

                    y = self.vae.module.decode(z).sample
                    samples = torch.clip((y + 1) / 2, 0, 1)
                    utils.save_image(samples,
                                     str(self.results_folder / f'rgb-sample-{milestone}.png'),
                                     nrow=int(math.sqrt(self.num_samples)))

                    int_samples = torch.round((self.num_labels - 1) * samples)
                    gray_samples = torch.round(torch.mean(int_samples, axis=1, keepdim=True)) / (
                                self.num_labels - 1)
                    # gray_samples = torch.mean(samples,axis=1,keepdim=True)

                    utils.save_image(gray_samples,
                                     str(self.results_folder / f'gray-sample-{milestone}.png'),
                                     nrow=int(math.sqrt(self.num_samples)))

                    # int_samples = torch.round(((self.num_labels-1)*gray_samples).cpu().detach()).type(torch.int32)/(self.num_labels-1)
                    # int_samples = torch.round(((self.num_labels - 1) * gray_samples).cpu().detach()).type(torch.int32)/(self.num_labels-1)
                    # utils.save_image(int_samples,
                    #                 str(self.results_folder / f'int-_float-sample-{milestone}.png'),
                    #                 nrow=int(math.sqrt(self.num_samples)))
                    '''
                   # z = self.vae.encode(
                   #     test_images[:self.num_samples]).latent_dist.sample()/50
                    test_masks = torch.ones((4, 1, 128, 128), device='cuda:0').int()
                    z = torch.ones((4, 4, 512 // 64, 512 // 64), device='cuda:0')
                   # test_masks = torch.ones((2, 2, 128, 128)).int()
                   # z = torch.ones((2, 4, 512 // 64, 512 // 64))
                    print('z',z.shape)
                    print('test',test_masks[:self.num_samples].shape)
                    z = self.ema.ema_model.sample(z, test_masks[:self.num_samples]) * 50
                    #z = self.ema.ema_model.sample(z) * 50
                    test_samples = torch.clip((self.vae.decode(50*z).sample+1)/2, 0, 1)


                utils.save_image((test_images[:self.num_samples]+1)/2,
                                 str(self.results_folder / f'images-{milestone}.png'),
                                 nrow=int(math.sqrt(self.num_samples)))
            #    print('saved test images')
            #    utils.save_image((test_masks > 0).float()[:self.num_samples],
            #                     str(self.results_folder / f'masks-{milestone}.png'),
            #                     nrow=int(math.sqrt(self.num_samples)))

                utils.save_image(test_samples,
                                 str(self.results_folder / f'sample-{milestone}.png'),
                                 nrow=int(math.sqrt(self.num_samples)))
                    '''
                self.save(milestone)

    def train(self):

        with tqdm(initial=self.step, total=self.train_num_steps // self.save_and_sample_every,
                  disable=not self.accelerator.is_main_process) as pbar:
            # with tqdm(initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps + 1:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data, masks = next(self.dl)
                    with self.accelerator.accumulate(self.model):
                        loss = self.train_loop(data, masks)
                        total_loss += loss.item()

                total_loss /= self.gradient_accumulate_every
                if self.step % self.save_loss_every == 0:
                    self.running_loss.append(total_loss)
                    self.running_lr.append(self.scheduler.get_lr()[0])

                #  pbar.set_description(f'loss: {total_loss:.4f}')

                # if self.step % self.save_and_sample_every == 0 or self.step==0:
                # if self.step ==0:
                # self.accelerator.print(f'it:{self.step}-loss:{total_loss}')
                # print(f'it:{self.step}-loss:{total_loss}')

                self.step += 1
                self.eval_loop()
                # pbar.update(1)
                if self.step % self.save_and_sample_every == 0:
                    pbar.update(1)

        self.accelerator.print('training complete')

# example

if __name__ == '__main__':
    num_classes = 10

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = num_classes,
        cond_drop_prob = 0.5
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000
    ).cuda()

    training_images = torch.randn(8, 3, 128, 128).cuda() # images are normalized from 0 to 1
    image_classes = torch.randint(0, num_classes, (8,)).cuda()    # say 10 classes

    loss = diffusion(training_images, classes = image_classes)
    loss.backward()

    # do above for many steps

    sampled_images = diffusion.sample(
        classes = image_classes,
        cond_scale = 3.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    sampled_images.shape # (8, 3, 128, 128)