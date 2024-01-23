import sys

import torch
import torchvision.transforms as T

class RandomDiffusionMasks:
    def __init__(self, trainer, patch_size=512, sampling_steps=50, cond_scale=3.0):
        self.vae=trainer.vae.cuda()
        self.ema_model=trainer.ema.ema_model.cuda()
        self.sampling_steps=sampling_steps
        self.cond_scale=cond_scale+1
        self.patch_size=patch_size
        
        times = torch.linspace(-1, 1000 - 1, steps=self.sampling_steps + 1)
        times = list(reversed(times.int().tolist()))
        self.time_pairs = list(zip(times[:-1], times[1:]))
        
        
    @torch.no_grad()
    def encode(self, images, labels):
        return self.vae.encode(images).latent_dist.sample()/50
    
    @torch.no_grad()
    def decode(self, z):
        return torch.clip(self.vae.decode(z).sample,0,1)
    
    def ddim_step(self, img, classes, time, time_next, cond_scale):
        
        time_cond = torch.full((len(img),), time, dtype=torch.long).to(img.device)
        pred_noise, x_start, *_ = self.ema_model.model_predictions(img, time_cond, classes, 
                                                                   cond_scale = cond_scale, 
                                                                   clip_x_start = True)

        if time_next < 0:
            img = x_start

        alpha = self.ema_model.alphas_cumprod[time]
        alpha_next = self.ema_model.alphas_cumprod[time_next]

        sigma = self.ema_model.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise
        
        return img
        
    @torch.no_grad()
    def sample_one(self, x_t, x_stack, masks, times):
        
        classes=torch.cat([torch.unique(mask).max()[None] for mask in masks],0).int()
        
        img=x_t.clone()
        uniques=torch.unique(times)
        
        vmin=uniques[0]
        if len(uniques)>1:
            for unique in uniques[1:]:
                to_change=torch.where(times==unique, 1,0)
                x_t=x_stack[:,vmin]*to_change+x_t*(to_change==0)

        t,t_next=self.time_pairs[vmin][0], self.time_pairs[vmin][1]
        x_t, classes = x_t.cuda(), classes.cuda()
        pred = self.ddim_step(x_t, classes, t, t_next, cond_scale=self.cond_scale).cpu()
                     
        time_mask=torch.where(times==vmin, 1,0)
        return pred*time_mask+img*(time_mask==0)
    
    
    def get_value_coordinates(self,tensor):
        value_indices = torch.nonzero(tensor == tensor.min(), as_tuple=False)
        random_indices = value_indices[torch.randperm(value_indices.size(0))]
        return random_indices
    
    
    def random_crop(self, image, i, j, latent=True):
        if latent:
            p=self.patch_size // 16
            return image[...,i-p:i+p, j-p:j+p]
        else:
            p=self.patch_size // 2
            return image[...,i*8-p:i*8+p, j*8-p:j*8+p]

    def decoding_tiled_image(self, z, size):
        p8=self.patch_size//8
        img = [self.decode(t[None].cuda()) for t in tile(z, p8)]
        return untile(torch.cat(img, 0), size, self.patch_size).cpu()        
        
    def hann_tile_overlap(self, z):
        assert z.shape[-1]%(self.patch_size//8)==0, 'For simplicity, the code is implemented to generate images that are multiples of the (patch size//8) in the latent. Crop z to a multiple of the (patch size//8)'
        
        windows = hann_window(self.patch_size)
        b, c, h, w = z.shape
        z_scaled = z.clone() * 50
        p = self.patch_size
        p16 = self.patch_size // 16

        # Full image decoding
        img = self.decoding_tiled_image(z_scaled, (b, 3, h * 8, w * 8))
        # Vertical slice decoding
        img_v = self.decoding_tiled_image(z_scaled[..., p16:-p16], (b, 3, h * 8, w * 8 - p))
        # Horizontal slice decoding
        img_h = self.decoding_tiled_image(z_scaled[..., p16:-p16, :], (b, 3, h * 8 - p, w * 8))
        # Cross slice decoding
        img_cross = self.decoding_tiled_image(z_scaled[..., p16:-p16, p16:-p16], 
                                              (b, 3, h * 8 - p, w * 8 - p))

        # Applying windows
        b, c, h, w = img.shape
        repeat_v = windows['vertical'].repeat(b, c, h // p, w // p - 1)
        repeat_h = windows['horizontal'].repeat(b, c, h // p - 1, w // p)
        repeat_c = windows['center'].repeat(b, c, h // p - 1, w // p - 1)

        img[..., p//2:-p//2] = img[..., p//2:-p//2] * (1 - repeat_v) + img_v * repeat_v
        img[..., p//2:-p//2, :] = img[..., p//2:-p//2, :] * (1 - repeat_h) + img_h * repeat_h
        img[..., p//2:-p//2, p//2:-p//2] = img[..., p//2:-p//2, p//2:-p//2] * (1 - repeat_c) + img_cross * repeat_c

        return img
    
    @torch.no_grad()
    def __call__(self, masks):
        
        b,c,_,image_size=masks.shape
        
        img_stack=torch.randn(b, self.sampling_steps, 4, image_size//8, image_size//8)
        times=torch.zeros((b,1,image_size//8,image_size//8)).int()

        img0=torch.randn(b,4,image_size//8,image_size//8)
        p=self.patch_size//16
        s=image_size//8
        
        while times.float().mean()!=(self.sampling_steps-1):
            sys.stdout.flush()        
            random_indices=self.get_value_coordinates(times[0,0])[0]
            i,j=torch.clamp(random_indices,p,s-p).tolist()
            print(f"\r Generation {times.float().mean()*100/(self.sampling_steps-1):.2f}%, indices=[{(i-p)*8:04d}:{(i+p)*8:04d},{(j-p)*8:04d}:{(j+p)*8:04d}]",end="")
            
            sub_img=self.random_crop(img0, i, j)
            sub_img_stack=self.random_crop(img_stack, i, j)
            sub_time=self.random_crop(times, i, j)
            sub_mask=self.random_crop(masks, i, j, latent=False)
            
            if sub_time.float().mean()!=(self.sampling_steps-1):
                sub_img=self.sample_one(sub_img, sub_img_stack, sub_mask, sub_time)

                mask_changed=torch.where(sub_time==sub_time.min(), 1 ,0)

                img0[...,i-p:i+p,j-p:j+p]=sub_img
                img_stack[:,sub_time.min()+1,:,i-p:i+p,j-p:j+p]=sub_img*mask_changed+sub_img_stack[:,sub_time.min()+1]*(mask_changed==0)
                times[...,i-p:i+p,j-p:j+p]=torch.where(sub_time==sub_time.min(), sub_time+1, sub_time)
            
        return img0
    
# Helpers
    
def tile(x, p):
    B, C, H, W = x.shape
    x_tiled = x.unfold(2, p, p).unfold(3, p, p)
    x_tiled = x_tiled.reshape(B, C, -1, p, p)
    x_tiled = x_tiled.permute(0, 2, 1, 3, 4).contiguous()
    x_tiled = x_tiled.reshape(-1, C, p, p)
    return x_tiled

def untile(x_tiled, original_shape, p):
    B, C, H, W = original_shape
    H_p = H // p
    W_p = W // p
    x_tiled=x_tiled.reshape(B, (H//p*W//p), C, p,p).permute(0,2,1,3,4).reshape(B,C,H//p,W//p,p,p)
    x_untiled=x_tiled.permute(0,1,2,4,3,5).reshape(B,C,H,W)
    return x_untiled

def save_tensor_as_png(tensor, filename):
    # Convert the tensor to a NumPy array
    tensor_np = tensor.permute(1,2,0).cpu().numpy()

    # Convert the NumPy array to an 8-bit unsigned integer array
    tensor_uint8 = (tensor_np * 255).astype(np.uint8)

    # Create a PIL image from the 8-bit unsigned integer array
    image = Image.fromarray(tensor_uint8)

    # Save the image as a PNG file
    image.save(filename, format='PNG', optimize=True, compress_level=9)
    
def corners(subwindows: list):
    (w_upleft,w_upright,w_downright,w_downleft)=subwindows
    window=torch.ones_like(w_upleft)
    size=window.shape[0]
    window[:size//2,:size//2]=w_upleft[:size//2,:size//2]
    window[:size//2,size//2:]=w_upright[:size//2,size//2:]
    window[size//2:,size//2:]=w_downright[size//2:,size//2:]
    window[size//2:,:size//2]=w_downleft[size//2:,:size//2]
    return window

def hann_window(size=512):
    i = torch.arange(size, dtype=torch.float)
    w = 0.5*(1 - torch.cos(2*torch.pi*i/(size-1)))
    window_center=w[:,None]*w
    window_up=torch.where(torch.arange(size)[:,None] < size//2, w, w[:,None]*w)
    window_down=torch.where(torch.arange(size)[:,None] > size//2, w, w[:,None]*w)
    window_right=torch.where(torch.arange(size) > size//2, w[:,None], w[:,None]*w)
    window_left=torch.where(torch.arange(size) < size//2, w[:,None], w[:,None]*w)
    window_upleft=corners([torch.ones((size,size)), window_up, window_center,window_left])
    window_upright=corners([window_up, torch.ones((size,size)), window_right, window_center])
    window_downright=corners([window_center, window_right,torch.ones((size,size)), window_down])
    window_downleft=corners([window_left, window_center, window_down, torch.ones((size,size))])
    
    window_rightright=corners([torch.ones((size,size)), window_up, window_down, torch.ones((size,size))])
    window_leftleft=corners([window_up, torch.ones((size,size)), torch.ones((size,size)), window_down])
    window_upup=corners([window_left, window_right, torch.ones((size,size)), torch.ones((size,size))])
    window_downdown=corners([torch.ones((size,size)), torch.ones((size,size)), window_right, window_left])

    window_vertical=corners([window_up, window_up, window_down, window_down])
    window_horizontal=corners([window_left, window_right, window_right, window_left])
    
    return {'up-left': window_upleft, 'up': window_up, 'up-right': window_upright, 
            'left': window_left, 'center': window_center,  'right': window_right, 
            'down-left': window_downleft, 'down': window_down, 'down-right': window_downright,
           'up-up': window_upup, 'down-down': window_downdown, 
            'left-left': window_leftleft, 'right-right': window_rightright,
           'vertical': window_vertical, 'horizontal': window_horizontal}

def hann_tensors(t1,t2, windows):
    n_windows=len(windows.keys())-6
    t_mix=torch.zeros_like(t1).to(t1.device)
    for key in windows.keys()[:9]:
        t_mix=t1+windows[key].repeat(1,3,1,1)*t2
    return t_mix/n_windows

