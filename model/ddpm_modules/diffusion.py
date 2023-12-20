import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import cv2
from utils import *


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# def noise_like(shape, device, repeat=False):
#     def repeat_noise(): return torch.randn(
#         (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

#     def noise(): return torch.randn(shape, device=device)
#     return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        global_corrector=None,
        sr=False
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)
        if global_corrector is not None:
            self.global_corrector = global_corrector
        self.sr=sr


    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()


    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        if schedule_opt['ddim_timestep']:
            self.ddim_timesteps = int(schedule_opt['ddim_timestep'])
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance


    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )


    # DDPM sample
    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))

        ir = x_in.shape[1] !=3
        if ir:
            ret_img = torch.cat([x_in[:,:3,:,:], x_in[:,3,:,:].repeat(1,3,1,1)], dim=0)
        else:
            ret_img = x_in[:,:3,:,:]        
        
        shape = x_in[:,:3,:,:].shape
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)            

            pred_noise = self.denoise_fn(torch.cat([x_in, img], dim=1),t)
            x_start = self.predict_start_from_noise(img, t=t, noise=pred_noise)
            x_start.clamp_(-1., 1.)

            x_start = self.global_corrector(x_start, t)
            x_start.clamp_(-1., 1.)

            if i==0:
                img = x_start
                if self.sr:
                    ret_img = torch.cat([ret_img, F.interpolate(img, scale_factor=0.5, mode='bilinear')], dim=0)
                else:
                    ret_img = torch.cat([ret_img, img], dim=0)
                break

            if self.sr:
                x_start = F.interpolate(x_start, scale_factor=0.5, mode='bicubic')
                x_start.clamp_(-1., 1.)

            # calculate x(t-1)
            model_mean = (extract(self.posterior_mean_coef1, t, shape) * x_start +
                          extract(self.posterior_mean_coef2, t, shape) * img
                          )

            model_log_variance = extract(self.posterior_log_variance_clipped, t, shape)

            # no noise when t == 0
            noise = torch.randn(shape, device=device)
            nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                          *((1,) * (len(shape) - 1)))
            img =  model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise            

            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
                ret_img = torch.cat([ret_img, x_start], dim=0)

                # save_img(tensor2img(x_start), f'./{i}.png')

        if continous:
            return ret_img
        else:
            return img


    @torch.no_grad()
    def ddim_sample(self, x_in, continuous=False):
        batch, device, total_timesteps, sampling_timesteps = x_in.shape[0], self.betas.device, self.num_timesteps, self.ddim_timesteps
        eta = 0 # control the noise injected to DDIM

        sample_inter = (1 | (sampling_timesteps//10))

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        ir = x_in.shape[1] !=3
        if ir:
            ret_img = torch.cat([x_in[:,:3,:,:], x_in[:,3,:,:].repeat(1,3,1,1)], dim=0)
        else:
            ret_img = x_in[:,:3,:,:]

        img = torch.randn(x_in[:,:3,:,:].shape, device = device)

        i=1
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            t = torch.full((batch,), time, device = device, dtype = torch.long)
            
            pred_noise = self.denoise_fn(torch.cat([x_in, img], dim=1),t)
            x_start = self.predict_start_from_noise(img, t=t, noise=pred_noise)
            x_start.clamp_(-1., 1.)

            x_start = self.global_corrector(x_start, t)
            x_start.clamp_(-1., 1.)

            if time_next < 0:
                img = x_start
                if self.sr:
                    ret_img = torch.cat([ret_img, F.interpolate(img, scale_factor=0.5, mode='bicubic')], dim=0)
                else:
                    ret_img = torch.cat([ret_img, img], dim=0)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            if self.sr:
                x_start = F.interpolate(x_start, scale_factor=0.5, mode='bicubic')
                x_start.clamp_(-1., 1.)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            i += 1
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
                ret_img = torch.cat([ret_img, x_start], dim=0)
        
        if continuous:
            return ret_img
        else:
            return img


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gamma
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )


    def p_losses(self, x_in, noise=None):
        x_gt = x_in['HR']   # For GC loss
        x_start = F.interpolate(x_in['HR'], scale_factor=0.5, mode='bicubic') if self.sr else x_in['HR']

        [b, c, h, w] = x_start.shape    # [B, 3, H, W]

        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))   # [B, 3, H, W]
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # [B, 3, H, W]

        if 'IR' in x_in:
            predicted_noise = self.denoise_fn(                  # [B, 3, H, W]
                torch.cat([x_in['LR'], x_in['IR'], x_noisy], dim=1), t
            )
        else:
            predicted_noise = self.denoise_fn(
                torch.cat([x_in['LR'], x_noisy], dim=1), t
                )         
        
        # Sampled noise vs UNet predicted noises
        l_noise = self.loss_func(noise, predicted_noise)
        
        x_recon = self.predict_start_from_noise(
            x_noisy, t=t, noise=predicted_noise.detach())          
        x_recon.clamp_(-1., 1.)
        x_recon = self.global_corrector(x_recon, t)
        x_recon.clamp_(-1., 1.)

        # Reconstructed img vs gt hi-res img
        l_recon = self.loss_func(x_recon, x_gt)

        return l_noise, l_recon


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)