import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.openaimodel import UNetModel
#from model.modules import SpatialRescaler
from inspect import isfunction


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class BridgedModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.ana_on = model_config.BB.params.ana_on #new

        self.predicted_noise_list = []  #new

        self.should_log = False  

        self.model_config = model_config

        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y, context=None):
        self.predicted_noise_list = []  
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, t)
    

    def p_losses(self, x0, y, context, t, noise=None):
        """
        # This function is called during every single training step.
        # Unlike the reverse process in sampling, we do not iterate from X_T to X_T-1 to X_0.
        # Instead, a single random timestep t is sampled using torch.randint (from forward function), and the model learns to denoise at that specific timestep.

        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))
        #print('TRAIN 1')
    
        x_t, objective = self.q_sample(x0, y, t, noise) #forward

        # ANA
        if self.ana_on:
            current_pred = self.denoise_fn(x_t, timesteps=t, context=context)
            #print('TRAIN 2')
            self.predicted_noise_list.append(current_pred)

    
            weights = torch.linspace(1, len(self.predicted_noise_list), len(self.predicted_noise_list), device=x_t.device) ## linear decay weights!
         
        
            weights = weights / weights.sum()
            weighted_pred = sum(w * n for w, n in zip(weights, self.predicted_noise_list))
            
            #print('TRAIN 3')
   
            objective_recon = weighted_pred #replace original predicted_noise with weighted result

        else:
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()
        
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        #print('TRAIN 4')
        return recloss, log_dict
    
#p_losses without ANA
    # def p_losses(self, x0, y, context, t, noise=None):
    #     """
    #     model loss
    #     :param x0: encoded x_ori, E(x_ori) = x0
    #     :param y: encoded y_ori, E(y_ori) = y
    #     :param y_ori: original source domain image
    #     :param t: timestep
    #     :param noise: Standard Gaussian Noise
    #     :return: loss
    #     """
    #     b, c, h, w = x0.shape
    #     noise = default(noise, lambda: torch.randn_like(x0))

    #     x_t, objective = self.q_sample(x0, y, t, noise)
    #     objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

    #     if self.loss_type == 'l1':
    #         recloss = (objective - objective_recon).abs().mean()
    #     elif self.loss_type == 'l2':
    #         recloss = F.mse_loss(objective, objective_recon)
    #     else:
    #         raise NotImplementedError()

    #     x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
    #     log_dict = {
    #         "loss": recloss,
    #         "x0_recon": x0_recon
    #     }
    #     return recloss, log_dict

    def q_sample(self, x0, y, t, noise=None):


        if self.should_log:
            print(f"x0 shape: {x0.shape}")  # This logs the shape of x0


        noise = default(noise, lambda: torch.randn_like(x0))
        #self.noise_list.append(noise)  # new


        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        # weighted_noise = torch.zeros_like(noise)      #new
        # for i, n in enumerate(self.noise_list):
        #     weight = 1.0 / (i + 1)  
        #     weighted_noise += weight * n              #new

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise #weighted noise
        elif self.objective == 'noise':
            objective = noise #noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()
        
        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,  #noise
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            
            if self.ana_on:
                #print('1')
                current_pred = self.denoise_fn(x_t, timesteps=t, context=context) #original UNet prediction
                self.predicted_noise_list.append(current_pred)
                #print('2')


                weights = torch.linspace(1, len(self.predicted_noise_list), len(self.predicted_noise_list), device=x_t.device) #linear weights
                weights = weights / weights.sum()
                weighted_pred = sum(w * n for w, n in zip(weights, self.predicted_noise_list))

                objective_recon = weighted_pred
                #print('3')
            else:
                objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            
            #objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)


            #       
            # weighted_noise = torch.zeros_like(x_t)
            # for j, n in enumerate(self.noise_list):
            #     weight = 1.0 / (j + 1)  # 
            #     weighted_noise += weight * n   

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon  #weighted_noise new

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        self.predicted_noise_list = []  
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)