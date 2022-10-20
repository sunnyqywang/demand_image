import sys
sys.path.append("../")

# import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_model import log_cosh_loss

# import visdom
# from util_plot import plot_vae_samples

# import pyro
# import pyro.distributions as dist
# from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
# from pyro.optim import Adam

# from resnext_mnist import Encoder, Decoder

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
# class VAE_Encoder(Encoder):
#     def __init__(self, config):
#         super().__init__(config)
#         self.output_dim = config['model_config']['output_dim']
#         if 'latent_dim' in config['model_config'].keys():
#             self.z_dim = config['model_config']['latent_dim']
#             self.fz1 = nn.Linear(self.output_dim*self.output_dim*2048, self.z_dim)
#             self.fz2 = nn.Linear(self.output_dim*self.output_dim*2048, self.z_dim)  
        
#         else:
#             self.z_dim = self.output_dim**2*2048
#             self.fz1 = nn.Linear(self.z_dim, self.z_dim)
#             self.fz2 = nn.Linear(self.z_dim, self.z_dim)  
        
        
#     def forward(self, x):
#         x = self._forward_conv(x)
# #         print("x:", torch.max(x), torch.min(x))
#         # then return a mean vector and a (positive) square root covariance
#         # each of size batch_size x z_dim
#         x = torch.flatten(x, start_dim=1)
# #         x = F.softplus(x)
#         z_loc = self.fz1(x)
#         z_scale = self.fz2(x)
        
#         #F.relu(self.fz2(x))+0.1
# #         print("loc:", torch.max(z_loc), torch.min(z_loc))
# #         print("scale:", torch.max(z_scale), torch.min(z_scale))
#         z_scale = F.softplus(z_scale)
# #         print("scale2:", torch.max(z_scale), torch.min(z_scale))
        
# #         z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
# #         z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
#         return z_loc, z_scale


# # define the PyTorch module that parameterizes the
# # observation likelihood p(x|z)
# class VAE_Decoder(Decoder):
#     def __init__(self, config):
#         super().__init__(config)
#         self.output_dim = config['model_config']['output_dim']
#         if 'latent_dim' in config['model_config'].keys():
#             self.compress = True
#             self.z_dim = config['model_config']['latent_dim']
#             self.fz1 = nn.Linear(self.z_dim, config['model_config']['output_dim']**2*2048)

#         else:
#             self.compress = False
#             self.z_dim = config['model_config']['output_dim']**2*2048
        
#         self.conv_shape = config['model_config']['conv_shape']
#         self.sigmoid = nn.Sigmoid()
#         self.bn = nn.BatchNorm2d(config['data_config']['color_channels'])

#     def forward(self, x):
        
#         # Cast input shape to ResNeXt encoder output shape
#         batch_size = x.size(0)
#         if self.compress:
#             x = self.fz1(x)
#         x = x.view(batch_size, 2048, self.output_dim, self.output_dim)
        
#         # ResNeXt decoder
#         x = F.interpolate(x, size=self.conv_shape, align_corners=False, mode='bilinear') 
# #         print(x.shape)
#         x = self._forward_conv(x)

# #         print(x.shape)
#         x = F.interpolate(x, scale_factor=2, align_corners=False, mode='bilinear')
# #         print(x.shape)
#         x = self.conv(x)

#         # BW images [0,1]
#         x = self.sigmoid(x)
        
#         # colored, normalized images
# #         x = self.bn(x)
        
# #         out_dim = x.shape[-1]
# #         trim = 2
# #         x = x[:,:,trim:-trim,trim:-trim]
        
#         return x

class VAE(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = config['encoder']
        self.decoder = config['decoder']
        
        self.loss_func = config['model_config']['loss_func']
        
    def encode(self, x):
        
        enc = self.encoder(x).view(x.size(0), -1)
        latent = enc.size(1)//2
        mu = enc[:, :latent]

        return mu
    
    def decode(self, z):
        
        z = z[:,:,None,None]
#         print(z.shape)
        return self.decoder(z)
    
    def reparameterize(self, mu, log_var):
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return eps * std + mu
    
    def forward(self, x):
        
        enc = self.encoder(x).view(x.size(0), -1)
        latent = enc.size(1)//2
        mu = enc[:, :latent]
        log_var = enc[:, latent:]
        
        z = self.reparameterize(mu, log_var)
        x_ = self.decode(z)

        return x_, mu, log_var
    
    def loss_function(self, recons, original, mu, log_var, kld_weight=1):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        if self.loss_func == 'mse':
            recons_loss = F.mse_loss(recons, original, reduction='sum').div(len(original))
        elif self.loss_func == 'cosh':
            recons_loss = log_cosh_loss(recons, original, reduction='sum').div(len(original))
            
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}
    
    def reconstruct_img(self, x):
        
        return self.forward(x)[0]
        