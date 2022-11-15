import sys
sys.path.append("../")

# import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_model import log_cosh_loss

class VAE(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = config['encoder']
        self.decoder = config['decoder']
        
        self.demo_channels = config['data_config']['demo_channels']
        self.latent_dim = config['model_config']['latent_dim']
        
        self.loss_func = config['model_config']['loss_func']
        
    def encode(self, x):
        
        enc = self.encoder(x).view(x.size(0), -1)
        latent = enc.size(1)//2
        mu = enc[:, :latent]

        return mu
    
    def decode(self, z):
        
        z = z[:,:,None,None]

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
        
        # add conditioning information
        if self.demo_channels != 0:
            demo_conv = x[:,-self.demo_channels:,0,0]
            z = torch.cat([z, demo_conv], 1)
            
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
        
        mu = self.encode(x)
        
        # add conditioning information
        if self.demo_channels != 0:
            demo_conv = x[:,-self.demo_channels:,0,0]
            z = torch.cat([mu, demo_conv], 1)
        
        return self.decode(z)
    