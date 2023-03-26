import argparse
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        
        if config['model_config']['model_class'] == 'vae':
            nz = config['model_config']['latent_dim']//2 
        else:
            nz = config['model_config']['latent_dim']
        
#         if config['data_config']['demo_channels'] > 0:
#             self.conditional = True
#             self.mapping = nn.Sequential(
#                 nn.Linear(config['data_config']['demo_channels'], 64),
#                 nn.ReLU(True),
#                 nn.Linear(64, nz)
#             )
#             nz = nz * 2
#         else:
#             self.conditional = False
            
        ngf = config['model_config']['base_channels']
        nc = config['data_config']['color_channels']
        self.im_norm = config['data_config']['im_norm']

        image_size = config['data_config']['image_size']
        num_layers = int(math.log2(image_size)) - 3
        n_channels = ngf * 2 ** (num_layers)
        
        layers = [nn.ConvTranspose2d(nz, n_channels, 4, 1, 0, bias=False),
                  nn.BatchNorm2d(n_channels),
                  nn.ReLU(True)]
        for i in range(1,num_layers+1):
            layers.append(nn.ConvTranspose2d(n_channels, n_channels//2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels//2))
            layers.append(nn.ReLU(True))
            n_channels = n_channels//2
        
        layers.append(nn.ConvTranspose2d(n_channels, nc, 4, 2, 1, bias=False))
        
        self.main = nn.Sequential(*layers)
       
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
# #             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
        
        self.apply(weights_init)

    def forward(self, x, demo=None):
#         if demo is not None:
#             demo = self.mapping(demo)
#             x = torch.cat((x, demo), dim=1)
                
        x_ = self.main(x)
        
        if self.im_norm == 0:
            x_ = torch.sigmoid(x_)
        elif self.im_norm == 2:
            x_ = torch.tanh(x_)

        return x_
    
    
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        
        ndf = config['model_config']['base_channels']
        self.model_class = config['model_config']['model_class']
            
        if self.model_class == 'gan':
            nc = config['data_config']['color_channels'] + config['data_config']['demo_channels']
        else:
            nc = config['data_config']['color_channels']
        nz = config['model_config']['latent_dim']

        image_size = config['data_config']['image_size']
        num_layers = int(math.log2(image_size)) - 3
        
        n_channels = ndf
        
        layers = [nn.Conv2d(nc, n_channels, 4, 2, 1, bias=False),
                  nn.LeakyReLU(0.2, inplace=True)]
        for i in range(1,num_layers+1):
            layers.append(nn.Conv2d(n_channels, n_channels*2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels*2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            n_channels = n_channels * 2
        
        if self.model_class == 'gan':
            layers.append(nn.Conv2d(n_channels, 1, 4, 1, 0, bias=False))
        else:
            assert nz <= ndf*(2**(num_layers))
            layers.append(nn.Conv2d(n_channels, nz, 4, 1, 0, bias=False))
            
        self.main = nn.Sequential(*layers)
        
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#         )
        
        
        if self.model_class == 'gan':
#             self.classification = nn.Sequential(
#                 nn.Conv2d(nz, 1, 1, 1, 0, bias=False),
#                 nn.Sigmoid()
#             )
            self.classification = nn.Sigmoid()
        
        self.apply(weights_init)

    def forward(self, x):
        
        x = self.main(x)
        if self.model_class == 'gan':
            x = torch.squeeze(self.classification(x))

        elif self.model_class == 'StylEx':
            x = torch.squeeze(x)

        return x
        
    def return_latent(self, x):
        
        x = self.main(x)
        
        return x
    
