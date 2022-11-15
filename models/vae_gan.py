import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class VAE_GAN(nn.Module):
    
    def __init__(self,  config):
        
        super(VAE_GAN, self).__init__()
        
        self.encoder = config['encoder']
        self.decoder = config['decoder']
        self.discriminator = config['discriminator']
        
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

    def forward(self, x):

        enc = self.encoder(x).view(x.size(0), -1)
        latent = enc.size(1)//2
        mu = enc[:, :latent]
        log_var = enc[:, latent:]

        std = log_var.mul(0.5).exp_()

        # sampling epsilon from normal distribution
        epsilon = torch.randn_like(std)
        z = mu+std*epsilon
        z = z[:,:,None,None]
        x_ = self.decoder(z)

        return x_, mu, log_var
    
    def decode(self, z):
        
        z = z[:,:,None,None]
        return self.decoder(z) 
    
    def reconstruct_img(self, x):
        
        enc = self.encoder(x).view(x.size(0), -1)
        latent = enc.size(1)//2
        mu = enc[:, :latent]
        
        return self.decode(mu)

            