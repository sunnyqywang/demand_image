import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()
        
        self.encoder = config['encoder']
        self.decoder = config['decoder']
        self.fc1 = nn.Linear(2048*config['model_config']['output_dim']**2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, config['model_config']['num_demo_vars'])
        if config['data_config']['demo_norm'] == 'minmax':
            self.demo_out = nn.Sigmoid()
        else:
            self.demo_out = nn.BatchNorm1d(config['model_config']['num_demo_vars'])
            
    def forward(self, x):
        xp = self.encoder(x) 
        xd = self.demo_out(self.fc3(F.relu(self.fc2(F.relu(self.fc1(xp.view(len(xp), -1)))))))
        xp = self.decoder(xp)
        
        return xp, xd

class Autoencoder_raw(nn.Module):
    def __init__(self, config):
        super(Autoencoder_raw, self).__init__()
        
        self.encoder = config['encoder']
        self.decoder = config['decoder']

    def forward(self, x):
        xp = self.encoder(x) 
        xp = self.decoder(xp)
        
        return xp

    
class Autoencoder_adv(nn.Module):
    def __init__(self, config):
        super(Autoencoder_adv, self).__init__()
        
        self.encoder = config['encoder']
        self.decoder = config['decoder']
        self.latent_dim = 2048*config['model_config']['output_dim']**2
        
        self.fc1 = nn.Linear(int(self.latent_dim/2), config['model_config']['num_demo_vars'])
        self.bn1 = nn.BatchNorm1d(config['model_config']['num_demo_vars'])
        
        self.fc1_adv = nn.Linear(int(self.latent_dim/2), config['model_config']['num_demo_vars'])
        self.bn2 = nn.BatchNorm1d(config['model_config']['num_demo_vars'])
        
    def forward(self, x):
        xp = self.encoder(x) 
        latent = xp.view(len(xp), self.latent_dim)
        
        xd = self.bn1(self.fc1(latent[:, :int(self.latent_dim/2)]))
        xd_adv = self.bn2(self.fc1_adv(latent[:, int(self.latent_dim/2):]))
        
        xp = self.decoder(xp)
        
        return xp, xd, xd_adv
