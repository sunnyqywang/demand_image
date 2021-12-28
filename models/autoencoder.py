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

    def forward(self, x):
        xp = self.encoder(x) 
        xd = torch.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(xp.view(len(xp), -1)))))))
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
