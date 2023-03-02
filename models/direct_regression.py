import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectRegression(nn.Module):
    def __init__(self, config):
        super(DirectRegression, self).__init__()
        
        self.encoder = config['encoder']
        self.encoder_output_dim = config['model_config']['latent_dim']
#         self.fc1 = nn.Linear(self.encoder_output_dim, self.encoder_output_dim//4)
#         self.fc2 = nn.Linear(self.encoder_output_dim//4, self.encoder_output_dim//16)
#         self.fc3 = nn.Linear(self.encoder_output_dim//16, config['model_config']['n_alts'])
        self.fc = nn.Linear(self.encoder_output_dim, config['data_config']['demo_channels'])
#         self.bn = nn.BatchNorm1d(self.encoder_output_dim)
#         self.demo_out = nn.Tanh()
        self.demo_out = nn.Sigmoid()
#         self.dropout = nn.Dropout(p=config['model_config']['dropout'])
        
    def forward(self, x):
#         batch_size,num_im, _,_,_ = x.shape
                
        xd = self.encoder(x).squeeze()
        xd = self.demo_out(self.fc(xd))
        
        return xd

class Supervised_Demo(nn.Module):
    def __init__(self, config):
        super(Supervised_Demo, self).__init__()
        
        self.encoder = config['encoder']
        self.fc1 = nn.Linear(config['model_config']['output_dim']**2*2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, config['model_config']['num_demo_vars'])
        if config['data_config']['demo_norm'] == 'minmax':
            self.demo_out = nn.Sigmoid()
        else:
            self.demo_out = nn.BatchNorm1d(config['model_config']['num_demo_vars'])
        self.float()
        
    def forward(self, x, xp=None):
        if xp is None:
            xp = self.encoder(x) 
        
        xd = self.demo_out(self.fc3(F.relu(self.fc2(F.relu(self.fc1(xp.view(len(xp), -1)))))))
        
        return xd.float()
