import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectRegression(nn.Module):
    def __init__(self, config):
        super(DirectRegression, self).__init__()
        
        self.encoder = config['encoder']
        self.fc1 = nn.Linear(2048*config['model_config']['output_dim']**2, 1)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        xp = self.encoder(x) 
#         xd = torch.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(xp.view(len(xp), -1)))))))
#         xd = self.fc3(F.relu(self.fc2(F.relu(self.fc1(xp.view(len(xp), -1))))))
        xd = self.fc1(xp.view(len(xp), -1))
    
        return xd
