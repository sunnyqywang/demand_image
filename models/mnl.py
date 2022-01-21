import torch
import torch.nn as nn
import torch.nn.functional as F

class MNL(nn.Module):
    def __init__(self, n_alts, n_features, n_alt_specific=0):
        super(MNL, self).__init__()
        # The first alternative is the reference
#         self.beta = nn.Linear(n_features, n_alts-1) 
        self.beta = nn.Linear(n_features, n_alts)

        if n_alt_specific > 0:
            self.beta_ref = nn.Parameter(torch.rand(n_alt_specific))
        self.n_alt_specific = n_alt_specific

    def forward(self, x, x_alt_specific=None):
        # x ~ (batch_size, n_features)
        
        if self.n_alt_specific == 0:
            assert x_alt_specific is None
            V = self.beta(x)
#             V = torch.cat((torch.zeros((len(x),1)),V), dim=1)
        else:
            assert x_alt_specific is not None
            pass
            # not yet implemented
#             V = torch.mul(x[:,1:,:], self.beta)
#             V = torch.sum(V, dim=2)
#             V = torch.cat((torch.sum(x[0, -self.n_alt_specific:] * self.beta_ref), V), dim=0)

        return V #F.relu(V)



class MNL2(nn.Module):
    def __init__(self, n_alts, dim_embed, dim_demo, dim_hidden=128):
        super(MNL2, self).__init__()
        self.dim_embed = dim_embed
        self.dim_demo = dim_demo
        
#         self.bn_e = nn.BatchNorm1d(dim_embed)
        self.bn_d = nn.BatchNorm1d(dim_demo)
#         self.bn = nn.BatchNorm1d(dim_embed+dim_demo)
    
        self.fc_embedding1 = nn.Linear(dim_embed, dim_hidden)
        self.fc_embedding2 = nn.Linear(dim_hidden, dim_demo)

        self.fc_demo1 = nn.Linear(dim_demo, dim_demo)
        self.fc_demo2 = nn.Linear(dim_demo, dim_demo)

        self.mnl = nn.Linear(2*dim_demo, n_alts)
        
    def forward(self, x):
        
        assert x.shape[1] == self.dim_embed + self.dim_demo
#         x = self.bn(x)
        
        demo = x[:, :self.dim_demo]
        embedding = x[:, self.dim_demo:]
        
        demo = self.bn_d(demo)
#         embedding = self.bn_e(embedding)
        
        embedding = F.relu(self.fc_embedding1(embedding))
        embedding = self.fc_embedding2(embedding)
        
        demo = F.relu(self.fc_demo1(demo))
        demo = self.fc_demo2(demo)
        
        V = self.mnl(torch.concat([demo, embedding], dim=1))
        
        return V
    