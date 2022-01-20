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


