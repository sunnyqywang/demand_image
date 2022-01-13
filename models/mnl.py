import torch
import torch.nn as nn

class MNL(nn.Module):
    def __init__(self, n_alts, n_features, n_alt_specific=0):
        super(MNL, self).__init__()
        self.beta = nn.Parameter(torch.rand(n_alts-1, n_features))
        if n_alt_specific > 0:
            self.beta_ref = nn.Parameter(torch.rand(n_alt_specific))
        self.n_alt_specific = n_alt_specific
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        V = torch.mul(x[:,1:,:], self.beta)
        V = torch.sum(V, dim=2)
        if self.n_alt_specific > 0:
            V = torch.cat((torch.sum(x[0, -self.n_alt_specific:] * self.beta_ref), V), dim=0)
        else:
            V = torch.cat((torch.zeros((len(x),1)),V), dim=1)
        # print(V.shape)
        # p = self.softmax(V)

        # return p
        return V


