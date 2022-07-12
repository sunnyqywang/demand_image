import torch
import torch.nn as nn
import torch.nn.functional as F


class LR(nn.Module):
    def __init__(self, dim_embed, dim_demo, demo_weights=None, image_weights=None):
        super(LR, self).__init__()

        if dim_embed != 0:
            self.embed = nn.Linear(dim_embed, 1)
            if image_weights is not None:
                self.embed.weight = torch.nn.Parameter(image_weights['embed.weight'])
                self.embed.bias = torch.nn.Parameter(image_weights['embed.bias'])
                
        if dim_demo != 0:
            self.demo = nn.Linear(dim_demo, 1)
            if demo_weights is not None:
                self.demo.weight = torch.nn.Parameter(demo_weights['demo.weight'])
                self.demo.bias = torch.nn.Parameter(demo_weights['demo.bias'])

        self.dim_embed = dim_embed
        self.dim_demo = dim_demo
        
    def forward(self, x_embed=None, x_demo=None):
        
        if self.dim_embed != 0:
            out1 = self.embed(x_embed)
        if self.dim_demo != 0:
            out2 = self.demo(x_demo)
        
        if (self.dim_embed != 0) & (self.dim_demo != 0):
#             print(out1, out2)
            return out1+out2
        elif self.dim_embed == 0:
            return out2
        elif self.dim_demo == 0:
            return out1
        else:
            return None
        
    
        