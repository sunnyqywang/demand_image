import torch
from torch import nn, optim
import torch.nn.functional as F
from math import log2

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        
class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
    
    
class GHFeat_ResNet_Enc(nn.Module):

    def __init__(self, image_size=64, num_Blocks=[2,2,2,2,2], z_dim=10, nc=3):
        
        super().__init__()
        self.in_planes = 2
        self.encoder_dim = 512
        self.num_layers = int(log2(image_size) - 1)

        self.filters = [2,8,32,128,512]
        self.conv1 = nn.Conv2d(nc, 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.layers = [self._make_layer(BasicBlockEnc, self.filters[0], num_Blocks[0], stride=1)]
        for i in range(self.num_layers-1):
            self.layers.append(self._make_layer(BasicBlockEnc, self.filters[i+1], num_Blocks[i+1], stride=2))
        
        self.layers = nn.Sequential(*self.layers)
        
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
#         self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
#         self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
#         self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.apply(weights_init)
        
    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        out = []
        for layer in self.layers:
            x = layer(x)
#             print(x.shape)
            out.append(F.avg_pool2d(x, 2).view(x.size(0), 1, -1))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)   
#         x = F.adaptive_avg_pool2d(x, 1)
#         x = x.view(x.size(0), -1)

        return torch.cat(out, dim=1)

