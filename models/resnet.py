# https://github.com/arnaghosh/Auto-Encoder/blob/master/resnet.py

import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models,transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import os
import matplotlib.pyplot as plt 
from torch.autograd import Function
from collections import OrderedDict
import torch.nn as nn
import math

# zsize = 48
# batch_size = 11
# iterations =  500
# learningRate= 0.0001

import torchvision.models as models

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
	
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

###############################################################



###############################################################
class Encoder(nn.Module):

    def __init__(self, config):
        block = Bottleneck
        layers = config['model_config']['layers']
        latent_dim = config['model_config']['latent_dim']
        image_size = config['data_config']['image_size']
        
        self.inplanes = 64
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(image_size//32, stride=1)
        self.fc = nn.Linear(512 * block.expansion, latent_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
	
        x = self.bn1(x)
        x = self.relu(x)
	
        x = self.maxpool(x)
	
        x = self.layer1(x)
#         print(x.size())
        x = self.layer2(x)
#         print(x.size())
        x = self.layer3(x)
#         print(x.size())
        x = self.layer4(x)
#         print(x.size())
        
        x = self.avgpool(x)
#         print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
    

##########################################################################
class Binary(Function):

    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

binary = Binary()
##########################################################################
class Decoder(nn.Module):
	def __init__(self, config):
		super(Decoder,self).__init__()
		latent_dim = config['model_config']['latent_dim']
		self.image_size = config['data_config']['image_size']
		self.im_norm = config['data_config']['im_norm']
        
		block = Bottleneck
        
		self.dfc3 = nn.Linear(latent_dim, block.expansion*512)
		self.bn3 = nn.BatchNorm2d(4096)
		self.dfc2 = nn.Linear(4096, 4096)
		self.bn2 = nn.BatchNorm2d(4096)
		self.dfc1 = nn.Linear(latent_dim, 2048)
		self.bn1 = nn.BatchNorm2d(256*6*6)
		self.upsample1=nn.Upsample(scale_factor=2)
		self.dconv5 = nn.ConvTranspose2d(2048, 1024, 2, padding = 0)
		self.dconv4 = nn.ConvTranspose2d(1024, 512, 3, padding = 1)
		self.dconv3 = nn.ConvTranspose2d(512, 256, 3, padding = 1)
		self.dconv2 = nn.ConvTranspose2d(256, 64, 5, padding = 2)
		self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)

	def forward(self,x):#,i1,i2,i3):
		
# 		x = self.dfc3(x)
# 		#x = F.relu(x)
# 		x = F.relu(self.bn3(x))
# 		x = self.dfc2(x)
# 		x = F.relu(self.bn2(x))
# 		#x = F.relu(x)
		x = self.dfc1(x)
# 		x = F.relu(self.bn1(x))
		#x = F.relu(x)
		batch_size = len(x)
		x = x.view(batch_size, 2048, 1, 1)
        
# 		print(x.size())
		x = F.relu(self.dconv5(x))
# 		print(x.size())
		x=self.upsample1(x)
        
# 		print(x.size())
		x = F.relu(self.dconv4(x))
		x=self.upsample1(x)
        
# 		print(x.size())		
		x = F.relu(self.dconv3(x))
		x=self.upsample1(x)
        
# 		print(x.size())		
		x = F.relu(self.dconv2(x))
# 		x=self.upsample1(x)
        
# 		print(x.size())
		x = self.dconv1(x)
		if self.im_norm == False:
			x = F.sigmoid(x)
		#print x
		return x
