import torch
from torch import nn, optim
import torch.nn.functional as F
from math import log2
import torchvision
from collections import OrderedDict 

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

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
    
class BottleneckEnc(nn.Module):
    
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckEnc, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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
    
class GHFeat_Enc(nn.Module):

    def __init__(self, image_size=64, num_Blocks=[3,4,6,3], z_dim=10, nc=3, **fpn_args):
        
        super().__init__()
        self.in_planes = 64
        self.encoder_dim = 512
        self.num_layers = int(log2(image_size) - 1)
        
#         self.filters = [64, 64, 128, 256, 512, 512] ## ResNet18
        self.filters = [64, 64, 128, 256, 512, 512] ## ResNet50

        # stage 1
        self.conv1 = nn.Conv2d(nc, self.filters[0], kernel_size=7, stride=1, padding=6, bias=False)
        self.bn1 = nn.BatchNorm2d(self.filters[0])
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        
        # stage 2-5
        self.layer1 = self._make_layer(BottleneckEnc, self.filters[1], num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BottleneckEnc, self.filters[2], num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BottleneckEnc, self.filters[3], num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BottleneckEnc, self.filters[4], num_Blocks[3], stride=2)
        
        # stage 6
        # add another residual block to obtain lower resolution feature map 
        # not part of resnet but part of ghfeat
        self.layer5 = self._make_layer(BottleneckEnc, self.filters[4], 1, stride=2)

        self.sam_conv = []
        
        # 1801
        self.fc = []
        
        self.start_level = 1
        for i in range(self.start_level, self.num_layers+1):
            self.sam_conv.append(nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=True))
            # 1801
            self.fc.append(nn.Linear(2048, 512))
            
        self.sam_conv = nn.Sequential(*self.sam_conv)
        # 1801
        self.fc = nn.Sequential(*self.fc)
        
        
        self.fpn = torchvision.ops.FeaturePyramidNetwork([self.filters[i]*4 for i in range(self.start_level, self.num_layers+1)], 512)
        self.upscale = nn.Upsample(scale_factor=2)
        self.downsample = nn.AvgPool2d(2)
        self.bn2 = nn.BatchNorm1d(5)
        self.apply(weights_init)
 
# Resnet18
#     def _make_layer(self, block, planes, num_Blocks, stride):
#         strides = [stride] + [1]*(num_Blocks-1)
#         layers = []
#         for stride in strides:
# #             print(self.in_planes)
#             layers += [block(self.in_planes, self.in_planes*stride, stride)]
#             self.in_planes = planes
#         return nn.Sequential(*layers)

# Resnet50
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)    
    
    def sam(self, inputs):
        # recurrent downsample
        for i in range(len(inputs)-1):
            for j in range(0, len(inputs)-1-i):
                inputs[j] = self.downsample(inputs[j])

        for i in range(len(inputs)):
            inputs[i] = self.sam_conv[i](inputs[i])
            
        # latent_fusion
        for i in range(len(inputs)-1):
            inputs[i] = inputs[i] + inputs[-1]

        return inputs
    
    def forward(self, x):
        
        batch_size = len(x)
        
        res1 = self.maxpool(self.leakyrelu(self.bn1(self.conv1(x)))) # 64
#         print("res1",  res1.shape)
        res2 = self.layer1(res1) # 64
#         print("res2",  res2.shape)
        res3 = self.layer2(res2) # 128
#         print("res3", res3.shape)
        res4 = self.layer3(res3) # 256
#         print("res4", res4.shape)
        res5 = self.layer4(res4) # 512  
#         print("res5", res5.shape)
        res6 = self.layer5(res5) # 512
#         print("res6", res6.shape)
       
        inputs = OrderedDict({'feat1':res2, 'feat2':res3, 'feat3':res4, 'feat4':res5, 'feat5':res6})
        
        inputs = self.fpn(inputs)
        inputs = [z for _,z in inputs.items()]

        inputs = self.sam(list(inputs))

        ## 1501
#         inputs = [F.adaptive_avg_pool2d(z, 1) for z in inputs] 
#         return torch.cat(inputs, axis=3).squeeze().permute(0,2,1)

        ## 1801
#         inputs = [self.fc[i](z.view(batch_size,-1))[:,None,:] for i,z in enumerate(inputs)]
#         return torch.cat(inputs, axis=1)


        ## 2801
        inputs = [self.fc[i](z.view(batch_size,-1))[:,None,:] for i,z in enumerate(inputs)]
        inputs = torch.cat(inputs, axis=1)
        
        return self.bn2(inputs)