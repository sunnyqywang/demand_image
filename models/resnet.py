import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


class Bottleneck(nn.Module):
    # a class used in ResNet.
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        
    def forward(self, x):
        shortcut = x

        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)
        out = self.conv1(x)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)

        out = self.bn1(out)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)

        out = self.relu(out)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)

        out = self.conv2(out)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)
        out = self.bn2(out)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)
        out = self.relu(out)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)

        out = self.conv3(out)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)
        out = self.bn3(out)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)
        out = self.relu(out)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)

        if self.downsample is not None:
            shortcut = self.downsample(x) # this downsample replaces shortcut.

        out += shortcut
        out = self.relu(out)

        return out
    
class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # sw: self.expansion rate. 
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out) # QW: I don't think this line should be here.

        if self.upsample is not None:
            shortcut = self.upsample(x) # sw: similarly, upsample replaces shortcut. 

        out += shortcut
        out = self.relu(out)

        return out
    
    
class ResNet_autoencoder(nn.Module):
    def __init__(self, downblock, upblock, num_layers):#, n_classes):
        super(ResNet_autoencoder, self).__init__()

        # Q: Why is this named as in_channels, when the actual in_channels number = 7?
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) # 3 channels for rgb
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # downsampling
        self.layer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.layer2 = self._make_downlayer(downblock, 128, num_layers[1],stride=2)
        self.layer3 = self._make_downlayer(downblock, 256, num_layers[2],stride=2)
        self.layer4 = self._make_downlayer(downblock, 512, num_layers[3],stride=2)
        
        # Q: Is this block (self.fc and self.bn2) used in the autoencoder architecture? 
        # A: No. It is used for the prediction part.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Q: What is the downblock.expansion? Default value = 4. 
        # A: Need to use this expansion to adjust the input dimension. 
        # self.fc = nn.Linear(512 * downblock.expansion, n_classes)
        # Q: n_classes are not used?
        # A: It is used.
        self.bn2=nn.BatchNorm1d(1)
        # C: I may want to add another linear module to make the structure symmetric.

        # upsampling
        self.uplayer1 = self._make_up_block(upblock, 512, num_layers[3], stride=2)
        self.uplayer2 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(upblock, 64, num_layers[0], stride=2)

        # Q: Was this upsample thing used? 
        # A: I see. Right below in self.uplayer_top 
        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        
        self.uplayer_top = DeconvBottleneck(
            self.in_channels, 64, 1, 2, upsample)

        self.conv1_1 = nn.ConvTranspose2d(64, 3, kernel_size=1, stride=1,
                                          bias=False)

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels * block.expansion),
            )
        layers = []
        layers.append(
            block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels * 2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels * 2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(
            block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def encoder(self, x):
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)
        x = self.conv1(x)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)

        x = self.bn1(x)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)

        x = self.relu(x)
        print(torch.cuda.memory_allocated('cuda:1')/1024//1024, torch.cuda.memory_reserved('cuda:1')/1024//1024)
        x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        return x

    def decoder(self, x, image_size):
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_top(x)
        # Q: where was the image_size defined? 
        # A: Not really defined. conv1_1 inherits nn.ConvTranspose2d, which has this output_size in the documentation's example.
        x = self.conv1_1(x, output_size=image_size)
        return x
    
#     def classifer(self,x): 
#         # Q: Was it not in the original autoencoder model? 
#         # A: Not really. 
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
# #         x=self.bn2(x)
# #         x= torch.sigmoid(x)
#         return x

    def forward(self, x):
        img = x
        latten_var = self.encoder(x)
        # Q: again, why do we need this img.size()?
        # A: W
        x_tilde = self.decoder(latten_var, img.size())
#         output=self.classifer(tmp1)
        return x_tilde
    
