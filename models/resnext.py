from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

class BottleneckBlock(nn.Module):
    # factor of bottleneck
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality):
        super(BottleneckBlock, self).__init__()

        bottleneck_channels = cardinality * out_channels // self.expansion

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels: 
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        # y = F.relu(self.bn1(self.conv1(x)))
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        y = self.conv1(x)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        y = self.bn1(y)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        y = F.relu(y)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        y = self.conv2(y)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        y = self.bn2(y)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        y = F.relu(y)
        # y = F.relu(self.bn2(self.conv2(y)))
        # y = self.bn3(self.conv3(y))  # not apply ReLU
        y = self.conv3(y)
        y = self.bn3(y)

        y += self.shortcut(x)
        y = F.relu(y)  # apply ReLU after addition
        return y

    
class DeconvBottleneckBlock(nn.Module):
    # factor of bottleneck
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality):
        super(DeconvBottleneckBlock, self).__init__()

        bottleneck_channels = cardinality * out_channels // self.expansion

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.ConvTranspose2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  
            padding=1,
            output_padding=stride-1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels: 
            self.shortcut.add_module(
                'conv',
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    output_padding=stride-1,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y)  # apply ReLU after addition
        return y
    
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        input_shape = config['input_shape']

        base_channels = config['base_channels']
        # depth = config['depth']
        self.cardinality = config['cardinality']
        self.output_dim = config['output_dim']

        # 2: initial conv layer + last fc layer
        # 9: 3 layers/block * 3 stages
        # --> 9 * blocks/stage = layers
        # n_blocks_per_stage = (depth - 2) // 9
        # assert n_blocks_per_stage * 9 + 2 == depth

        block = BottleneckBlock
        n_channels = [
            base_channels, base_channels * block.expansion,
            base_channels * 2 * block.expansion,
            base_channels * 4 * block.expansion,
            base_channels * 8 * block.expansion
        ]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(n_channels[0], n_channels[1], 3, stride=1)
        self.stage2 = self._make_stage(n_channels[1], n_channels[2], 4, stride=2)
        self.stage3 = self._make_stage(n_channels[2], n_channels[3], 6, stride=2)
        self.stage4 = self._make_stage(n_channels[3], n_channels[4], 3, stride=2)

        # compute conv feature size
        #with torch.no_grad():
        #    self.feature_size = self.forward(torch.zeros(*input_shape)).shape
        #    print('Encoder', self.feature_size)
        #    print(self.feature_size)

        # self.fc = nn.Linear(reduce(lambda x,y: x*y, self.feature_size[1:]), self.n_features)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                s = stride
            else:
                s = 1
                in_channels = out_channels

            stage.add_module(
                    block_name,
                    BottleneckBlock(
                        in_channels,
                        out_channels,
                        s,  
                        self.cardinality))
        return stage

    def _forward_conv(self, x):
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        # x = F.relu(self.bn(self.conv(x)))
        x = self.conv(x)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        x  = self.bn(x)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        x = F.relu(x)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        x = self.maxpool(x)

        # print(x.shape)
        x = self.stage1(x)
        # print(x.shape)
        x = self.stage2(x)
        # print(x.shape) 
        x = self.stage3(x)
        # print(x.shape)
        x = self.stage4(x)
        # print(x.shape)
        x = F.adaptive_avg_pool2d(x, output_size=self.output_dim)
        # print(x.shape)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x   

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.input_shape = config['input_shape']

        base_channels = config['base_channels']
        # depth = config['depth']
        self.cardinality = config['cardinality']
        self.conv_shape = config['conv_shape']
        # self.hidden_dim = config['hidden_dim']
        # self.conv_dim = reduce(lambda x,y: x*y, self.input_shape)
        # self.fc = nn.Linear(self.hidden_dim, self.conv_dim) 

        # 2: initial conv layer + last fc layer
        # 9: 3 layers/block * 3 stages
        # --> 9 * blocks/stage = layers
        # n_blocks_per_stage = (depth - 2) // 9
        # assert n_blocks_per_stage * 9 + 2 == depth

        block = DeconvBottleneckBlock
        n_channels = [
            base_channels, base_channels * block.expansion,
            base_channels * 2 * block.expansion,
            base_channels * 4 * block.expansion,
            base_channels * 8 * block.expansion
        ]
        n_channels.reverse()

        self.conv = nn.ConvTranspose2d(
            n_channels[-1],
            config['output_channels'],
            kernel_size=6,
            stride=2,
            padding=2,
            bias=False)
        # self.bn = nn.BatchNorm2d(base_channels)
        self.bn = nn.BatchNorm2d(config['output_channels'])

        self.stage1 = self._make_stage(n_channels[0], n_channels[1], 3, stride=2)
        self.stage2 = self._make_stage(n_channels[1], n_channels[2], 6, stride=2)
        self.stage3 = self._make_stage(n_channels[2], n_channels[3], 4, stride=2)
        self.stage4 = self._make_stage(n_channels[3], n_channels[4], 3, stride=1)


        # initialize weights
        self.apply(initialize_weights)

#         with torch.no_grad():
#             self.feature_size = self.forward(
#                 torch.zeros(*self.input_shape)).shape
#             print('Decoder:', self.feature_size)


    def _make_stage(self, in_channels, out_channels, n_blocks, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                s = stride
            else:
                s = 1
                in_channels = out_channels

            stage.add_module(
                block_name,
                DeconvBottleneckBlock(
                    in_channels,
                    out_channels,
                    s,  
                    self.cardinality))
        return stage

    def _forward_conv(self, x):
        x = self.stage1(x)
        # print(x.shape)
        x = self.stage2(x)
        # print(x.shape)
        x = self.stage3(x)
        # print(x.shape)
        x = self.stage4(x)
        # print(x.shape)
        return x

    def forward(self, x):
        conv_shape = self.input_shape
        conv_shape[0] = x.size(0)
        x = x.view(conv_shape)
        x = F.interpolate(x, size=self.conv_shape, align_corners=False, mode='bilinear') 
        x = self._forward_conv(x)
        # x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.interpolate(x, scale_factor=2, align_corners=False, mode='bilinear')
        # print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        # print(x.shape)

        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        y = encoder(x)
        # print(y.shape)
        y = decoder(y)

        return y

