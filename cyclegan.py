import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

from torch.utils.data import Dataset
from PIL import Image



###############################################
# Residual block with two convolutional layers
###############################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(
                1
            ), # Reflection padding is used because it gives better image quality at edges
            nn.Conv2d(
                in_channel, in_channel, 3
            ), # Paper says - same number of filters on both layers
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel)
        )

        def forward(self, x):
            return x + self.block(x)

###############################################
# Generator
###############################################

"""
As per Paper -- Generator with 9 residual blocks consists of:
c7s1-64, d128, d256, R256, R256, R256,R256, R256, R256,R256, R256, R256,
u128, u64, c7s1-3
"""

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolutional block
        out_channels = 64
        # I define a variable 'model' which I will continue to update
        # throughout the 3 blocks of Residual -> Downsampling -> Upsampling

        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        in_channels = out_channels

        # Downsampling

        for i in range(2):
            out_channels *= 2
            model += [
                nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels

        # Residual Blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_channels)]

        # Upsampling
        # u128 -> u64
        for _ in range(2):
            out_channels //=2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels,kernel_size=3,stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]

            in_channels = out_channels

        # Output Layer
        # c7s1-3
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_channels, channels,kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


###############################################
# Discriminator
###############################################

"""
We use 70 x 70 PatchGAN. Let Ck denote a 4 x4 Convolution-InstanceNorm-LeakyRelU layer with k filters and stride 3
After the last layer, we apply a convolution to produce a 1-dimensional output
We do not use InstanceNorm for the first C64 layer
We use LeakyReLUs with a slope of 0.2. The discriminaor architecture is: C64-C128-C256-C512
"""
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator,self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 **4, width //2 **4)

        def discriminator_block(in_channels, out_channels, normalize=True):
            """Return downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        # C64 -> C128 -> C256 -> C512 called PatchGAN
        self.model = nn.Sequential(
            *discriminator_block(channels, out_channels=64, normalize=False),
            *discriminator_block(in_channels=64, out_channels=128),
            *discriminator_block(in_channels=128, out_channels=256),
            *discriminator_block(in_channels=256, out_channels=512),
            nn.ZeroPad2d(1, 0, 1, 0),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)
        )

        def forward(self,img):
            return self.model(img)

