import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class FiLM(nn.Module):
    
    def __init__(self, n_input, n_output, n_hidden, n_hidden_layers):
        super().__init__()
        nets = []
        nets.append(nn.Linear(n_input, n_hidden))
        nets.append(nn.ReLU())
        for i in range(n_hidden_layers):
            nets.append(nn.Linear(n_hidden, n_hidden))
            nets.append(nn.ReLU())

        nets.append(nn.Linear(n_hidden, n_output))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        out = self.net(x)
        gamma = out[:, 0]
        beta = out[:, 1]
        return gamma, beta


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvFiLM(nn.Module):
    """(convolution => [BN] => ReLU => FiLM) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.double_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, gamma, beta):        
        t1 = self.double_conv1(x)        
        t2 = t1 * gamma[:, None, None, None] + beta[:, None, None, None]
        return self.double_conv2(t2)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale_factor),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class DownFiLM(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(scale_factor)
        self.doubleconvfilm = DoubleConvFiLM(in_channels, out_channels)
        
    def forward(self, x, gamma, beta):
        t1 = self.maxpool(x)
        return self.doubleconvfilm(t1, gamma, beta)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()

        n_channels = config['n_channels']
        channels_latent = config['channels_latent']
        n_classes = config['n_classes']
        n_hidden_film = config['n_hidden_film']
        n_hidden_layers_film = config['n_hidden_layers_film']
        n_conditioning = config['n_conditioning']
        factor = 2

        self.film = FiLM(n_conditioning, 2, n_hidden_film, n_hidden_layers_film)

        self.inc = DoubleConv(n_channels, channels_latent)
        self.down1 = Down(channels_latent, 2*channels_latent, scale_factor=2)
        self.down2 = Down(2*channels_latent, 4*channels_latent, scale_factor=2)
        self.down3 = Down(4*channels_latent, 8*channels_latent, scale_factor=2)        
        self.down4 = Down(8*channels_latent, 16*channels_latent // factor, scale_factor=2)

        self.up1 = Up(16*channels_latent, 8*channels_latent // factor, scale_factor=2)
        self.up2 = Up(8*channels_latent, 4*channels_latent // factor, scale_factor=2)
        self.up3 = Up(4*channels_latent, 2*channels_latent // factor, scale_factor=2)
        self.up4 = Up(2*channels_latent, channels_latent, scale_factor=2)
        self.outc = OutConv(channels_latent, n_classes)

    def forward(self, image, modes, instrument):
        x = torch.cat((image[:, None, :, :], modes), dim=1)

        film = self.film(instrument)

        breakpoint()

        x1 = self.inc(x)        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)        
        out = self.outc(x)

        return out

class UNetFiLM(nn.Module):
    def __init__(self, config):
        super(UNetFiLM, self).__init__()

        n_channels = config['n_channels']
        channels_latent = config['channels_latent']
        n_classes = config['n_classes']
        n_hidden_film = config['n_hidden_film']
        n_hidden_layers_film = config['n_hidden_layers_film']
        n_conditioning = config['n_conditioning']
        factor = 2

        self.film = FiLM(n_conditioning, 2, n_hidden_film, n_hidden_layers_film)

        self.inc = DoubleConv(n_channels, channels_latent)
        self.down1 = DownFiLM(channels_latent, 2*channels_latent, scale_factor=2)
        self.down2 = DownFiLM(2*channels_latent, 4*channels_latent, scale_factor=2)
        self.down3 = DownFiLM(4*channels_latent, 8*channels_latent, scale_factor=2)        
        self.down4 = DownFiLM(8*channels_latent, 16*channels_latent // factor, scale_factor=2)

        self.up1 = Up(16*channels_latent, 8*channels_latent // factor, scale_factor=2)
        self.up2 = Up(8*channels_latent, 4*channels_latent // factor, scale_factor=2)
        self.up3 = Up(4*channels_latent, 2*channels_latent // factor, scale_factor=2)
        self.up4 = Up(2*channels_latent, channels_latent, scale_factor=2)
        self.outc = OutConv(channels_latent, n_classes)

    def forward(self, image, modes, instrument):
        x = torch.cat((image[:, None, :, :], modes), dim=1)

        gamma, beta = self.film(instrument)

        x1 = self.inc(x)        
        x2 = self.down1(x1, gamma, beta)
        x3 = self.down2(x2, gamma, beta)
        x4 = self.down3(x3, gamma, beta)
        x5 = self.down4(x4, gamma, beta)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)        
        out = self.outc(x)

        return out
    
if (__name__ == '__main__'):
    
    config = {                
        'n_channels': 8,
        'channels_latent': 64,
        'n_classes': 1,
        'bilinear': True
        }
    
    x = torch.zeros((10, 8, 64, 64))
            
    tmp = UNet(config)

    out = tmp(x)