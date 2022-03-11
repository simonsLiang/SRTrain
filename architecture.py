import torch.nn as nn
from . import block as B
import torch
import numpy as np

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

class ERB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x+res

class IMDN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=1, out_nc=3, upscale=4):
        super(IMDN, self).__init__()
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.sobel_x, self.sobel_y = get_sobel(nf, nf)

        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf)

        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_fea = run_sobel(self.sobel_x,self.sobel_y,out_fea)
        
        out_B1 = self.IMDB1(out_fea)

        out_B = self.c(out_B1)
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

# AI in RTC Image Super-Resolution Algorithm Performance Comparison Challenge (Winner solution)
class IMDN_RTC(nn.Module):
    def __init__(self, in_nc=3, nf=12, num_modules=5, out_nc=3, upscale=2):
        super(IMDN_RTC, self).__init__()

        fea_conv = [B.conv_layer(in_nc, nf, kernel_size=3)]
        rb_blocks = [B.IMDModule_speed(in_channels=nf) for _ in range(num_modules)]
        LR_conv = B.conv_layer(nf, nf, kernel_size=1)

        upsample_block = B.pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

        self.model = B.sequential(*fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                  *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output


class IMDN_RTE(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=20, out_nc=3):
        super(IMDN_RTE, self).__init__()
        self.upscale = upscale
        self.fea_conv = nn.Sequential(B.conv_layer(in_nc, nf, 3),
                                      nn.ReLU(inplace=True),
                                      B.conv_layer(nf, nf, 3, stride=2, bias=False))

        self.block1 = IMDModule_Large(nf)
        self.block2 = IMDModule_Large(nf)
        self.block3 = IMDModule_Large(nf)
        self.block4 = IMDModule_Large(nf)
        self.block5 = IMDModule_Large(nf)
        self.block6 = IMDModule_Large(nf)

        self.LR_conv = B.conv_layer(nf, nf, 1, bias=False)

        self.upsampler = B.pixelshuffle_block(nf, out_nc, upscale_factor=upscale**2)

    def forward(self, input):

        fea = self.fea_conv(input)
        out_b1 = self.block1(fea)
        out_b2 = self.block2(out_b1)
        out_b3 = self.block3(out_b2)
        out_b4 = self.block4(out_b3)
        out_b5 = self.block5(out_b4)
        out_b6 = self.block6(out_b5)

        out_lr = self.LR_conv(out_b6) + fea

        output = self.upsampler(out_lr)

        return output

