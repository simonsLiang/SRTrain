import torch.nn as nn
from . import block as B
import torch
import numpy as np

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

