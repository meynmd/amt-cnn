import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import cuda


class ConvBlock2d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1), dilation=(1, 1), slice=None):
        super(ConvBlock2d, self).__init__()

        print('\tconv block {}x{}'.format(kernel_size[0], kernel_size[1]), end=' ')
        print('{} => {}'.format(ch_in, ch_out))

        kernel_size, stride, dilation, padding = ((var, var) if type(var) is int else var
                                                  for var in (kernel_size, stride, dilation, padding))
        self.layers = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

        if slice is None:
            self.left_slice, self.right_slice = 0, padding[1]
        else:
            self.left_slice, self.right_slice = slice

    def forward(self, x):
        z = self.layers(x)
        return z[:, :, :, self.left_slice: -self.right_slice]


class Downsample(nn.Module):
    def __init__(self, batch_size=8, ch_out=64, use_cuda=None, max_w=None):
        print('Downsample, out channels: {}'.format(ch_out))

        super(Downsample, self).__init__()
        self.channels_out = ch_out
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda

        self.conv_layers = nn.Sequential(
            ConvBlock2d(1, 8, (3, 3), padding=(1, 3), slice=(1, 1), stride=(1, 2), dilation=(1, 1)),
            ConvBlock2d(8, 16,(3, 3), padding=(2, 2), slice=(0, 1), stride=(1, 2), dilation=(2, 1)),
            ConvBlock2d(16, 32, (3, 3), padding=(4, 4), stride=(1, 1), dilation=(4, 2)),
            ConvBlock2d(32, 64, (3, 3), padding=(8, 8), stride=(1, 1), dilation=(8, 4)),
            ConvBlock2d(64, 64, (3, 3), padding=(16, 16), stride=(1, 1), dilation=(16, 8)),
            ConvBlock2d(64, 64, (3, 3), padding=(16, 32), stride=(1, 1), dilation=(16, 16)),
            ConvBlock2d(64, 64, (3, 3), padding=(16, 64), stride=(1, 1), dilation=(16, 32)),
            ConvBlock2d(64, 64, (3, 3), padding=(16, 128), stride=(1, 1), dilation=(16, 64)),
            ConvBlock2d(64, 64, (3, 3), padding=(16, 256), stride=(1, 1), dilation=(16, 128)),

            nn.Conv2d(64, self.channels_out, 1),
            nn.BatchNorm2d(self.channels_out),
            nn.ReLU()
        )

    def forward(self, x_batch):
        if (self.cuda_dev is not None) and (not x_batch.is_cuda):
            x_batch = x_batch.cuda(self.cuda_dev)
        z = self.conv_layers(x_batch)
        return z


class Upsample(nn.Module):
    def __init__(self, batch_size=8, ch_in=64, scale_factor=(1, 4), use_cuda=None, max_w=None):
        print('Upsample: in-channels {}'.format(ch_in))
        super(Upsample, self).__init__()
        self.channels_in = ch_in
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda
        self.scale_factor = scale_factor

        self.layers = nn.Sequential(
            nn.modules.Upsample(scale_factor=self.scale_factor, mode='nearest'),
            ConvBlock2d(ch_in, 32, (1, 5), padding=(0, 4)),
            ConvBlock2d(32, 32, (1, 5), padding=(0, 4)),
            nn.Conv2d(32, 16, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return self.layers(x)


class AMT_CNN(nn.Module):
    def __init__(self, batch_size=8, ch_downsample=32, ch_upsample=32, use_cuda=None, max_w=None):
        super(AMT_CNN, self).__init__()

        self.name = 'cnn v8\n'
        self.downsample = Downsample(ch_out=ch_downsample, use_cuda=use_cuda, max_w=max_w)
        self.upsample = Upsample(ch_in=ch_upsample, use_cuda=use_cuda, max_w=max_w)

    def forward(self, x):
        z = self.downsample(x)
        z = self.upsample(z)
        return z[:, :, :, z.shape[3]//2:]
