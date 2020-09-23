import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """
    def forward(self, x):
        return x.view(x.size(0), -1)
class viz_stack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel):
        super(viz_stack, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel = pool_kernel
        self.conv_output = int(np.floor((out_channels+in_channels)/2))
        #self.out_shape = out_shape

        self.conv_input = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                                    padding=self.kernel_size // 2)
        self.norm_input = nn.GroupNorm(self.out_channels, self.out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(self.out_channels, self.out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(self.pool_kernel)

        #self.output = Identity()  # for an easy access to this block's output
    def forward(self, inp):
        '''
        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)


        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state
        '''
        inp = self.conv_input(inp)
        #print('conv_inp: '+str(inp.shape))
        inp = self.norm_input(inp)
        inp = self.nonlin_input(inp)
        x = self.conv1(inp)
        #print('conv1: ' + str(x.shape))
        x = self.norm1(x)
        #print('norm1: ' + str(x.shape))
        x = self.nonlin1(x)
        #print('nonlin1: ' + str(x.shape))
        x = self.pool(x)
        #print('pool: ' + str(x.shape))
        return x
class viz_model(nn.Module):
    def __init__(self, inp_channels):
        super(viz_model, self).__init__()
        self.inp_channels = inp_channels
        #self.num_classes = num_classes
        self.V1_ip = [self.inp_channels, 64, 7, 4]
        self.V2_ip = [64, 128, 3, 2]
        self.V4_ip = [128, 256, 3, 2]
        self.IT_ip = [256, 512, 3, 2]
        self.V1 = viz_stack(in_channels=self.V1_ip[0], out_channels=self.V1_ip[1], kernel_size=self.V1_ip[2], pool_kernel=self.V1_ip[3])
        self.V2 = viz_stack(in_channels=self.V2_ip[0], out_channels=self.V2_ip[1], kernel_size=self.V2_ip[2], pool_kernel=self.V2_ip[3])
        self.V4 = viz_stack(in_channels=self.V4_ip[0], out_channels=self.V4_ip[1], kernel_size=self.V4_ip[2], pool_kernel=self.V4_ip[3])
        self.IT = viz_stack(in_channels=self.IT_ip[0], out_channels=self.IT_ip[1], kernel_size=self.IT_ip[2], pool_kernel=self.IT_ip[3])
        self.decoder =  nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 150))]))
            #,('softmax', nn.Softmax(dim=1))]))
    def forward(self, input):
        #print('V1!!')
        output = self.V1(input)
        #print('V1!!')
        output = self.V2(output)
        #print('V1!!')
        output = self.V4(output)
        #print('V1!!')
        output = self.IT(output)
        output = self.decoder(output)
        return output
