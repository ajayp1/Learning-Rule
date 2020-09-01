import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data
import random

transform_imagenet = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

imagenet_data = datasets.ImageFolder(root='/home/ajay/imagenet_test2', transform=transform_imagenet)  # your "master" dataset
n = len(imagenet_data)  # how many total elements you have
n_test = int( n * .30 )  # number of test/val elements
n_train = n - n_test
idx = list(range(n))  # indices to all elements
random.shuffle(idx)  # in-place shuffle the indices to facilitate random splitting
train_idx = idx[:n_train]
test_idx = idx[n_train:]

train_set = data.Subset(imagenet_data, train_idx)
test_set = data.Subset(imagenet_data, test_idx)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=15,
                                          shuffle=True, num_workers=2)
#testloader = torch.utils.data.DataLoader(test_set, batch_size=15,
#                                         shuffle=False, num_workers=4)

train_tensor = torch.tensor(train_idx)
#test_tensor = torch.tensor(test_idx)
torch.save(train_tensor, 'train_tensor_hebbian.pt')
#torch.save(test_tensor, 'test_tensor_hebbian.pt')

del train_tensor
#del test_tensor



import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hebb as H
import params as P
import utils


class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

class Net(nn.Module):
    # Layer names
    CONV1 = 'conv1'
    POOL1 = 'pool1'
    BN1 = 'bn1'
    CONV2 = 'conv2'
    BN2 = 'bn2'
    CONV_OUTPUT = BN2  # Symbolic name for the last convolutional layer providing extracted features
    #FC5 = 'fc5'
    #BN5 = 'bn5'
    #FC6 = 'fc6'
    #CLASS_SCORES = FC6  # Symbolic name of the layer providing the class scores as output

    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel):
        super(Net, self).__init__()

        # Shape of the tensors that we expect to receive as input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel = pool_kernel
        # Here we define the layers of our network

        # First convolutional layer
        self.padding1 = nn.ZeroPad2d(self.kernel_size//2)
        self.conv1 = H.HebbianMap2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            out=H.clp_vector_proj2d,
            eta=0.1,
        )  # 3 input channels, 8x12=96 output channels, 5x5 convolutions
        self.bn1 = nn.GroupNorm(self.out_channels, self.out_channels)  # Batch Norm layer

        # Second convolutional layer
        self.padding2 = nn.ZeroPad2d(1)
        self.conv2 = H.HebbianMap2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            out=H.clp_vector_proj2d,
            eta=0.1,
        )  # 96 input channels, 8x16=128 output channels, 3x3 convolutions
        self.pool = nn.MaxPool2d(self.pool_kernel)
        self.bn2 = nn.GroupNorm(self.out_channels, self.out_channels)  # Batch Norm layer
        #self.conv_output_shape = utils.get_conv_output_shape(self)

    # This function forwards an input through the convolutional layers and computes the resulting output
    def forward(self, x):
       # print('input:'+str(x.shape))
        conv1_out = self.conv1(self.padding1(x))
        #print('conv1: '+str(conv1_out.shape))
        bn1_out = self.bn1(conv1_out)

        conv2_out = self.conv2(self.padding2(bn1_out))
        #print('conv2: ' + str(conv2_out.shape))
        bn2_out = self.bn2(conv2_out)
        #print('bn2: ' + str(bn2_out.shape))
        pool2_out = self.pool(bn2_out)
        #print('pool2: ' + str(pool2_out.shape))

        return pool2_out

class Hebb_model(nn.Module):
    def __init__(self, inp_channels, num_classes):
        super(Hebb_model, self).__init__()
        self.inp_channels = inp_channels
        self.num_classes = num_classes
        self.V1_ip = [self.inp_channels, 64, 7, 4]
        self.V2_ip = [64, 128, 3, 2]
        self.V4_ip = [128, 256, 3, 2]
        self.IT_ip = [256, 512, 3, 2]
        self.V1 = Net(in_channels=self.V1_ip[0], out_channels=self.V1_ip[1], kernel_size=self.V1_ip[2], pool_kernel=self.V1_ip[3])
        #print('V2!!')
        self.V2 = Net(in_channels=self.V2_ip[0], out_channels=self.V2_ip[1], kernel_size=self.V2_ip[2], pool_kernel=self.V2_ip[3])
        #print('V4!!')
        self.V4 = Net(in_channels=self.V4_ip[0], out_channels=self.V4_ip[1], kernel_size=self.V4_ip[2], pool_kernel=self.V4_ip[3])
        #print('IT!!')
        self.IT = Net(in_channels=self.IT_ip[0], out_channels=self.IT_ip[1], kernel_size=self.IT_ip[2], pool_kernel=self.IT_ip[3])

        self.decoder =  nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, self.num_classes))]))
    def forward(self, input):
        #print('V1!!')
        #print('init_inp: '+str(input.shape))
        output = self.V1(input)
        #print('V2!!')
        #print('V1_shape: ' + str(output.shape))
        output = self.V2(output)
        #print('V4!!')
        #print('V2_shape: ' + str(output.shape))
        output = self.V4(output)
        #print('IT!!')
        #print('V4_shape: ' + str(output.shape))
        output = self.IT(output)
        output = self.decoder(output)
        return output

'''

#Load CIFAR
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#NEXT: train according to github - just save representations during course of training for a given class

# misc demoing
dataiter = iter(trainloader)
images, labels = dataiter.next()


hebb = Hebb_model(3, 10)
hebb_total_params = sum(p.numel() for p in hebb.parameters() if p.requires_grad)

demo_ip = images[1, :, :, :].expand(1,3,32,32)

pool4 = nn.MaxPool2d(4)
pool2 = nn.MaxPool2d(2)

out = pool(demo_ip)
out4 = pool2(out2)
print('out2: '+str(out4.shape))

def hello(x):
    x*2

x = 2
hello(x)
'''
hebb = Hebb_model(3, 6)
hebb = hebb.cuda()
hebb.train()
for epoch in range(40):  # loop over the dataset multiple times
    print('epoch:: '+str(epoch))
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs = inputs.cuda()
        hebb(inputs)
hebb.eval()
PATH = './viz_netHebb_stringer.pth'
torch.save(hebb.state_dict(), PATH)
