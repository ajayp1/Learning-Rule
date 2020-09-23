import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import random
import os
import shutil
import argparse
import time
import datetime
import pickle
#from torch.utils.tensorboard import SummaryWriter

print('started running')

transform_imagenet = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

imagenet_data = datasets.ImageFolder(root='/home/paperspace/Desktop/Learning_Rule/data/kamitani_data/train_sub1/imagenet_images/', transform=transform_imagenet)  # your "master" dataset
n = len(imagenet_data)  # how many total elements you have
n_test = int( n * .30 )  # number of test/val elements
n_train = n - n_test
idx = list(range(n))  # indices to all elements
random.shuffle(idx)  # in-place shuffle the indices to facilitate random splitting
train_idx = idx[:n_train]
test_idx = idx[n_train:]

train_set = data.Subset(imagenet_data, train_idx)
test_set = data.Subset(imagenet_data, test_idx)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                         shuffle=False, num_workers=4)
train_tensor = torch.tensor(train_idx)
test_tensor = torch.tensor(test_idx)
torch.save(train_tensor, 'train_tensor_GD.pt')
torch.save(test_tensor, 'test_tensor_GD.pt')

del train_tensor
del test_tensor



#Viz Model (Adapted CORnet)
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
##


print('loaded model')


#Train ImageNet
viz = viz_model(3) #, int(max(example_targets)+1))
viz = viz.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(viz.parameters(), lr=0.001)
#Train Models


def train(epoch):

    loss_hist = []
    train_acc_hist = []
    correct = 0
    total = 0
    running_loss = 0.0
    batch_num = 0
    for inputs, labels in trainloader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = viz(inputs)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # print statistics
        running_loss += loss.item()
        #print("running loss: "+str(running_loss))
        batch_num += 1
        if batch_num % 50 == 0 or batch_num == 1:
            print('done with batch: '+str(batch_num))

    print(f"**************Epoch: {epoch}. Loss: {loss}")
    PATH_epoch = '/home/paperspace/Desktop/Learning_Rule/GD/saved_epochs_919/GD_epoch_' + str(epoch + 1) + '.pth'
    torch.save(viz.state_dict(), PATH_epoch)

    loss_hist.append(loss)
    train_acc = (100 * correct / total)
    train_acc_hist.append(train_acc)
    return loss_hist, train_acc_hist


#viz1 = viz_model(3)
#viz1.load_state_dict(torch.load(PATH))

#Test ImageNet
def test():
    correct = 0
    total = 0
    loss_hist_test = []
    test_acc_hist = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            #print(str(inputs))
            outputs = viz(inputs)
            loss_test = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = (100 * correct / total)
    test_acc_hist.append(test_acc)
    loss_hist_test.append(loss_test)

    print('Test accuracy: '+str(test_acc)+'%')
    return loss_hist_test, test_acc_hist



print('started training')
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

for epoch in range(40):

    print('starting epoch: '+str(epoch))
    tr_loss_hist, tr_acc_hist = train(epoch)
    train_loss_hist.append(tr_loss_hist)
    train_acc_hist.append(tr_acc_hist)

    te_loss_hist, te_acc_hist = test()
    test_loss_hist.append(te_loss_hist)
    test_acc_hist.append(te_acc_hist)

PATH_final_model = './viz_netGD_kamitani_919.pth'
torch.save(viz.state_dict(), PATH_final_model)
with open("loss_history_GD_train_919.txt", "wb") as f0:  # Pickling
    pickle.dump(train_loss_hist, f0)
with open("train_acc_history_GD_train_919.txt", "wb") as p0:  # Pickling
    pickle.dump(train_acc_hist, p0)
with open("test_acc_hist_GD_919.txt", "wb") as f1:   #Pickling
    pickle.dump(test_acc_hist, f1)
with open("loss_hist_test_GD_919.txt", "wb") as p1:   #Pickling
    pickle.dump(test_loss_hist, p1)

#import pickle
#with open("loss_hist_test_GD-1.txt", "rb") as p1:   #Pickling
#    test = pickle.load(p1)