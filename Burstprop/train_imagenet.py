'''
Main script for training a network on ImageNet using backprop, feedback alignment or burstprop as presented in

"Payeur, A., Guerguiev, J., Zenke, F., Richards, B., & Naud, R. (2020).
Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits. bioRxiv."

This code was partially adapted from https://github.com/pytorch/examples/tree/master/imagenet.

     Author: Jordan Guergiuev
     E-mail: jordan.guerguiev@me.com
       Date: April 5, 2020
Institution: University of Toronto Scarborough

Copyright (C) 2020 Jordan Guerguiev

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data

import os
import datetime
import shutil
import argparse
import time
import datetime
import random

#from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter

from networks_imagenet import *

import pickle
'''
t = torch.cuda.get_device_properties(0).total_memory
c = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = c-a  # free inside cache


parser = argparse.ArgumentParser()
parser.add_argument('folder_prefix', help='Prefix of folder name where data will be saved')
parser.add_argument('data_path', help='Path to the dataset', type=str)
parser.add_argument("-n_epochs", type=int, help="Number of epochs", default=500)
parser.add_argument("-batch_size", type=int, help="Batch size", default=128)
parser.add_argument('-validation', default=False, help="Whether to the validation set", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-hidden_lr", help="Learning rate for hidden layers", type=float, default=0.01)
parser.add_argument("-output_lr", help="Learning rate for output layer", type=float, default=0.01)
parser.add_argument("-weight_fa_std", help="Standard deviation of initial feedback weights for hidden layers", type=float, default=0.01)
parser.add_argument("-momentum", type=float, help="Momentum", default=0.9)
parser.add_argument("-weight_decay", type=float, help="Weight decay", default=1e-4)
parser.add_argument("-p_baseline", type=float, help="Output layer baseline burst probability", default=0.2)
parser.add_argument('-use_backprop', default=False, help="Whether to train using backprop", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-weight_fa_learning', default=True, help="Whether to update feedback weights", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-kappa", type=float, help="Scaling factor used in target burst probability at output layer", default=1e-5)
parser.add_argument('-use_adam', default=False, help="Whether to use the Adam optimizer", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-resume_path', default='', help=' (Optional) Path to latest saved checkpoint to resume from', type=str)
parser.add_argument("-info", type=str, help="Any other information about the simulation", default="")

args=parser.parse_args()
'''
folder_prefix          = '/home/paperspace/Documents/Learning_Rule/Burstprop/Burstprop-master/saved_data' #args.folder_prefix
data_path              = '/home/paperspace/Documents/Learning_Rule/data/kamitani_data/train_sub1/imagenet_images/' #args.data_path
n_epochs               = 40 #args.n_epochs
batch_size             = 128 #args.batch_size
validation             = False #args.validation
hidden_lr              = 0.01 #args.hidden_lr
output_lr              = 0.01 #args.output_lr
weight_fa_std          = 0.01 #args.weight_fa_std
momentum               = 0.9 #args.momentum
weight_decay           = 1e-4 #args.weight_decay
p_baseline             = 0.2 #args.p_baseline
use_backprop           = False #args.use_backprop
weight_fa_learning     = True #args.weight_fa_learning
kappa                  = 1e-5 #args.kappa
use_adam               = False #args.use_adam
resume_path            = '' #args.resume_path
info                   = "" #args.info

n_gpus_per_node = torch.cuda.device_count()

best_acc1 = 0

if use_backprop:
    weight_fa_learning = False

if use_backprop:
    lr = [output_lr]*8
else:
    lr = [hidden_lr]*7 + [output_lr]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if use_backprop:
    net = ImageNetConvNetBP(input_channels=3)
else:
    net = ImageNetConvNet(input_channels=3, p_baseline=p_baseline, weight_fa_std=weight_fa_std, weight_fa_learning=weight_fa_learning, kappa=kappa)

net = net.cuda()

#module = net.module

criterion = torch.nn.CrossEntropyLoss()

if use_backprop:
    if not use_adam:
        optimizer = torch.optim.SGD(net.parameters(), lr=output_lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=output_lr, betas=[0.9, 0.99], eps=0.1)
else:
    if not use_adam:
        optimizer = torch.optim.SGD([
                                    {"params": net.conv1.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": net.conv2.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": net.conv3.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": net.conv4.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": net.conv5.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": net.conv6.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": net.conv7.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    #{"params": net.conv8.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": net.fc1.parameters(), "lr": output_lr, "weight_decay": weight_decay, "momentum": momentum}
                                    ], output_lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam([
                                    {"params": net.conv1.parameters(), "lr": hidden_lr},
                                    {"params": net.conv2.parameters(), "lr": hidden_lr},
                                    {"params": net.conv3.parameters(), "lr": hidden_lr},
                                    {"params": net.conv4.parameters(), "lr": hidden_lr},
                                    {"params": net.conv5.parameters(), "lr": hidden_lr},
                                    {"params": net.conv6.parameters(), "lr": hidden_lr},
                                    {"params": net.conv7.parameters(), "lr": hidden_lr},
                                    {"params": net.fc1.parameters(), "lr": output_lr}
                                    ], output_lr, betas=[0.9, 0.99], eps=0.1)

start_epoch = 0
if len(resume_path) > 0:
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

#Load Training Data
transform_imagenet = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])
imagenet_data = datasets.ImageFolder(root='/home/paperspace/Documents/Learning_Rule/data/kamitani_data/train_sub1/imagenet_images/', transform=transform_imagenet)  # your "master" dataset
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
torch.save(train_tensor, 'train_tensor_burstprop.pt')
torch.save(test_tensor, 'test_tensor_burstprop.pt')

del train_tensor
del test_tensor


def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, 'best_model.pth.tar'))

def adjust_learning_rate(optimizer, starting_lrs, epoch):
    # set the learning rate to the initial learning rate decayed by 10 every 30 epochs
    for i in range(len(optimizer.param_groups)):
        param_group = optimizer.param_groups[i]
        lr = starting_lrs[i] * (0.1 ** (epoch // 30))
        param_group['lr'] = lr

class AverageMeter(object):
    # computes and stores the average and current value
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    # computes the accuracy over the k top predictions for the specified values of k
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(trainloader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    net.train()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = inputs.cuda()
        targets = targets.cuda(non_blocking=True)

        if not use_backprop:
            t = F.one_hot(targets, num_classes=150).float()

        # compute output
        if use_backprop:
            outputs = net(inputs)
        else:
            outputs = net(inputs, t)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            progress.display(batch_idx)

    return top1.avg, top5.avg, losses.avg

def test():
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(testloader), [batch_time, losses, top1, top5], prefix='Test: ')

    net.eval()

    with torch.no_grad():
        end = time.time()

        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.cuda(non_blocking=True)
            inputs = inputs.cuda()
            if not use_backprop:
                t = F.one_hot(targets, num_classes=150).float()

            # compute output
            if use_backprop:
                outputs = net(inputs)
            else:
                outputs = net(inputs, t)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0:
                progress.display(batch_idx)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
'''
if folder_prefix is not None:
    # generate a name for the folder where data will be stored
    lr_string = " ".join([ str(i) for i in lr ])
    weight_fa_std_string = "{}".format(weight_fa_std)

    folder = "{} - {} - {} - {} - {} - {} - {}".format(folder_prefix, lr_string, weight_fa_std_string, batch_size, momentum, weight_decay, p_baseline) + " - BP"*(use_backprop == True) + " - {}".format(info)*(info != "")
else:
    folder = None

if folder is not None:
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save a human-readable text file containing simulation details
    timestamp = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    with open(os.path.join(folder, "params.txt"), "w") as f:
        f.write("Simulation run @ {}\n".format(timestamp))
        f.write("Number of epochs: {}\n".format(n_epochs))
        f.write("Batch size: {}\n".format(batch_size))
        f.write("Using validation set: {}\n".format(validation))
        f.write("Feedforward learning rates: {}\n".format(lr))
        f.write("Feedback weight initialization standard deviation: {}\n".format(weight_fa_std))
        f.write("Momentum: {}\n".format(momentum))
        f.write("Weight decay: {}\n".format(weight_decay))
        f.write("Output layer baseline burst probability: {}\n".format(p_baseline))
        f.write("Using backprop: {}\n".format(use_backprop))
        f.write("Feedback weight learning: {}\n".format(weight_fa_learning))
        f.write("Output layer target burst probability scaling factor: {}\n".format(kappa))
        f.write("Using Adam optimizer: {}\n".format(use_adam))
        f.write("Resuming from path: {}\n".format(resume_path))
        if info != "":
            f.write("Other info: {}\n".format(info))

    filename = os.path.basename(__file__)
    if filename.endswith('pyc'):
        filename = filename[:-1]
    shutil.copyfile(filename, os.path.join(folder, filename))
    shutil.copyfile("networks_imagenet.py", os.path.join(folder, "networks_imagenet.py"))
    shutil.copyfile("layers_imagenet.py", os.path.join(folder, "layers_imagenet.py"))

'''
# test_acc1, test_acc5, test_loss = test()



starting_lrs = [ param_group['lr'] for param_group in optimizer.param_groups ]

train_hist = []
test_hist = []
for epoch in range(start_epoch, 80):
    print("\nEpoch {}.".format(epoch+1))
    adjust_learning_rate(optimizer, starting_lrs, epoch)

    train_acc1, train_acc5, train_loss = train(epoch)
    train_hist.append([train_acc1, train_acc5, train_loss])
    test_acc1, test_acc5, test_loss = test()
    test_hist.append([test_acc1, test_acc5, test_loss])

    PATH_epoch = './saved_epochs/Burst_epoch_' + str(epoch + 1) + '.pth'
    torch.save(net.state_dict(), PATH_epoch)

PATH = './burstprop_kamitani.pth'
torch.save(net.state_dict(), PATH)
with open("train_hist_burstprop.txt", "wb") as f0:  # Pickling
    pickle.dump(train_hist, f0)
with open("test_hist_burstprop.txt", "wb") as p1:   #Pickling
    pickle.dump(test_hist, p1)

'''
    if folder is not None:


    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict()},
            is_best, folder)
'''
'''
#for batch_idx, (inputs, targets) in enumerate(test_loader):
#    targets = targets.cuda()
net.train()
loss_hist = []
for epoch in range(80):  # loop over the dataset multiple times
    running_loss = 0.0
    for inputs, labels in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = inputs.cuda()
        labels = labels.cuda()
        t = F.one_hot(labels, num_classes=10).float()
        outputs = net(inputs, t)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #print("running loss: "+str(running_loss))
    print(f"Epoch: {epoch}. Loss: {loss}")
    loss_hist.append(loss)
print('Finished Training')

#Test ImageNet
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        t = F.one_hot(labels, num_classes=10).float()
        outputs = net(inputs, t)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
'''