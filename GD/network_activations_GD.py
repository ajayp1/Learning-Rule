import torch
import torch.nn as nn
import torchvision.transforms as transforms
from collections import namedtuple
from GD.GD_viz_model import *
import numpy as np
#from Hebbian.Hebbian_viz_model import *
#from Burstprop.networks_imagenet import *
import os.path
import pickle
import torchvision.datasets as datasets
import os
import argparse

'''
parser = argparse.ArgumentParser()
parser.add_argument('epoch_num')
args = parser.parse_args()
epoch_num = args.epoch_num
'''
class get_activation(torch.nn.Module):
    def __init__(self, model):
        super(get_activation, self).__init__()
        layer = model.__dict__.get('_modules').keys()
        self.layers = []
        for layer in layer:
            self.layers.append(getattr(model, layer))

    def forward(self, x):
        results = []
        for model in self.layers:
            #print(str(model))
            #print(type(x))
            x = model(x)
            results.append(x)
        return results

'''
import os,sys
import shutil
folder_path = '/home/ajay/Documents/Learning_Rule/data/kamitani_images/image_per_category/'
for i in range(150):
    path = os.path.join(folder_path, str('cat_'+str(i+1)))
    folder = folder_path+str(i+1)+'/'

    os.makedirs(path)
    shutil.move(folder, path)
    
    for filename in os.listdir(folder):
           infilename = os.path.join(folder,filename)
           if not os.path.isfile(infilename): continue
           oldbase = os.path.splitext(filename)
           newname = infilename.replace('.JPEG', '.jpg')
           output = os.rename(infilename, newname)

for i in range(150):
    path1 = os.path.join(folder_path, str('cat_'+str(i+1)+'/'+str(i+1)))
    for filename in os.listdir(path1):
           infilename = os.path.join(path1,filename)
           print(filename)
           if not os.path.isfile(infilename): continue
           os.rename(infilename, infilename+'.jpeg')
'''

index = 20
while index <= 150:

    # Load Gradient Descent/Backprop
    #PATH_GD = './GD/saved_epochs_919/GD_epoch_'+epoch_num+'.pth'
    PATH_GD = './viz_netGD_kamitani_919.pth'
    viz = viz_model(3)
    viz.load_state_dict(torch.load(PATH_GD))
    viz = viz.cuda()

    activation_per_category = []
    if index == 150:
        difference = 10
    else:
        difference = 20

    for category in range(index-difference, index):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                 std = [ 0.229, 0.224, 0.225 ]),
        ])

        trainset = datasets.ImageFolder(root=str('/home/paperspace/Desktop/Learning_Rule/data/kamitani_data/image_per_category/cat_'+str(category+1)+'/'), transform=transform)

        DIR = './data/kamitani_data/image_per_category/'+'cat_'+str(category+1)+'/'+str(category+1)+'/'
        num_files = len([f for f in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, f))])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=num_files, shuffle=True, num_workers=2)
        avg_activations = []
        avg_GD = []
        get_GD_activation = get_activation(viz)
        GD_activations = []
        for inputs, labels in trainloader:

                activations = []
                inputs = inputs.cuda()
                labels = labels.cuda()
                num, x, y, z = inputs.shape
                #print(str(num))

                GD_activations.append(get_GD_activation(inputs))

        for i in range(len(GD_activations[0])):
            avg_GD.append(torch.unsqueeze(torch.mean(GD_activations[0][i],0) ,0))
        activation_per_category.append(avg_GD)

    #with open("/home/paperspace/Desktop/Learning_Rule/GD/epoch_data_919/epoch_"+str(epoch_num)+"/activations_per_category_"+str(index)+".txt", "wb") as f0:  # Pickling
    #    pickle.dump(activation_per_category, f0)
    with open("/home/paperspace/Desktop/Learning_Rule/GD/final_model_data/activations_per_category_" + str(index) + ".txt", "wb") as f0:  # Pickling
        pickle.dump(activation_per_category, f0)
    print('saved activations per category; index: '+str(index))
    torch.cuda.empty_cache()

    if index == 140:
        index = 150
    elif index == 150:
        index += 1
    else:
        index += 20
