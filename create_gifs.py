#import sys
#sys.path.append('/home/ajay/Documents/Learning_Rule/')
#sys.path.append('/home/ajay/Desktop/Learning_Rule_paperspace/')
import imageio as io
import os
import argparse
from PIL import Image
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

'''
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('fps')
args=parser.parse_args()
model = args.model
fps = args.fps
'''

models = ['Burstprop']
d = .15

for model in models:

    dur_list = []
    if model == 'Burstprop':
        max_epoch = 80
    else: max_epoch = 40
    for i in range(max_epoch):
        if i == max_epoch-1:
            dur_list.append(3)
        else:
            dur_list.append(d)
        # d += 5
        # if i == 4:

    rdm_img_path = '/home/ajay/Desktop/Learning_Rule_paperspace/' + str(model) + '/rdm_epoch_plots/'

    files = (fn for fn in os.listdir(rdm_img_path) if fn.startswith(str(model)+'_rdm_epoch'))
    file_list = []
    for idx, i in enumerate(files):
        file_list.append(i)
    file_list.sort(key=natural_keys)

    images = []
    for filename in file_list:
        images.append(io.imread('/home/ajay/Desktop/Learning_Rule_paperspace/'+str(model)+'/rdm_epoch_plots/'+filename))
    io.mimwrite('./rsa_evolution/'+str(model)+'_rsa_evolution.gif', images, duration=dur_list)
