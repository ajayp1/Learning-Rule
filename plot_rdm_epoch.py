import matplotlib.pyplot as plt
from pyrsa.vis import rdm_plot
from pyrsa.vis.colors import rdm_colormap
import pickle
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
import os

#for epoch in epochs
#load data, get rdm, plot
#save to png

with open("stimuli_perception_order.txt", "rb") as p1:   #Pickling
    order = pickle.load(p1)

models = ['GD', 'Hebbian', 'Burstprop']

for model in models:
    DIR_epoch_data = '/home/ajay/Desktop/Learning_Rule_paperspace/'+str(model)+'/epoch_data/'
    DIR_rdm_save = '/home/ajay/Desktop/Learning_Rule_paperspace/'+str(model)+'/rdm_epoch_plots/'
    num_files = len([f for f in os.listdir(DIR_epoch_data) if os.path.isfile(os.path.join(DIR_epoch_data, f))])
    for epoch in range(num_files):

        # Burstprop
        with open(DIR_epoch_data+str(model)+"_brain_area_rsa_epoch"+str(epoch+1)+".pkl","rb") as f:
            rsa_model = pickle.load(f)

        rdm_model = rsa_model['V4'][0].get_matrices()
        rdm_model = rdm_model[0]
        rdm_model = rdm_model[:, order][order]
        cmap = rdm_colormap()

        plt.imshow(rdm_model, cmap)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.xlabel('Epoch: '+str(epoch+1))
        if model == 'GD':
            plt.title('Gradient Descent')
        else: plt.title(model)

        plt.savefig(DIR_rdm_save+str(model)+'_rdm_epoch_'+str(epoch+1)+'.png')