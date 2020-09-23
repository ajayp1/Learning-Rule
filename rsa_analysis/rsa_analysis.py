import numpy as np
from pyrsa import rdm
from pyrsa import data
import pickle
import torch
import argparse
import os

model = 'GD'

#print(os.getcwd())
#model = 'GD'
#epoch_num = 1
#Load ANN data
with open("/home/paperspace/Desktop/Learning_Rule/"+model+"/final_model_data/activations_per_category.txt", "rb") as f:
    activation_per_category =  pickle.load(f)

#Generate brain data RDMs
a_file = open("./kamitani_fmri_data/brain_area_data_sub1_training1_NEW.pkl", "rb")
brain_area = pickle.load(a_file)
ars = ['V1', 'V2', 'V3', 'V4']
brain_area_rsa = {}
if model == 'Burstprop':
    ar_count = 1
else:
    ar_count = 0
for ar in ars:
    area_data_LH = brain_area[ar]['LH']
    area_data_LH = area_data_LH['data']
    area_data_RH = brain_area[ar]['RH']
    area_data_RH = area_data_RH['data']
    data_LH = np.empty([len(area_data_LH),len(area_data_LH[1])])
    for i in range(len(area_data_LH)):
        data_LH[i,:] = area_data_LH[i]
    data_RH = np.empty([len(area_data_RH),len(area_data_RH[1])])
    for i in range(len(area_data_RH)):
        data_RH[i,:] = area_data_RH[i]
    del area_data_LH
    del area_data_RH

    dataset_LH = data.dataset.Dataset(data_LH)
    rdm_LH = rdm.calc.calc_rdm(dataset_LH)
    dataset_RH = data.dataset.Dataset(data_RH)
    rdm_RH = rdm.calc.calc_rdm(dataset_RH)
    rdm_area = rdm_LH
    rdm_area.dissimilarities = np.vstack([rdm_LH.dissimilarities,rdm_RH.dissimilarities])
    rdm_area.dissimilarities = np.mean(rdm_area.dissimilarities,axis=0)
    rdm_area.dissimilarities = np.expand_dims(rdm_area.dissimilarities,axis=1)
    rdm_area.dissimilarities = rdm_area.dissimilarities.T

    #Generate ANN RDMs
    area_data_ann = []
    for i in range(150):
            area_data_ann.append(activation_per_category[i][ar_count])

            #area_data_burstprop.append(activation_per_category[i][1][ar_count_burst])
            #area_data_ann.append(activation_per_category[i][2][ar_count_hebb])
    if model=='Burstprop':
        ar_count += 2
    else:
        ar_count += 1

    data_ann = np.empty([len(area_data_ann),list(torch.flatten(area_data_ann[0]).shape)[0]])
    #data_hebb = np.empty([len(area_data_hebb),list(torch.flatten(area_data_hebb[0]).shape)[0]])
    #data_burst = np.empty([len(area_data_burstprop),list(torch.flatten(area_data_burstprop[0]).shape)[0]])
    for i in range(len(area_data_ann)):
        area_data_ann[i] = area_data_ann[i].cpu()
        #area_data_hebb[i] = area_data_hebb[i].cpu()
        #area_data_burstprop[i] = area_data_burstprop[i].cpu()
        data_ann[i,:] = torch.flatten(area_data_ann[i]).detach().numpy()
        #data_hebb[i,:] = torch.flatten(area_data_hebb[i]).detach().numpy()
        #data_burst[i,:] = torch.flatten(area_data_burstprop[i]).detach().numpy()
    dataset_ann = data.dataset.Dataset(data_ann)
    rdm_ann = rdm.calc.calc_rdm(dataset_ann)
    #dataset_hebb = data.dataset.Dataset(data_hebb)
    #rdm_hebb = rdm.calc.calc_rdm(dataset_hebb)
    #dataset_burst = data.dataset.Dataset(data_burst)
    #rdm_burst = rdm.calc.calc_rdm(dataset_burst)

    similarity_ann_area = rdm.compare(rdm_area, rdm_ann)
    #similarity_hebb_area = rdm.compare(rdm_area, rdm_hebb)
    #similarity_burst_area = rdm.compare(rdm_area, rdm_burst)

    rsa = [rdm_ann, similarity_ann_area]
    #rsa_hebb = [rdm_hebb, similarity_hebb_area]
    #rsa_burst = [rdm_burst, similarity_burst_area]
    #rsa_all = [rsa_ann, rsa_hebb, rsa_burst]

    brain_area_rsa.update({ar: rsa})

a_file = open("./"+model+"/final_model_data/"+model+"_brain_area_rsa.pkl", "wb")
pickle.dump(brain_area_rsa, a_file)
a_file.close()
print('Completed RSA for model: '+model)

#Transfer to local:
'''
scp -r  paperspace@184.105.3.25:/home/paperspace/Desktop/Learning_Rule/GD/final_model_data/ /home/ajay/Desktop/gd_other/
'''