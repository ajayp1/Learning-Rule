from nilearn import plotting
import nilearn
from nipy import labs
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.image import mean_img
import pandas as pd
import numpy as np
import os
import pickle

img = nib.load("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/sub-01/ses-perceptionTraining01/func/sub01_run01.nii")
img_prec = nib.load("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/derivatives/preproc-spm/output/sub-01/ses-perceptionTraining01/func/sub-01_ses-perceptionTraining01_task-perception_run-01_bold_preproc.nii")
img_test = nib.load("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/derivatives/preproc-spm/output/sub-01/ses-perceptionTest01/func/sub-01_ses-perceptionTest01_task-perception_run-01_bold_preproc.nii.gz")
anat = nib.load("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/derivatives/preproc-spm/output/sub-01/ses-anatomy/anat/sub-01_ses-anatomy_T1w_preproc.nii.gz")
mask = nib.load("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/sourcedata/ds001246-download-source/sub-01/anat/sub-01_mask_LH_V1d.nii.gz")


#TUTORIAL: https://nilearn.github.io/auto_examples/plot_decoding_tutorial.html (also see http://nilearn.github.io/auto_examples/01_plotting/plot_visualization.html#sphx-glr-auto-examples-01-plotting-plot-visualization-py)
from nilearn import datasets
# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()
# 'func' is a list of filenames: one for each subject
fmri_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask_vt[0]
masker_ex = NiftiMasker(mask_img=mask_filename, standardize=True)
img_ex = nib.load(fmri_filename)
print(str(img_ex.shape))
fmri_masked_ex = masker_ex.fit_transform(fmri_filename)
print(fmri_masked_ex.shape)
import pandas as pd
# Load behavioral information
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')
conditions_ex = behavioral['labels']
condition_mask = conditions.isin(['face', 'cat'])
fmri_masked = fmri_masked[condition_mask]
conditions = conditions[condition_mask]

plotting.plot_epi(mean_img(img))#, threshold=None)
plotting.show()

mask12 = labs.mask.intersect_masks([mask1, mask2], threshold=0)
masker = NiftiMasker(mask_img=mask1, standardize=True)
fmri_masked1 = masker.fit_transform(img_prec)     ##MASKED DATA!!!!!!!!!!
print(fmri_masked.shape)
mask12 = nilearn.masking.intersect_masks([mask1, mask2], threshold=0)
plotting.plot_roi(mask12, bg_img=anat, cmap='Paired')
plotting.show()
masker.generate_report()


experiment_info = pd.read_csv("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/sub-01/ses-perceptionTraining01/func/sub-01 ses-perceptionTraining01 func sub-01_ses-perceptionTraining01_task-perception_run-01_events.tsv",sep='\t')
conditions = experiment_info['stimulus_id']
condition_mask = conditions.isin([conditions[1]])
fmri_masked = fmri_masked[condition_mask]


mask_directory = "/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/sourcedata/ds001246-download-source/sub-01/anat/ventral_masks"
mask_str = ['V1', 'V2', 'V3', 'V4']
hemispheres = ['LH', 'RH']
brain_area = {}
for substring in mask_str:
    hemisphere_data = {}
    for hemisphere in hemispheres:
        for root, subdirs, files in os.walk(mask_directory):
            for filename in files:
                if substring in filename and hemisphere in filename:
                    mask_path = os.path.join(root, filename)

                mask = nib.load(mask_path)
                runs = 10
                stimuli_list = []
                all_conditions = pd.Series([],dtype=pd.StringDtype())
                all_imgs = np.empty([1,1])
                for r in range(runs):
                    if r == 9:
                        img_prec = nib.load(
                            "/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/derivatives/preproc-spm/output/sub-01/ses-perceptionTraining01/func/sub-01_ses-perceptionTraining01_task-perception_run-" + str(
                                r + 1) + "_bold_preproc.nii.gz")
                        experiment_info = pd.read_csv(
                            "/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/sub-01/ses-perceptionTraining01/func/sub-01 ses-perceptionTraining01 func sub-01_ses-perceptionTraining01_task-perception_run-" + str(
                                r + 1) + "_events.tsv", sep='\t')
                    else:
                        img_prec = nib.load("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/derivatives/preproc-spm/output/sub-01/ses-perceptionTraining01/func/sub-01_ses-perceptionTraining01_task-perception_run-0"+str(r+1)+"_bold_preproc.nii.gz")
                        experiment_info = pd.read_csv("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/sub-01/ses-perceptionTraining01/func/sub-01 ses-perceptionTraining01 func sub-01_ses-perceptionTraining01_task-perception_run-0"+str(r+1)+"_events.tsv",sep='\t')
                    conditions = experiment_info['category_index']
                    stimuli = experiment_info['stimulus_name']
                    stimuli_list.append(stimuli[1:-1])
                    all_conditions = pd.concat([all_conditions, conditions[1:-1]])
                    all_conditions = all_conditions.reset_index(drop=True)
                    img_prec_trunc = img_prec.slicer[:, :, :, 10:]
                    masker = NiftiMasker(mask_img=mask, standardize=True)
                    fmri_masked = masker.fit_transform(img_prec_trunc)
                    if r==0:
                        all_imgs = fmri_masked
                    else:
                        all_imgs = np.vstack((all_imgs,fmri_masked))

                conditions_rep = all_conditions.repeat(3)
                cond_trial = pd.DataFrame({'trial_no': range(1,len(conditions_rep)+1), 'category_index': conditions_rep})
                cond_trial = cond_trial.sort_values(by ='category_index')
                cond_trial = cond_trial.reset_index(drop=True)
                trial_order = cond_trial['trial_no']
                trial_order = trial_order-1
                img_prec_ordered = all_imgs[trial_order]
                img_list = []
                img_cat = np.empty([1,1])
                for i in range(img_prec_ordered.shape[0]):
                    if i==0:
                        img_cat = img_prec_ordered[i,:]
                    elif cond_trial['category_index'][i-1] == cond_trial['category_index'][i]:
                        img_cat = np.column_stack((img_cat,img_prec_ordered[i,:]))
                    elif cond_trial['category_index'][i-1] != cond_trial['category_index'][i]:
                        img_list.append(img_cat)
                        img_cat = img_prec_ordered[i,:]
                img_list.append(img_cat)
                del img_cat
                for i in range(len(img_list)):
                    img_list[i] = np.mean(img_list[i],axis=1)
                cond = all_conditions.sort_values()
                cond = cond.drop_duplicates()
                cond = cond.reset_index(drop=True)
                cond_data = pd.DataFrame({'data': img_list,'category_index': cond})
                hemisphere_data.update({hemisphere: cond_data})
    brain_area.update({substring: hemisphere_data})

import pickle
a_file = open("./kamitani_fmri_data/brain_area_data_sub1_training1.pkl", "wb")
pickle.dump(brain_area, a_file)
a_file.close()

a_file = open("./kamitani_fmri_data/brain_area_data_sub1_training1.pkl", "rb")
brain_area = pickle.load(a_file)

#get stimuli list
runs = 10
stimuli_list = []
for r in range(runs):
    if r == 9:
        img_prec = nib.load(
            "/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/derivatives/preproc-spm/output/sub-01/ses-perceptionTraining01/func/sub-01_ses-perceptionTraining01_task-perception_run-" + str(
                r + 1) + "_bold_preproc.nii.gz")
        experiment_info = pd.read_csv(
            "/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/sub-01/ses-perceptionTraining01/func/sub-01 ses-perceptionTraining01 func sub-01_ses-perceptionTraining01_task-perception_run-" + str(
                r + 1) + "_events.tsv", sep='\t')
    else:
        img_prec = nib.load("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/derivatives/preproc-spm/output/sub-01/ses-perceptionTraining01/func/sub-01_ses-perceptionTraining01_task-perception_run-0"+str(r+1)+"_bold_preproc.nii.gz")
        experiment_info = pd.read_csv("/home/ajay/Documents/Learning_Rule/kamitani_fmri_data/sub-01/ses-perceptionTraining01/func/sub-01 ses-perceptionTraining01 func sub-01_ses-perceptionTraining01_task-perception_run-0"+str(r+1)+"_events.tsv",sep='\t')
    conditions = experiment_info['category_index']
    conditions = conditions[1:-1]
    stimuli = experiment_info['stimulus_name']
    stimuli = stimuli[1:-1]
    stim_info = pd.concat([stimuli, conditions], axis=1)
    stimuli_list.append(stim_info)

stimulus_info = []
for i in range(len(stimuli_list)):
    for n in range(55):
        thing = stimuli_list[i].iloc[n]
        thing1 = thing[0]
        thing2 = thing[1]
        if i == 0:
            stimulus_info.append([thing1, thing2])
        elif thing2 in stimulus_info:
            print('no')
            continue
        else:
            stimulus_info.append([thing1, thing2])

out = pd.DataFrame.from_records(stimulus_info)
out = out.sort_values(by =out.columns[1])
out = out.reset_index(drop=True)
out = out.drop_duplicates(subset=out.columns[1], keep="last")
out = out.reset_index(drop=True)
stimulus_IDs_full = out[out.columns[0]].tolist()
category_IDs_full = out[out.columns[1]].tolist()
stimulus_IDs = []
for i in range(len(stimulus_IDs_full)):
    stimulus_IDs.append(stimulus_IDs_full[i][0:9])
import pickle
with open("stimulus_IDs_full.txt", "wb") as fp:   #Pickling
    pickle.dump(stimulus_IDs_full, fp)
text_out = ''
for i in range(len(stimulus_IDs)):
    text_out = text_out+stimulus_IDs[i]+' '

with open("./kamitani_fmri_data/kamitani_training_sub1_imagenet_IDs.txt", "w") as output:
    output.write(text_out)

class_names = pd.read_csv('./imagenet_classnames.csv',delimiter=',')
stim = class_names['synid'].to_list()
class_names = class_names['class_name']

stimulus_ind = []
for i in range(len(stimulus_IDs)):
    stimulus_ind.append(stim.index(stimulus_IDs[i]))

class_names = class_names.iloc[stimulus_ind]
class_names = class_names.reset_index(drop=True)
with open("./kamitani_fmri_data/kamitani_category_names_sub1.txt", "wb") as f0:  # Pickling
    pickle.dump(class_names.to_list(), f0)


#redownload
#for loop: for each folder,
# (1) load data and set all in folder to batch size (see command below)
# (2) get activations for each (make list)
# (3) average together all activations
# (4) save as per current code

for filename in os.listdir('./data/kamitani_images/image_per_category/'):
    batch_size = len([image_name for image_name in os.listdir(str('./data/kamitani_images/image_per_category/'+filename)) if os.path.isfile(image_name)])
    print(str(batch_size))
training_data_path = '/home/ajay/Documents/Learning_Rule/data/kamitani_images/training'
training_dest_path = '/home/ajay/Documents/Learning_Rule/data/kamitani_images/image_per_category/'
from shutil import copyfile
for i in range(len(category_IDs_full)):
    if not os.path.exists(training_dest_path+'/'+str(int(category_IDs_full[i]))):
        os.makedirs(training_dest_path+'/'+str(int(category_IDs_full[i])))
for i in range(len(stimulus_IDs_full)):
    src = training_data_path+'/'+stimulus_IDs_full[i]+'.JPEG'
    copyfile(src, os.path.join(training_dest_path, str(int(category_IDs_full[i]))+'/'+stimulus_IDs_full[i]))

