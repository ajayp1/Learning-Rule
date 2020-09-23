import matplotlib.pyplot as plt
from pyrsa.vis import rdm_plot
from pyrsa.vis.colors import rdm_colormap
import pickle
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    #if isinstance(corr_array, pd.DataFrame):
    #    return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

##RDMs

#fmri data
with open("/home/ajay/Desktop/Learning_Rule_paperspace/kamitani_fmri_data/fmri_brain_area_rsa_.pkl", "rb") as f:
    rsa_fmri =  pickle.load(f)
rdm_fmri = rsa_fmri['V4'][0].get_matrices()

rdm_cluster = cluster_corr(rdm_fmri[0])
cmap = rdm_colormap()

order = np.argsort(rdm_fmri[0,0,:])[np.argsort(np.argsort(rdm_cluster[0,:]))]
with open("./kamitani_fmri_data/kamitani_category_names_sub1.txt", "rb") as f0:  #Import category names
    category_names = pickle.load(f0)

with open("stimuli_perception_order.txt", "wb") as p1:   #Pickling
    pickle.dump(order.tolist(), p1)

thing = [order < 40]

plt.imshow(rdm_cluster, cmap)
ax = plt.gca()
ax.set_xticks(np.arange(150))
ax.set_xticklabels([category_names[i] for i in order.tolist()])
ax.set_yticks(np.arange(150))
ax.set_yticklabels([category_names[i] for i in order.tolist()])
ax.set_yticks(ax.get_yticks()[::2])
#for label in ax.yaxis.get_ticklabels()[::1]:
#    label.set_visible(False)
for label in ax.yaxis.get_ticklabels()[::2]:
    label.set_visible(False)
ax.set_xticks([])
plt.title('fMRI')
# plt.savefig('/home/ajay/Documents/Learning_Rule/saved_plots/fMRI_RDM.png')
plt.show()


#GD
with open("/home/ajay/Desktop/gd_other/GD/GD_brain_area_rsa.pkl", "rb") as f:
    rsa_GD =  pickle.load(f)
rdm_GD = rsa_GD['V4'][0].get_matrices()
rdm_GD = rdm_GD[0]
rdm_GD = rdm_GD[:,order][order]
cmap = rdm_colormap()

plt.imshow(rdm_GD, cmap)
ax = plt.gca()
ax.set_xticks(np.arange(150))
ax.set_xticklabels([category_names[i] for i in order.tolist()])
ax.set_yticks(np.arange(150))
ax.set_yticklabels([category_names[i] for i in order.tolist()])
ax.set_yticks(ax.get_yticks()[::2])
#for label in ax.yaxis.get_ticklabels()[::1]:
#    label.set_visible(False)
for label in ax.yaxis.get_ticklabels()[::2]:
    label.set_visible(False)
ax.set_xticks([])
plt.title('Gradient Descent')
plt.savefig('/home/ajay/Documents/Learning_Rule/saved_plots/GD_RDM.png')
plt.show()



#Burstprop
with open("/home/ajay/Desktop/Learning_Rule_paperspace/Burstprop/final_model_data/Burstprop_brain_area_rsa.pkl", "rb") as f:
    rsa_burst =  pickle.load(f)

rdm_burst = rsa_burst['V4'][0].get_matrices()
rdm_burst = rdm_burst[0]
rdm_burst = rdm_burst[:,order][order]
cmap = rdm_colormap()

plt.imshow(rdm_burst, cmap)
ax = plt.gca()
ax.set_xticks(np.arange(150))
ax.set_xticklabels([category_names[i] for i in order.tolist()])
ax.set_yticks(np.arange(150))
ax.set_yticklabels([category_names[i] for i in order.tolist()])
ax.set_yticks(ax.get_yticks()[::2])
#for label in ax.yaxis.get_ticklabels()[::1]:
#    label.set_visible(False)
for label in ax.yaxis.get_ticklabels()[::2]:
    label.set_visible(False)
ax.set_xticks([])
plt.title('Burstprop')
plt.savefig('/home/ajay/Documents/Learning_Rule/saved_plots/Burstprop_RDM.png')
plt.show()


#Hebbian
with open("/home/ajay/Desktop/Learning_Rule_paperspace/Hebbian/final_model_data/Hebbian_brain_area_rsa.pkl", "rb") as f:
    rsa_hebb =  pickle.load(f)

rdm_hebb = rsa_hebb['V4'][0].get_matrices()
rdm_hebb = rdm_hebb[0]
rdm_hebb = rdm_hebb[:,order][order]
cmap = rdm_colormap()

plt.imshow(rdm_hebb, cmap)
ax = plt.gca()
ax.set_xticks(np.arange(150))
ax.set_xticklabels([category_names[i] for i in order.tolist()])
ax.set_yticks(np.arange(150))
ax.set_yticklabels([category_names[i] for i in order.tolist()])
ax.set_yticks(ax.get_yticks()[::2])
#for label in ax.yaxis.get_ticklabels()[::1]:
#    label.set_visible(False)
for label in ax.yaxis.get_ticklabels()[::2]:
    label.set_visible(False)
ax.set_xticks([])
plt.title('Hebbian')
plt.savefig('/home/ajay/Documents/Learning_Rule/saved_plots/Hebbian_RDM.png')
plt.show()


#RSA Bar Charts

rsa_GD_sim = np.empty([4])
for index, key in enumerate(rsa_GD):
    rsa_GD_sim[index] = rsa_GD[key][1]

rsa_burst_sim = np.empty([4])
for key in rsa_burst.keys():
    rsa_burst_sim[list(rsa_burst).index(key)] = rsa_burst[key][1]

rsa_hebb_sim = np.empty([4])
for key in rsa_hebb.keys():
    rsa_hebb_sim[list(rsa_hebb).index(key)] = rsa_hebb[key][1]


fig, ax = plt.subplots()
ind = np.arange(4)   # the x locations for the groups
width = 0.25        # the width of the bars
r1 = np.arange(len(rsa_GD_sim))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]
p1 = ax.bar(ind, rsa_GD_sim, width, bottom=0)
p2 = ax.bar(r2, rsa_burst_sim, width, bottom=0)
p3 = ax.bar(r3, rsa_hebb_sim, width, bottom=0)

ax.set_title('Similarity to brain activity')
ax.set_xticks([r + width for r in range(len(rsa_GD_sim))])
ax.set_xticklabels(('V1', 'V2', 'V3', 'V4'))
plt.xlabel('Brain Area', fontsize=12)
plt.ylabel('Cosine Similarity')

ax.legend((p1[0], p2[0], p3[0]), ('Gradient Descent', 'Burstprop', 'Hebbian'))
ax.autoscale_view()
plt.savefig('/home/ajay/Documents/Learning_Rule/saved_plots/similarity_ventral_bar.png')
plt.show()
