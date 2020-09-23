import matplotlib.pyplot as plt
from pyrsa.vis import rdm_plot
from pyrsa.vis.colors import rdm_colormap
import pickle
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch


font_size = 17
#Learning Curves
#GD

with open("/home/ajay/Desktop/gd_other/GD/test_acc_hist_GD_919.txt", "rb") as f:
    test_acc_GD =  pickle.load(f)

with open("/home/ajay/Desktop/gd_other/GD/loss_hist_test_GD_919.txt", "rb") as f:
    test_err_GD =  pickle.load(f)

with open("/home/ajay/Desktop/gd_other/GD/loss_history_GD_train_919.txt", "rb") as f:
    train_err_GD =  pickle.load(f)

#f, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 3, 3]})
ax1 = plt.gca()
ax1.plot(test_err_GD, label='test')
ax1.plot(train_err_GD, label='train')
# plt.legend()
ax1.set_title('Gradient Descent')
plt.ylabel('Loss', fontsize=font_size)
plt.savefig('/home/ajay/Documents/Learning_Rule/saved_plots/GD_learning_curve.png')
plt.show()


#Burstprop
with open("/home/ajay/Desktop/Learning_Rule_paperspace/Burstprop/test_hist_burstprop.txt", "rb") as f:
    test_error_burst =  pickle.load(f)

with open("/home/ajay/Desktop/Learning_Rule_paperspace/Burstprop/train_hist_burstprop.txt", "rb") as f:
    train_error_burst =  pickle.load(f)




train_err_burst, test_err_burst = [], []
for i in range(len(train_error_burst)):
    train_err_burst.append(train_error_burst[i][2])
    test_err_burst.append(test_error_burst[i][2])

ax2 = plt.gca()
ax2.plot(train_err_burst, label='train')
ax2.plot(test_err_burst, label='test')
# plt.legend()
ax2.set_title('Burstprop')
plt.xlabel('Epoch', fontsize=font_size)
plt.savefig('/home/ajay/Documents/Learning_Rule/saved_plots/Burstprop_learning_curve.png')
plt.show()

#Hebbian
with open("/home/ajay/Desktop/Learning_Rule_paperspace/Hebbian/918_data/test_loss_hist_Hebb_918.txt", "rb") as f:
    test_err_hebb =  pickle.load(f)

with open("/home/ajay/Desktop/Learning_Rule_paperspace/Hebbian/918_data/train_loss_hist_Hebb_918.txt", "rb") as f:
    train_err_hebb =  pickle.load(f)

ax3 = plt.gca()
ax3.plot(train_err_hebb, label='train')
ax3.plot(test_err_hebb, label='test')
plt.legend()
ax3.set_title('Hebbian')
# ax2.set_xlabel('Epochs')
# plt.set_ylabel('Loss')
plt.savefig('/home/ajay/Documents/Learning_Rule/saved_plots/Hebbian_learning_curve.png')
plt.show()