import os

#add args to activation files
#create save folders
#

# GD:
for epoch in range(40):
   print('epoch: ' + str(epoch + 1))
   os.system("sudo mkdir GD/epoch_data_919/epoch_" + str(epoch + 1))  # create epoch
   os.system("sudo python network_activations_GD.py " + str(epoch + 1))  # get activations for epoch (in epoch folder)
   os.system("sudo python collect_GD_activations.py " + str(epoch + 1))  # collect into one file (save in epoch folder)
   os.system("sudo python rsa_analysis_epoch.py " + str(epoch + 1) + " GD")  # do rsa (save in epoch data folder)

'''
#Hebb
for epoch in range(40):
   print('epoch: '+str(epoch+1))
   #os.system("sudo mkdir Hebbian/epoch_data/epoch_"+str(epoch+1)) #create epoch
   os.system("sudo python network_activations_hebb.py "+str(epoch+1))  #get activations for epoch (in epoch folder)
   os.system("sudo python rsa_analysis_epoch.py "+str(epoch+1)+" Hebbian")  #do rsa (save in epoch data folder)

#Burstprop
for epoch in range(1, 80):
   print('epoch: '+str(epoch+1))
   #os.system("sudo mkdir Burstprop/epoch_data/epoch_"+str(epoch+1)) #create epoch
   os.system("sudo python network_activations_burst.py "+str(epoch+1))  #get activations for epoch (in epoch folder)
   os.system("sudo python rsa_analysis_epoch.py "+str(epoch+1)+" Burstprop")  #do rsa (save in epoch data folder)
'''