#Gather GD
import pickle
import argparse

'''
parser = argparse.ArgumentParser()
parser.add_argument('epoch_num')
args=parser.parse_args()
epoch_num = args.epoch_num
'''

tags = ["20", "40", "60", "80", "100", "120", "140", "150"]
for tag in tags:
    #with open(str("./GD/epoch_data_919/epoch_"+str(epoch_num)+"/activations_per_category_"+tag+".txt"), "rb") as f:
    with open(str("./GD/final_model_data/activations_per_category_" + tag + ".txt"), "rb") as f:
        activations = pickle.load(f)
    #with open(str("./GD/final_model_data/activations_per_category_" + tag + ".txt"), "rb") as f:
    #    activations = pickle.load(f)
        #print('activation length: '+str(len(activations)))
    if tag == "20":
        act_GD = activations
    else:
        act_GD = act_GD + activations
del activations

'''
with open("./activations/activation_per_category_burst_Hebb.txt", "rb") as f:
    activation_per_category_burst_Hebb =  pickle.load(f)

for i in range(len(activation_per_category_burst_Hebb)):
    activation_per_category_burst_Hebb[i].append(act_GD[i])
'''

#with open("./GD/epoch_data_919/epoch_"+str(epoch_num)+"/activations_per_category.txt", "wb") as f0:  #ADD EPOCH NUM FILE
with open("./GD/final_model_data/activations_per_category.txt","wb") as f0:  # ADD EPOCH NUM FILE
    pickle.dump(act_GD, f0)

#with open("./GD/final_model_data/activations_per_category.txt", "wb") as f0:  #ADD EPOCH NUM FILE
#    pickle.dump(act_GD, f0)
