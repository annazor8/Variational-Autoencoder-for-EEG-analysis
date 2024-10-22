"""
Create a binary image which indicates the presence or absence of artifacts in the dataset
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
from library.config import config_model as cm
from library.model import hvEEGNet
import torch
from library.analysis import dtw_analysis

train_session=13
dataset=np.load(f"/home/azorzetto/data1/Train/train{train_session}/dataset.npz")

#dataset=np.load("/home/azorzetto/train{}/dataset.npz".format(train_session))

train_data=dataset["train_data"]
test_data=dataset["test_data"]


model_epoch=160

path_to_model="/home/azorzetto/data1/Train/train{}/model_weights_backup{}/model_epoch{}.pth".format(train_session, train_session, model_epoch)

#-------------------------------------------------------------------------------------------------------------
    #reconstruction of the trials 
model_config = cm.get_config_hierarchical_vEEGNet(22, 1000)
model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
model.load_state_dict(torch.load(path_to_model, map_location= torch.device('cpu')))
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"

E_test= np.full((22, test_data.shape[0]), np.inf)
test_reconstructed=[]
for j in range(test_data.shape[0]):
    x_eeg_test=test_data[j,:,:,:]
    x_eeg_test = x_eeg_test.astype(np.float32)
    x_eeg_test = torch.from_numpy(x_eeg_test).unsqueeze(0)
    x_eeg_test_reconstructed=model.reconstruct(x_eeg_test)

    test_reconstructed.append(x_eeg_test_reconstructed) #ogni elemento è ([1, 1, 22, 1000])
    dtw_test=dtw_analysis.compute_recon_error_between_two_tensor(x_eeg_test, x_eeg_test_reconstructed,device=device, 
                                                                 average_channels=False, average_time_samples=True)
    E_test[:,j]=dtw_test.squeeze()

test_reconstructed=np.concatenate(test_reconstructed) #trailsx1x22x1000

E_train= np.full((22, train_data.shape[0]), np.inf)
train_reconstructed=[]
for t in range(train_data.shape[0]):
    x_eeg_train=train_data[t,:,:,:]
    x_eeg_train = x_eeg_train.astype(np.float32)
    x_eeg_train = torch.from_numpy(x_eeg_train).unsqueeze(0)
    x_eeg_train_reconstructed=model.reconstruct(x_eeg_train)
    train_reconstructed.append(x_eeg_train_reconstructed) #ogni elemento è ([1, 1, 22, 1000])
    dtw_train=dtw_analysis.compute_recon_error_between_two_tensor(x_eeg_train, x_eeg_train_reconstructed,device=device, 
                                                                 average_channels=False, average_time_samples=True)
    E_train[:,t]=dtw_train.squeeze()

train_reconstructed=np.concatenate(train_reconstructed) #trailsx1x22x1000

np.savez_compressed('reconstructed_dataset.npz', E_train=E_train, E_test=E_test)