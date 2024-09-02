import mne
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage
from library.analysis.dtw_analysis import compute_recon_error_between_two_tensor, compute_divergence_SDTW_recon_error_between_two_tensor
import pandas as pd
import torch 
from pathlib import Path
import pickle
import mpld3
from library.config import config_model as cm
from library.model import hvEEGNet
import torch.nn.functional as F

train_session='Shuffle_jrj'
#load the test data
data = np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))

train_data=data['train_data']
validation_data=data['validation_data']
test_data=data['test_data']

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']

final_train_mse_array=[]
final_val_mse_array=[]
final_test_mse_array=[]
final_train_sdtw_array=[]
final_val_sdtw_array=[]
final_test_sdtw_array=[]
final_train_divergence_array=[]
final_validation_divergence_array=[]
final_test_divergence_array=[]

for i in range(80):
    #load the Reconstruction error with average_channels  and average_time_samples
    model_config = cm.get_config_hierarchical_vEEGNet(22, 1000)
    model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
    model.load_state_dict(torch.load('/home/azorzetto/trainShuffle_jrj/model_weights_backup_shuffle_jrj/model_epoch{}.pth'.format(i+1), map_location= torch.device('cpu')))
    train_mse_array=[]
    val_mse_array=[]
    test_mse_array=[]
    train_sdtw_array=[]
    val_sdtw_array=[]
    test_sdtw_array=[]
    train_divergence_array=[]
    validation_divergence_array=[]
    test_divergence_array=[]

    for i in range(train_data.shape[0]):
        x_eeg=train_data[i,:,:,:]
        x_eeg = x_eeg.astype(np.float32)
        x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
        #train_data=torch.unsqueeze(train_data, 1)
        train_rec=model.reconstruct(x_eeg)
        train_mse_array.append(F.mse_loss(x_eeg, train_rec))
        train_sdtw_array.append(compute_recon_error_between_two_tensor(x_eeg, train_rec, average_channels=True, average_time_samples=True))
        train_divergence_array.append(compute_divergence_SDTW_recon_error_between_two_tensor(x_eeg, train_rec, average_channels=True, average_time_samples=True))
        print(i)

    final_train_mse_array.append(np.mean(train_mse_array))
    final_train_sdtw_array.append(np.mean(train_sdtw_array))
    final_train_divergence_array.append(np.mean(train_divergence_array))

    for i in range(validation_data.shape[0]):
        x_eeg=validation_data[i,:,:,:]
        x_eeg = x_eeg.astype(np.float32)
        x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
        val_rec=model.reconstruct(x_eeg)
        val_mse_array.append(F.mse_loss(x_eeg, val_rec))
        val_sdtw_array.append(compute_recon_error_between_two_tensor(x_eeg, val_rec, average_channels=True, average_time_samples=True))
        validation_divergence_array.append(compute_divergence_SDTW_recon_error_between_two_tensor(x_eeg, val_rec, average_channels=True, average_time_samples=True))
        print(i)

    final_val_mse_array.append(np.mean(val_mse_array))
    final_val_sdtw_array.append(np.mean(val_sdtw_array))
    final_validation_divergence_array.append(np.mean(validation_divergence_array))

    for i in range(test_data.shape[0]):
        x_eeg=test_data[i,:,:,:]
        x_eeg = x_eeg.astype(np.float32)
        x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
        test_rec=model.reconstruct(x_eeg)
        test_mse_array.append(F.mse_loss(x_eeg, test_rec))
        test_sdtw_array.append(compute_recon_error_between_two_tensor(x_eeg, test_rec, average_channels=True, average_time_samples=True))
        test_divergence_array.append(compute_divergence_SDTW_recon_error_between_two_tensor(x_eeg, test_rec, average_channels=True, average_time_samples=True))
        print(i)

    final_test_mse_array.append(np.mean(test_mse_array))
    final_test_sdtw_array.append(np.mean(test_sdtw_array))
    final_test_divergence_array.append(np.mean(test_divergence_array))

t_train = torch.linspace(0, len(final_train_mse_array) - 1, steps=len(final_train_mse_array)).numpy()  # Convert to NumPy array if needed
t_val = torch.linspace(0, len(final_val_mse_array) - 1, steps=len(final_val_mse_array)).numpy()  # Convert to NumPy array if needed
t_test = torch.linspace(0, len(final_test_mse_array) - 1, steps=len(final_test_mse_array)).numpy()  # Convert to NumPy array if needed

plt.rcParams.update()
fig, ax = plt.subplots()  # Adjust figsize for better visibility
ax.plot(t_train, final_train_mse_array, label='MSE on the train set', color='green', linewidth=1)
ax.plot(t_val, final_val_mse_array, label='MSE on the validation set', color='red', linewidth=1)
ax.plot(t_test, final_test_mse_array, label='MSE on the test set', color='blue', linewidth=1)
ax.legend(loc='upper right')
ax.set_ylabel('MSE')
ax.set_xlabel('trial number')
ax.grid(True)
# Adjust layout
plt.tight_layout()

output_path = Path('/home/azorzetto/train{}/img_rec_ability_over_epochs/{}.png'.format(train_session, 'MSE'))
plt.savefig(output_path)
mpld3.save_html(plt.gcf(), "//home/azorzetto/train{}/img_rec_ability_over_epochs/{}.html".format(train_session, 'MSE'))
plt.close()

plt.rcParams.update()
fig, ax = plt.subplots()  # Adjust figsize for better visibility
ax.plot(t_train, final_train_sdtw_array, label='Soft DTW on the train set', color='green', linewidth=1)
ax.plot(t_val, final_val_sdtw_array, label='Soft DTW on the validation set', color='red', linewidth=1)
ax.plot(t_test, final_test_sdtw_array, label='Soft DTW on the test set', color='blue', linewidth=1)
ax.legend(loc='upper right')
ax.set_ylabel('Soft DTW')
ax.set_xlabel('trial number')
ax.grid(True)
# Adjust layout
plt.tight_layout()

output_path = Path('/home/azorzetto/train{}/img_rec_ability_over_epochs/{}.png'.format(train_session, 'Soft_DTW'))
plt.savefig(output_path)
mpld3.save_html(plt.gcf(), "//home/azorzetto/train{}/img_rec_ability_over_epochs/{}.html".format(train_session, 'Soft_DTW'))
plt.close()

plt.rcParams.update()
fig, ax = plt.subplots()  # Adjust figsize for better visibility
ax.plot(t_train, final_train_divergence_array, label='divergence DTW on the train set', color='green', linewidth=1)
ax.plot(t_val, final_validation_divergence_array, label='divergence DTW on the validation set', color='red', linewidth=1)
ax.plot(t_test, final_test_divergence_array, label='divergence DTW on the test set', color='blue', linewidth=1)
ax.legend(loc='upper right')
ax.set_ylabel('Soft DTW')
ax.set_xlabel('trial number')
ax.grid(True)
# Adjust layout
plt.tight_layout()

output_path = Path('/home/azorzetto/train{}/img_rec_ability_over_epochs/{}.png'.format(train_session, 'Soft_DTW_divergence'))
plt.savefig(output_path)
mpld3.save_html(plt.gcf(), "//home/azorzetto/train{}/img_rec_ability_over_epochs/{}.html".format(train_session, 'Soft_DTW_divergence'))
plt.close()