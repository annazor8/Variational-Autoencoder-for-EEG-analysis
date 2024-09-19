import mne
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage
import pandas as pd
import torch 
from pathlib import Path
import pickle
import mpld3
from library.config import config_model as cm
from library.model import hvEEGNet

train_session='Shuffle6'
#train_session=10
#load the test data
data = np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))

test_data=data['train_data']

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
#load the Reconstruction error with average_channels  and average_time_samples
model_config = cm.get_config_hierarchical_vEEGNet(22, 1000)
model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
model.load_state_dict(torch.load('/home/azorzetto/trainShuffle9/model_weights_backup_shuffle9/model_epoch80.pth', map_location= torch.device('cpu')))

model.eval()
trial=50
rec_array=[]
for i in range(test_data.shape[0]):
    x=test_data[i,:,:,:]
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x=torch.unsqueeze(x, 1)
    x_eeg_rec=model.reconstruct(x)
    rec_array.append(x_eeg_rec)
    t = torch.linspace(0, 4, 1000).numpy()  # Convert to NumPy array if needed
    trial_eeg=np.squeeze(x)
    x_eeg_rec=np.squeeze(x_eeg_rec)

    for idx_ch, ch in enumerate(new_channel_names):
        # Plot the original and reconstructed signal
        plt.rcParams.update()
        fig, ax = plt.subplots()  # Adjust figsize for better visibility
        ax.plot(t, trial_eeg.squeeze()[idx_ch].squeeze(), label=f'Original EEG - channel {ch}', color='black', linewidth=0.5)
        ax.plot(t, x_eeg_rec.squeeze()[idx_ch].squeeze(), label=f'Reconstructed EEG - channel {ch}', color='red', linewidth=0.5)

        ax.legend(loc='upper right')
        ax.set_xlim([0, 4])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r"Amplitude [$\mu$V]")
        ax.grid(True)
        # Adjust layout
        plt.tight_layout()

        output_path = Path('/home/azorzetto/train{}/img_train_trial/trial_{}_channel_{}.png'.format(train_session, i, ch))
        plt.savefig(output_path)
        mpld3.save_html(plt.gcf(), "/home/azorzetto/train{}/img_train_trial/trial_{}_channel_{}.html".format(train_session, i, ch))
        plt.close()


"""x_rec_complete=np.concatenate(rec_array)
x_rec_complete=x_rec_complete.flatten()
test_data=test_data.flatten()

plt.figure(figsize=(12, 6))  # Optional: Set the size of the plot
plt.hist(test_data, bins=300, color='blue', edgecolor='black')  # bins sets the number of bars in the histogram
# Adding titles and labels
plt.yscale('log')
plt.xlim((-300, 300))
plt.title('Histogram of train data')
plt.xlabel('Value')
plt.ylabel('Frequency')
output_path = Path('/home/azorzetto/train{}/hist/train_data_original.png'.format(train_session))
plt.savefig(output_path)

plt.figure(figsize=(12, 6))  # Optional: Set the size of the plot
plt.hist(x_rec_complete, bins=300, color='blue', edgecolor='black')  # bins sets the number of bars in the histogram
# Adding titles and labels
plt.yscale('log')
plt.xlim((-300, 300))
plt.title('Histogram of reconstructed train data')
plt.xlabel('Value')
plt.ylabel('Frequency')
output_path = Path('/home/azorzetto/train{}/hist/train_data_Reconstructed.png'.format(train_session))
plt.savefig(output_path)"""