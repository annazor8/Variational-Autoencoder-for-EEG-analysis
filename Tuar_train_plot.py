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

train_session='Shuffle_jrj2'
#load the test data
data = np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))

train_data=data['train_data']

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
#load the Reconstruction error with average_channels  and average_time_samples
model_config = cm.get_config_hierarchical_vEEGNet(22, 1000)
model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
model.load_state_dict(torch.load('/home/azorzetto/trainShuffle_jrj2/model_weights_backup_shuffle_jrj2/model_epoch84.pth', map_location= torch.device('cpu')))
train_data=train_data[50,:,:,:]
train_data = train_data.astype(np.float32)
train_data = torch.from_numpy(train_data)
train_data=torch.unsqueeze(train_data, 1)
x_eeg_rec=model.reconstruct(train_data)

t = torch.linspace(0, 4, 1000).numpy()  # Convert to NumPy array if needed
trial_eeg=np.squeeze(train_data)
x_eeg_rec=np.squeeze(x_eeg_rec)

for idx_ch, ch in enumerate(new_channel_names):
    # Plot the original and reconstructed signal
    plt.rcParams.update()
    fig, ax = plt.subplots()  # Adjust figsize for better visibility
    ax.plot(t, trial_eeg.squeeze()[idx_ch].squeeze(), label=f'Original EEG - channel {ch}', color='green', linewidth=2)
    ax.plot(t, x_eeg_rec.squeeze()[idx_ch].squeeze(), label=f'Reconstructed EEG - channel {ch}', color='red', linewidth=1)

    ax.legend(loc='upper right')
    ax.set_xlim([0, 4])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r"Amplitude [$\mu$V]")
    ax.grid(True)
    # Adjust layout
    plt.tight_layout()

    output_path = Path('/home/azorzetto/train{}/img_train_trial/{}.png'.format(train_session, ch))
    plt.savefig(output_path)
    mpld3.save_html(plt.gcf(), "//home/azorzetto/train{}/img_train_trial/{}.html".format(train_session, ch))
    plt.close()