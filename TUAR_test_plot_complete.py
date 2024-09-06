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


train_session='8'
#load the test data
data = np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))
x_r_eeg1=np.load('/home/azorzetto/train{}/reconstructed_eeg.npz'.format(train_session))

x_r_eeg1=x_r_eeg1['x_r_eeg']
test_data=data['test_data']

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
t = torch.linspace(0, 4, 1000).numpy()  # Convert to NumPy array if needed

for trial in range(x_r_eeg1.shape[0]):
    print('trial number:')
    print(trial)
    if(np.isnan(x_r_eeg1[trial]).any()):
        x_eeg= test_data[trial] #.float().double() #nd array
        x_r_eeg = x_r_eeg1[trial]

        for idx_ch, ch in enumerate(new_channel_names):
            print('channel')
            print(idx_ch)
            # Plot the original and reconstructed signal
            plt.rcParams.update()
            fig, ax = plt.subplots()  # Adjust figsize for better visibility
            ax.plot(t, x_eeg.squeeze()[idx_ch].squeeze(), label=f'Original EEG - trial {trial}-channel {ch}', color='green', linewidth=1)
            #ax.plot(t, x_r_eeg.squeeze()[idx_ch].squeeze(), label=f'Reconstructed EEG - trial {trial}- channel {ch}', color='red', linewidth=1)

            ax.legend(loc='upper right')
            ax.set_xlim([0, 4])
            ax.set_ylim([-10, 10])
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(r"Amplitude [$\mu$V]")
            ax.grid(True)
            # Adjust layout
            plt.tight_layout()
            output_path = Path('/home/azorzetto/train{}/all_trials2/trial{}_{}.png'.format(train_session, trial, ch))
            plt.savefig(output_path)
            mpld3.save_html(plt.gcf(), "//home/azorzetto/train{}/all_trials2/trial{}_{}.html".format(train_session, trial, ch))
            plt.close()
        else:
            continue