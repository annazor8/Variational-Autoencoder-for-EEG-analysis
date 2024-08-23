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


train_session='Shuffle_jrj2'
#load the test data
data = np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))
x_r_eeg=np.load('/home/azorzetto/train{}/reconstructed_eeg.npz'.format(train_session))

x_r_eeg=x_r_eeg['x_r_eeg']
test_data=data['test_data']

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
#load the Reconstruction error with average_channels  and average_time_samples
df_mean_reconstruction_err = pd.read_csv('/home/azorzetto/train{}/mean_reconstruction_errors_error_epoch62.csv'.format(train_session))
mean_reconstruction_err = df_mean_reconstruction_err.iloc[:, 0].to_numpy()

min_rec_error=np.min(mean_reconstruction_err)
indx_min_rec_error=np.argmin(mean_reconstruction_err)
max_rec_error=np.max(mean_reconstruction_err)
indx_max_rec_error=np.argmax(mean_reconstruction_err)

#x_eeg=test_data[indx_min_rec_error]
#test_data[indx_max_rec_error]


"""x_eeg_min = test_data[indx_min_rec_error] #.float().double() #nd array

x_r_eeg_min = x_r_eeg[indx_min_rec_error] #ndarray
with open('/home/azorzetto/train{}/resconstruction_error.pkl'.format(train_session), 'rb') as file:
    reconstruction_error = pickle.load(file)

best_ch=np.argmin(reconstruction_error[indx_min_rec_error] ['Reconstruction error with no average_channels and average_time_samples'])
print("the channel with the smallest reconstruction error is {}".format(new_channel_names[best_ch]))
t = torch.linspace(0, 4, 1000).numpy()  # Convert to NumPy array if needed"""

x_eeg_min = test_data[indx_min_rec_error] #.float().double() #nd array

x_r_eeg_min = x_r_eeg[indx_min_rec_error] #ndarray
with open('/home/azorzetto/train{}/resconstruction_error_epoch62.pkl'.format(train_session), 'rb') as file:
    reconstruction_error = pickle.load(file)

best_ch=np.argmin(reconstruction_error[indx_min_rec_error] ['Reconstruction error with no average_channels and average_time_samples'])
print("the channel with the smallest reconstruction error is {}".format(new_channel_names[best_ch]))
t = torch.linspace(0, 4, 1000).numpy()  # Convert to NumPy array if needed

for idx_ch, ch in enumerate(new_channel_names):
    # Plot the original and reconstructed signal
    plt.rcParams.update()
    fig, ax = plt.subplots()  # Adjust figsize for better visibility
    ax.plot(t, x_eeg_min.squeeze()[idx_ch].squeeze(), label=f'Original EEG - channel {ch}', color='green', linewidth=2)
    ax.plot(t, x_r_eeg_min.squeeze()[idx_ch].squeeze(), label=f'Reconstructed EEG - channel {ch}', color='red', linewidth=1)

    ax.legend(loc='upper right')
    ax.set_xlim([0, 4])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r"Amplitude [$\mu$V]")
    ax.grid(True)
    # Adjust layout
    plt.tight_layout()

    output_path = Path('/home/azorzetto/train{}/img_min_error_epoch62/{}.png'.format(train_session, ch))
    plt.savefig(output_path)
    mpld3.save_html(plt.gcf(), "//home/azorzetto/train{}/img_min_error_epoch62/{}.html".format(train_session, ch))
    plt.close()


x_eeg_max=test_data[indx_max_rec_error]
x_r_eeg_max=x_r_eeg[indx_max_rec_error]


for idx_ch, ch in enumerate(new_channel_names):
    # Plot the original and reconstructed signal
    plt.rcParams.update()
    fig, ax = plt.subplots()

    ax.plot(t, x_eeg_max.squeeze()[idx_ch].squeeze(), label = 'Original EEG with maximum reconstruction error - channel ={}'.format(ch), color = 'green', linewidth = 2)
    ax.plot(t, x_r_eeg_max.squeeze()[idx_ch].squeeze(), label = 'Reconstructed EEG with maximum reconstruction error - channel ={}'.format(ch), color = 'red', linewidth = 1)

    ax.legend()
    ax.set_xlim([0, 4])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r"Amplitude [$\mu$V]")
    ax.grid(True)
    fig.tight_layout()

    output_path = Path('/home/azorzetto/train{}/img_max_error_epoch62/{}.png'.format(train_session, ch))
    plt.savefig(output_path)
    mpld3.save_html(plt.gcf(), "//home/azorzetto/train{}/img_max_error_epoch62/{}.html".format(train_session, ch))

worse_ch=np.argmax(reconstruction_error[indx_max_rec_error] ['Reconstruction error with no average_channels and average_time_samples'])
print("the channel with the highest reconstruction error is {}".format(new_channel_names[worse_ch]))
