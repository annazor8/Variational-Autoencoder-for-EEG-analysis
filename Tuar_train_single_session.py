from collections import defaultdict
import torch
from typing import Dict, List, Tuple
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm
import os
import mne
import numpy as np
import pandas as pd
from tuar_training_utils import leave_one_session_out, reconstruction_metrics
from torch.utils.data import DataLoader
from library.training.loss_function import compute_dtw_loss_along_channels
from library.training.soft_dtw_cuda import SoftDTW
import pickle
import matplotlib.pyplot as plt
import mpld3
import torch 
from pathlib import Path

train_session='8'
np.random.seed(43)
#directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar'
#directory_path="/content"
#directory_path='/home/azorzetto/dataset/01_tcp_ar'
#directory_path = '/home/lmonni/Documents/01_tcp_ar'
directory_path="/home/azorzetto/data1/01_tcp_ar_jrj"

channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                       'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                       'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                       'EEG T1-REF', 'EEG T2-REF']

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'E1', 'E2']

split_mapping={'FP1': 'Fp1', 'FP2':'Fp2', 'F3':'F3', 'F4':'F4', 'C3':'C3', 'C4':'C4', 'P3':'P3', 'P4':'P4', 'O1':'O1', 'O2':'O2', 'F7':'F7', 'T3':'T3', 'T4':'T4', 'T5':'T5', 'T6':'T6',
'A1':'A1', 'A2':'A2', 'FZ':'Fz', 'CZ':'Cz', 'PZ':'Pz', 'E1':'E1', 'E2':'E2'}

# List all files in the directory
all_files = os.listdir(directory_path)
# Filter out only EDF files
edf_files = [file for file in all_files if file.endswith('.edf')]

file_name=edf_files[1]

file_path_edf = os.path.join(directory_path, file_name)
sub_id, session, time = file_name.split(".")[0].split("_")

file_path_csv=file_name.split(".")[0]+".csv"
file_path_csv=os.path.join(directory_path, file_path_csv )
df_artifact=pd.read_csv(file_path_csv, skiprows=6)

ch_names = []

# Iterate over each channel name in the 'channel' column of df_artifact
for i, channel_name in enumerate(df_artifact['channel'].tolist()):
    
    # Split the channel name at the hyphen and take the first part
    base_channel_name = channel_name.split('-')[0]
    
    # Attempt to map the base channel name using the reverted_mapping dictionary
    # If the base_channel_name exists in the dictionary, use the mapped value
    # Otherwise, use the original base_channel_name
    mapped_channel_name = split_mapping.get(base_channel_name)
    if mapped_channel_name==None: #if None it means that the channel is not contained in the list of interesting channels
        df_artifact.drop(index=i,inplace=True)
        continue
    # Append the mapped channel name inside a list (to create a list of lists)
    ch_names.append([mapped_channel_name])
    df_artifact['channel'][i]=mapped_channel_name

df_artifact['duration_artifact'] = df_artifact.iloc[:, 2] - df_artifact.iloc[:, 1]
onset = (df_artifact.iloc[:, 1]).astype(float)
duration=(df_artifact['duration_artifact']).astype(float)   
description=df_artifact['label']
annotations = mne.Annotations(onset=onset, duration=duration,ch_names= ch_names, description=description)

# Load the EDF file
raw_mne = mne.io.read_raw_edf(file_path_edf, preload=True)  # Set preload to True to load data into memory
raw_mne.pick_channels(channels_to_set, ordered=True)
rename_mapping = dict(zip(channels_to_set, new_channel_names))
raw_mne.rename_channels(rename_mapping)
raw_mne.set_annotations(annotations)
# Resample the data to 250 Hz
raw_mne.resample(250)
epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=False)
epoch_data = epochs_mne.get_data(copy=False)

# Extract annotations from raw data
annotations = raw_mne.annotations

# Extract onset times of annotations and convert to sample indices
sfreq = raw_mne.info['sfreq']

# Initialize artifact flag array with zeros
n_epochs, n_channels, n_samples_per_epoch = epoch_data.shape
artifact_flags = np.zeros((n_epochs, n_channels, n_samples_per_epoch), dtype=int)

for index, row in df_artifact.iterrows():
    onset_sample = int(np.round(row['start_time'] * sfreq).astype(int))
    duration_seconds = row['duration_artifact']
    duration_samples =int( np.round(duration_seconds * sfreq).astype(int))
    affected_channels = row['channel']  # List or array of affected channel names

    # Calculate epoch and position within the epoch for the annotation
    epoch_idx = onset_sample // n_samples_per_epoch
    within_epoch_idx = onset_sample % n_samples_per_epoch

    if epoch_idx < n_epochs:
        start_sample = int(within_epoch_idx)
        end_sample = start_sample +  duration_samples # Use duration from the row

        if start_sample < n_samples_per_epoch:
            end_sample = min(end_sample, n_samples_per_epoch)
            
            # Update artifact_flags array only for the affected channels
        ch_idx = new_channel_names.index(affected_channels)  # Find index of channel in raw data
            
        if ch_idx >= 0:  # Ensure the channel exists in the raw data
            artifact_flags[epoch_idx, ch_idx, start_sample:end_sample] = 1
mean=np.mean(epoch_data, axis =(1,2))
plt.rcParams.update()
t = torch.linspace(0, len(mean) - 1, steps=len(mean)).numpy()  # Convert to NumPy array if needed
fig, ax = plt.subplots()  # Adjust figsize for better visibility
ax.plot(t, mean, label='mean values of the trials', color='green', linewidth=1)
ax.legend(loc='upper right')
ax.set_ylabel('mean values')
ax.set_xlabel('trial number')
ax.grid(True)
# Adjust layout
plt.tight_layout()
plt.show(block=True)
output_path = Path('/home/azorzetto/train{}/{}.png'.format(train_session, 'mean'))
#plt.savefig(output_path)
#mpld3.save_html(plt.gcf(), "//home/azorzetto/train{}/{}.html".format(train_session, 'mean'))

std = np.std(epoch_data)
epoch_data = (epoch_data-mean) / std  # normalization for session
del mean
del std
epoch_data = np.expand_dims(epoch_data, 1)  # number of epochs for that signal x 1 x channels x time samples

