"questa era una bozza di training per un dataset alternativo"
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from library.analysis import dtw_analysis
from library.training.soft_dtw_cuda import SoftDTW
from library.dataset import preprocess as pp
from library.model import hvEEGNet
import matplotlib.pyplot as plt
from pathlib import Path
import mne
import torch
from numpy.typing import NDArray
from typing import Dict, List, Tuple
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm
from tuar_training_utils import get_data_TUAR, normalize_to_range
import os
import mne
import numpy as np
import random
import pandas as pd
from statistics_TUAR import Calculate_statistics
from collections import defaultdict
from library.analysis import dtw_analysis
from typing import Dict
import os
import mne
import torch
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm
import os
import numpy as np
import pandas as pd 
from tuar_training_utils import reconstruction_metrics
from torch.utils.data import DataLoader
import pickle

np.random.seed(43)
    
directory_path='/home/azorzetto/dataset/alternative_dataset/EEG/EEG_Filtered_Data/' #dataset in local PC

channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                       'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                       'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                       'EEG T1-REF', 'EEG T2-REF']
split_mapping={'FP1': 'Fp1', 'FP2':'Fp2', 'F3':'F3', 'F4':'F4', 'C3':'C3', 'C4':'C4', 'P3':'P3', 'P4':'P4', 'O1':'O1', 'O2':'O2', 'F7':'F7', 'T3':'T3', 'T4':'T4', 'T5':'T5', 'T6':'T6',
'A1':'A1', 'A2':'A2', 'FZ':'Fz', 'CZ':'Cz', 'PZ':'Pz', 'E1':'E1', 'E2':'E2'}
new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'E1', 'E2']
    # List all files in the directory
all_files = os.listdir(directory_path)


#mne.io.read_raw_eeglab() 
#mne.read_epochs_eeglab()

all_sessions = []
artifact_session=[]
all_sessions_names=[]
for file in sorted(all_files):
    file_path = os.path.join(directory_path, file)
    all_sessions_names.append(file_name)
    sub_id, session, time = file_name.split(".")[0].split(
        "_")  # split the filname into subject, session and time frame

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

    raw_mne = mne.io.read_raw_edf(file_path,
                                    preload=True)  # Load the EDF file: NB raw_mne.info['chs'] is the only full of information
    raw_mne.pick_channels(channels_to_set,
                            ordered=True)  # reorders the channels and drop the ones not contained in channels_to_set
    rename_mapping = dict(zip(channels_to_set, new_channel_names))
    raw_mne.rename_channels(rename_mapping)

    raw_mne.filter(l_freq=0.5, h_freq=50)
    raw_mne.notch_filter(freqs=60, picks='all', method='spectrum_fit')
    raw_mne.resample(250)  # resample to standardize sampling frequency to 250 Hz
 
    epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=False, reject_by_annotation=False)  # divide the signal into fixed lenght epoch of 4s with 1 second of overlapping: the overlapping starts from the left side of previous epoch
    del raw_mne
    epoch_data = epochs_mne.get_data(copy=False)  # trasform the raw eeg into a 3d np array
    del epochs_mne

    # Extract onset times of annotations and convert to sample indices
    sfreq = 250

    # Initialize artifact flag array with zeros
    n_epochs, n_channels, n_samples_per_epoch = epoch_data.shape
    epoch_data=epoch_data*1e6

    mean=np.mean(epoch_data)
    std = np.std(epoch_data)
    epoch_data = (epoch_data-mean) / std  # normalization for session
    del mean
    del std
    #epoch_data=normalize_to_range(epoch_data)
    epoch_data = np.expand_dims(epoch_data, 1)  # number of epochs for that signal x 1 x channels x time samples
# initialize a list containing all sessions
    all_sessions.append(epoch_data)
    artifact_session.append(artifact_flags)
