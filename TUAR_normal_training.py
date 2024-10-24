from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
    
#directory_path='/home/azorzetto/dataset/01_tcp_ar' #dataset in local PC
#directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar' #dataset in workstation
#directory_path="/home/azorzetto/data1/Dataset_controllato"
directory_path="/home/azorzetto/data1/datset_no_SEIZ"
channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                       'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF','EEG T3-REF', 'EEG T4-REF',
                       'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                       'EEG T1-REF', 'EEG T2-REF']
split_mapping={'FP1': 'Fp1', 'FP2':'Fp2', 'F3':'F3', 'F4':'F4', 'C3':'C3', 'C4':'C4', 'P3':'P3', 'P4':'P4', 'O1':'O1', 'O2':'O2', 'F7':'F7', 'T3':'T3', 'T4':'T4', 'T5':'T5', 'T6':'T6',
'A1':'A1', 'A2':'A2', 'FZ':'Fz', 'CZ':'Cz', 'PZ':'Pz', 'E1':'E1', 'E2':'E2'}
new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'E1', 'E2']
    # List all files in the directory
all_files = os.listdir(directory_path)
    # Filter out only EDF files
edf_files = [file for file in all_files if file.endswith('.edf')]

start_index=0
end_index=10
# Process each EDF file
if start_index == None:
    start_index=0

if end_index == None:
    end_index=len(edf_files) -1

#subj_list=[]
all_sessions = []
artifact_session=[]
all_sessions_names=[]
for file_name in sorted(edf_files)[start_index:end_index]:
    file_path = os.path.join(directory_path, file_name)
    all_sessions_names.append(file_name)
    sub_id, session, time = file_name.split(".")[0].split(
        "_")  # split the filname into subject, session and time frame
    """if sub_id in subj_list:
        continue
    else:"""
        #subj_list.append(sub_id)

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
    

    raw_mne.filter(l_freq=0.5, h_freq=100)
    raw_mne.notch_filter(freqs=60, picks='all', method='spectrum_fit')
    raw_mne.resample(250)  # resample to standardize sampling frequency to 250 Hz

    #raw_mne.set_eeg_reference(ref_channels=['EEG A1-REF'])

    raw_mne.pick_channels(channels_to_set,
                            ordered=True)  # reorders the channels and drop the ones not contained in channels_to_set

    raw_mne = mne.io.read_raw_edf(file_path,
                                    preload=True)  # Load the EDF file: NB raw_mne.info['chs'] is the only full of information
    raw_mne.pick_channels(channels_to_set,
                            ordered=True)  # reorders the channels and drop the ones not contained in channels_to_set
    rename_mapping = dict(zip(channels_to_set, new_channel_names))
    raw_mne.rename_channels(rename_mapping)


    raw_mne.filter(l_freq=0.5, h_freq=50, verbose=True)
    raw_mne.notch_filter(freqs=60, picks='all', method='spectrum_fit')
    raw_mne.resample(250)  # resample to standardize sampling frequency to 250 Hz
 
    epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=False, reject_by_annotation=False)  # divide the signal into fixed lenght epoch of 4s with 1 second of overlapping: the overlapping starts from the left side of previous epoch
    del raw_mne
    epoch_data = epochs_mne.get_data(copy=False)  # trasform the raw eeg into a 3d np array
    del epochs_mne
    epoch_data=epoch_data*1e6

    # Extract onset times of annotations and convert to sample indices
    sfreq = 250

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

    epoch_data=epoch_data*1e6

    """mean=np.mean(epoch_data)
    std = np.std(epoch_data)
    epoch_data = (epoch_data-mean) / std  # normalization for session
    del mean
    del std""" 
    #epoch_data=normalize_to_range(epoch_data)
    epoch_data = np.expand_dims(epoch_data, 1)  # number of epochs for that signal x 1 x channels x time samples
    artifact_flags= np.expand_dims(artifact_flags, 1) 

# initialize a list containing all sessions
    all_sessions.append(epoch_data)
    artifact_session.append(artifact_flags)

global_min=(np.concatenate(all_sessions)).min()
global_max=(np.concatenate(all_sessions)).max()

new_all_sessions=[]

for session in all_sessions:
    session_normalized=normalize_to_range(x=session,min_val=global_min, max_val=global_max, alpha=-50, beta=50)
    new_all_sessions.append(session_normalized)

all_sessions=new_all_sessions

dataset_artifact=np.concatenate(artifact_session)
perc_artifacts=dataset_artifact.sum()/dataset_artifact.size *100
perc_clean= (dataset_artifact.size - dataset_artifact.sum())/dataset_artifact.size *100
print(f"percentage artifactual dataset in the whole dataset: {perc_artifacts}")
print(f"percentage clean dataset in the whole dataset: {perc_clean}")

test_size = int(np.ceil(0.2 * len(all_sessions)))
test_data = np.concatenate(all_sessions[0:test_size])
test_data_artifact=np.concatenate(artifact_session[0:test_size])
perc_artifacts=test_data_artifact.sum()/test_data_artifact.size *100
perc_clean= (test_data_artifact.size - test_data_artifact.sum())/test_data_artifact.size *100
print(f"percentage artifactual dataset in the test dataset: {perc_artifacts}")
print(f"percentage clean dataset in the test dataset: {perc_clean}")


validation_data = np.concatenate(all_sessions[test_size:2*test_size])
validation_data_artifact=np.concatenate(artifact_session[test_size:2*test_size])
perc_artifacts=validation_data_artifact.sum()/validation_data_artifact.size *100
perc_clean= (validation_data_artifact.size - validation_data_artifact.sum())/validation_data_artifact.size *100
print(f"percentage artifactual dataset in the validation dataset: {perc_artifacts}")
print(f"percentage clean dataset in the validation dataset: {perc_clean}")

train_data=np.concatenate(all_sessions[2*test_size:])
train_data_artifact=np.concatenate(artifact_session[2*test_size:])
perc_artifacts=train_data_artifact.sum()/train_data_artifact.size *100
perc_clean= (train_data_artifact.size - train_data_artifact.sum())/train_data_artifact.size *100
print(f"percentage artifactual dataset in the train dataset: {perc_artifacts}")
print(f"percentage clean dataset in the train dataset: {perc_clean}")

train_label: np.ndarray = np.random.randint(0, 4, train_data.shape[0])
validation_label: np.ndarray = np.random.randint(0, 4, validation_data.shape[0])

#save as npz for reproducibility
#np.savez_compressed('/home/azorzetto/train5/dataset.npz', test_data=test_data, validation_data=validation_data, train_data=train_data, train_label=train_label, validation_label=validation_label)
np.savez_compressed('dataset.npz', edf_files= sorted(edf_files)[start_index:end_index], test_data=test_data,test_data_artifact=test_data_artifact, validation_data=validation_data, validation_data_artifact= validation_data_artifact, train_data=train_data,train_data_artifact= train_data_artifact, train_label=train_label, validation_label=validation_label)

train_dataset = ds_time.EEG_Dataset(train_data, train_label, channels_to_set)
validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, channels_to_set)

print("EEG_Dataset function called")

# Get number of channels and length of time samples
C = train_data.shape[2]
T = train_data.shape[3]
del train_data
del validation_data

train_config = ct.get_config_hierarchical_vEEGNet_training()
epochs = 160
# path_to_save_model = 'model_weights_backup'
path_to_save_model = 'model_weights_backup16' # the folder is model wights backup_iterationOfTheTuple and inside we have one file for each epoch
os.makedirs(path_to_save_model, exist_ok=True)
epoch_to_save_model = 1

# Update train config
train_config['epochs'] = epochs
train_config['path_to_save_model'] = path_to_save_model
train_config['epoch_to_save_model'] = epoch_to_save_model
train_config['log_dir'] = './logs16'
os.makedirs(train_config['log_dir'], exist_ok=True)
train_config['early_stopping'] = False #if you want to activate the early stopping

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get model


# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']
# hvEEGNet creation
model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
# Declare loss function
# This method return the PyTorch loss function required by the training function.
# The loss function for hvEEGNet is not directy implemented in PyTorch since it is a combination of different losses. So I have to create my own function to combine all the components.

loss_function = train_generic.get_loss_function(model_name='hvEEGNet_shallow', config=train_config)
#loss_function= CustomMSELoss()
# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(),
                                lr=train_config['lr'],
                                weight_decay=train_config['optimizer_weight_decay']
                                )
# (OPTIONAL) Setup lr scheduler
if train_config['use_scheduler']:
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_config['lr_decay_rate'])
else:
    lr_scheduler = None

# Move the model to training device (CPU/GPU)
model.to(train_config['device'])
print("-----------------------------------------to call the dataloader------------------------------------------")
# Create dataloader
#does not work with a batch size greater than 16
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True,
                                num_workers=6, drop_last=True)  
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=8, shuffle=True,
                                    num_workers=6, drop_last=True)
loader_list = [train_dataloader, validation_dataloader]
print("-----------------------------------------loader list created------------------------------------------")
train_generic.train(model=model, loss_function=loss_function, optimizer=optimizer,
                                                                        loader_list=loader_list, train_config=train_config, lr_scheduler=lr_scheduler,
                                                                        model_artifact=None)


"""results = [] #list containing the dictionaries
av_reconstruction_error=[]
i=0
to_save_eeg=[]
for i in range(test_data.shape[0]):
#for i in range(10):
    x_eeg_=test_data[i]
    x_eeg = x_eeg_.astype(np.float32)
    x_eeg = torch.from_numpy(x_eeg)
    x_eeg = x_eeg.unsqueeze(1)  
    x_eeg = x_eeg.to(train_config['device'])
    model.to(train_config['device'])
    x_r_eeg = model.reconstruct(x_eeg)
   
    recon_error_avChannelsF_avTSF, recon_error_avChannelsF_avTST, recon_error_avChannelsT_avTSF, recon_error_avChannelsT_avTST = reconstruction_metrics(
        x_eeg, x_r_eeg, train_config['device'])
    new_row = {
        'Reconstruction error with no average_channels and no average_time_samples': recon_error_avChannelsF_avTSF,
        'Reconstruction error with no average_channels and average_time_samples': recon_error_avChannelsF_avTST,
        'Reconstruction error with average_channels  and no average_time_samples': recon_error_avChannelsT_avTSF,
        'Reconstruction error with average_channels  and average_time_samples': recon_error_avChannelsT_avTST}
    results.append(new_row)
    av_reconstruction_error.append(recon_error_avChannelsT_avTST.cpu().numpy())
    to_save_eeg.append(x_r_eeg.cpu().numpy())

with open('resconstruction_error.pkl', 'wb') as file:
    pickle.dump(results, file)

to_save_eeg=np.concatenate(to_save_eeg)
np.savez('reconstructed_eeg.npz', x_r_eeg=to_save_eeg)

df_reconstuction_error = pd.DataFrame(av_reconstruction_error)

df_reconstuction_error.to_csv('mean_reconstruction_errors.csv', index=False)"""