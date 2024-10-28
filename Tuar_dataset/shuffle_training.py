
import torch
import json
import matplotlib
matplotlib.use('TkAgg')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm
from library.config import config_dataset as cd
import mne
import numpy as np
import pandas as pd
from utils import statistics_clean_eeg, normalize_to_range
from torch.utils.data import DataLoader

np.random.seed(43)
train_session=17
directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar'
path_config_json=f'/home/azorzetto/trainShuffle{train_session}/config.json'
path_TUAR_config_json=f'/home/azorzetto/trainShuffle{train_session}/TUAR_config.json'
#directory_path="/content"
#directory_path='/home/azorzetto/dataset/01_tcp_ar'
#directory_path = '/home/lmonni/Documents/01_tcp_ar'
#directory_path="/home/azorzetto/data1/01_tcp_ar_jrj"
#directory_path="/home/azorzetto/data1/Dataset_controllato"
#directory_path="/home/azorzetto/dataset/Dataset_controllato/"

with open(path_TUAR_config_json, 'r') as config_file:
    TUAR_json_config = json.load(config_file)

TUAR_config=cd.get_preprocessing_config()
TUAR_config.update(TUAR_json_config)

channels_to_set=TUAR_config['channels_to_set']
split_mapping=TUAR_config['split_mapping']
new_channel_names=TUAR_config['new_channel_names']
sfreq = TUAR_config['sfreq']

# List all files in the directory
all_files = os.listdir(directory_path)
# Filter out only EDF files
edf_files = [file for file in all_files if file.endswith('.edf')]
start_index=TUAR_config['start_index']
end_index=TUAR_config['end_index']

if end_index == None:
    end_index=len(edf_files) -1

all_sessions=[]
artifact_session=[]
# Loop through each EDF file
for file_name in sorted(edf_files)[start_index:end_index]:
    
    file_path_edf = os.path.join(directory_path, file_name)

    # Split the filename into subject, session, and time frame
    sub_id, session, time = file_name.split(".")[0].split("_")
    #corresponding csv file
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
            df_artifact.drop(index=i,inplace=True) #then it's dropped
            continue
        # Append the mapped channel name inside a list (to create a list of lists)
        ch_names.append([mapped_channel_name])
        df_artifact['channel'][i]=mapped_channel_name

    df_artifact['duration_artifact'] = df_artifact.iloc[:, 2] - df_artifact.iloc[:, 1]
    onset = (df_artifact.iloc[:, 1]).astype(float)
    duration=(df_artifact['duration_artifact']).astype(float)   
    description=df_artifact['label']

    # Load the EDF file
    raw_mne = mne.io.read_raw_edf(file_path_edf, preload=True)  # Set preload to True to load data into memory
    if TUAR_config['band_pass_filter']==True:
        raw_mne.filter(l_freq=TUAR_config['l_freq'], h_freq=TUAR_config['h_freq'])
    if TUAR_config['notch_filter']==True:
        raw_mne.notch_filter(freqs=60, picks='all', method='spectrum_fit')
    if TUAR_config['monopolar_reference']==True:
        raw_mne.set_eeg_reference(ref_channels=TUAR_config['ref_channel'])

    raw_mne.pick_channels(channels_to_set,
                            ordered=True)  # reorders the channels and drop the ones not contained in channels_to_set
    
    rename_mapping = dict(zip(channels_to_set, new_channel_names))
    raw_mne.rename_channels(rename_mapping)

    raw_mne.resample(sfreq)  # resample to standardize sampling frequency to 250 Hz
 
    epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=False, reject_by_annotation=False)  # divide the signal into fixed lenght epoch of 4s with 1 second of overlapping: the overlapping starts from the left side of previous epoch
    del raw_mne
    epoch_data = epochs_mne.get_data(copy=False)  # trasform the raw eeg into a 3d np array
    del epochs_mne
    epoch_data=epoch_data*1e6

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

    if TUAR_config['z_score_session_wise']:
        mean=np.mean(epoch_data)
        std = np.std(epoch_data)
        epoch_data = (epoch_data-mean) / std  # normalization for session
        del mean
        del std
    if TUAR_config['z_score_cleand']:
        mean, std = statistics_clean_eeg(epoch_data, artifact_flags)
        epoch_data = (epoch_data-mean) / std 
        del mean
        del std

    all_sessions.append(epoch_data)
    artifact_session.append(artifact_flags)

dataset=np.concatenate(all_sessions)
dataset_artifact=np.concatenate(artifact_session)
shuffle_indices = np.random.permutation(dataset.shape[0])
dataset=dataset[shuffle_indices]
dataset_artifact=dataset_artifact[shuffle_indices]

dataset = np.expand_dims(dataset, 1)
dataset_artifact = np.expand_dims(dataset_artifact, 1)

#per tenere solo i trials senza artefatti-----------------------
if TUAR_config['only_cleand_trials']:
    valid_trials = []

    # Itera sui trial
    for trial in range(dataset.shape[0]):  # trial va da 0 al numero di trial
        # Somma degli elementi nel trial di dataset_artifact
        if dataset_artifact[trial].sum() < 1:
            valid_trials.append(trial)

    # Filtra dataset e dataset_artifact mantenendo solo i trial validi
    dataset = dataset[valid_trials]
    dataset_artifact = dataset_artifact[valid_trials]
#¯-------------------------------------------

perc_artifacts=dataset_artifact.sum()/dataset_artifact.size *100
perc_clean= (dataset_artifact.size - dataset_artifact.sum())/dataset_artifact.size *100
print(f"percentage artifactual dataset in the whole dataset: {perc_artifacts}")
print(f"percentage clean dataset in the whole dataset: {perc_clean}")

test_size = int(np.ceil(0.2 * dataset.shape[0]))
test_data = dataset[0:test_size]
test_data_artifact=dataset_artifact[0:test_size]
perc_artifacts=test_data_artifact.sum()/test_data_artifact.size *100
perc_clean= (test_data_artifact.size - test_data_artifact.sum())/test_data_artifact.size *100
print(f"percentage artifactual dataset in the test dataset: {perc_artifacts}")
print(f"percentage clean dataset in the test dataset: {perc_clean}")

validation_data = dataset[test_size:2*test_size]
validation_data_artifact=dataset_artifact[test_size:2*test_size]
perc_artifacts=validation_data_artifact.sum()/validation_data_artifact.size *100
perc_clean= (validation_data_artifact.size - validation_data_artifact.sum())/validation_data_artifact.size *100
print(f"percentage artifactual dataset in the validation dataset: {perc_artifacts}")
print(f"percentage clean dataset in the validation dataset: {perc_clean}")

train_data=dataset[2*test_size:]
train_data_artifact=dataset_artifact[2*test_size:]
perc_artifacts=train_data_artifact.sum()/train_data_artifact.size *100
perc_clean= (train_data_artifact.size - train_data_artifact.sum())/train_data_artifact.size *100
print(f"percentage artifactual dataset in the train dataset: {perc_artifacts}")
print(f"percentage clean dataset in the train dataset: {perc_clean}")

train_label: np.ndarray = np.random.randint(0, 4, train_data.shape[0])
validation_label: np.ndarray = np.random.randint(0, 4, validation_data.shape[0])
#save as npz for reproducibility
np.savez_compressed('dataset1.npz', edf_files= sorted(edf_files)[start_index:end_index], test_data=test_data,test_data_artifact=test_data_artifact, validation_data=validation_data, validation_data_artifact= validation_data_artifact, 
                    train_data=train_data,train_data_artifact= train_data_artifact, train_label=train_label, validation_label=validation_label)

train_dataset = ds_time.EEG_Dataset(train_data, train_label, channels_to_set)
validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, channels_to_set)

print("EEG_Dataset function called")

# Get number of channels and length of time samples
C = train_data.shape[2]
T = train_data.shape[3]
del train_data
del validation_data
with open(path_config_json, 'r') as config_file:
    json_config = json.load(config_file)
train_config = ct.get_config_hierarchical_vEEGNet_training()
train_config.update(json_config)

os.makedirs(train_config['path_to_save_model'], exist_ok=True)
os.makedirs(train_config['log_dir'], exist_ok=True)


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
print('-----------------------------STOP-------------------------------------------')
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