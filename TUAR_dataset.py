import mne
from mne.io import Raw
import torch
from numpy.typing import NDArray
from typing import Dict, List, Tuple
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm
import os
import mne
import numpy as np
import random
import pandas as pd
from collections import defaultdict
np.random.seed(43) 
# Directory containing the EDF files
#directory_path='C:\Users\albin\OneDrive\Desktop\TCParRidotto'
#directory_path = r'C:\Users\albin\OneDrive\Desktop\TCParRidotto'

#directory_path = '/home/azorzetto/data1/TCParRidotto'
directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar'
channels_to_set=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
# List all files in the directory
all_files = os.listdir(directory_path)
# Filter out only EDF files
edf_files = [file for file in all_files if file.endswith('.edf')]

# data structure Dict[str, Dict[str, NDArray] --> Dict[subj_id, Dict[sess, NDArray]]
session_data: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.array([])))
#data=[] #I create a list where to store the arrays. Each array corresponds to an EDF file channel x time_sampel
# Process each EDF file
for file_name in sorted(edf_files):
    file_path = os.path.join(directory_path, file_name)
    sub_id, session, time = file_name.split(".")[0].split("_") #split the filname into subject, session and time frame
    raw_mne = mne.io.read_raw_edf(file_path, preload=True)  # Load the EDF file
    raw_mne.pick_channels(channels_to_set) #reorders the channels and drop the ones not contained in channels_to_set
    raw_mne.resample(250) #resample to standardize sampling frequency to 250 Hz
    epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=True, overlap=3) #divide the signal into fixed lenght epoch of 4s with 1 second of overlapping: the overlapping starts from the left side of previous epoch
    epoch_data = epochs_mne.get_data() #trasform the raw eeg into a 3d np array
    epoch_data = np.expand_dims(epoch_data, 1) # number of epochs for that signal x 1 x channels x time samples 

    # If session_data[sub_id][session] exists, concatenate
    if session_data[sub_id][session].size > 0:
        session_data[sub_id][session] = np.concatenate([session_data[sub_id][session], epoch_data], axis=0)
    else:
        session_data[sub_id][session] = epoch_data
    


#the normalization z2 is performed on each subject or session regardless the training/test set split. 
# the normalization z3 is performed based on the entrie traing set 

#leave one session out from the training for testing
#we are loosing the subject information level
def leave_one_session_out(session_data: Dict[str, Dict[str, np.ndarray]], normalization: str = None,shuffle: bool = True): #-> np.ndarray, np.ndarray, np.ndarray
    #a list of the dictionaries [{session: arrays}, {session: arrays}, {session: arrays},...]  
    if normalization == 'z2':
        for subject, session_dict in session_data.items(): 
            # Collect all session arrays for this subject
            sessions_list = list(session_dict.values())
            # Concatenate all arrays to compute the mean and std
            total_for_subject = np.concatenate(sessions_list)
            mean = np.mean(total_for_subject)
            std = np.std(total_for_subject)
            # Normalize each session
            for key in session_dict:
                session_dict[key] = (session_dict[key] - mean) / std
    #list of defautdict [{'session': array}]
    list_dict_session=session_data.values() #the type is dict_values
    all_sessions=[] #a list containing all sessions
    for el in list_dict_session:

        all_sessions.extend(list(el.values()))
    test_size = int(np.ceil(0.2 * len(all_sessions)))
    test_data=all_sessions[0:test_size]
    train_val_data=all_sessions[test_size:]
    if normalization=='z3':
        total=np.concatenate(train_val_data)
        mean= np.mean(total)
        std=np.std(total)
        for i in range(len(train_val_data)):
            train_val_data[i] = (train_val_data[i] - mean) / std
        mean_test=np.mean(np.concatenate(test_data))
        std_test=np.std(np.concatenate(test_data))
        for i in range(len(test_data)):
            test_data[i] = (test_data[i] - mean_test) / std_test
    train_label = np.random.randint(0, 4, len(train_val_data)-1) #because in the traingin dataset is the train_val_datset minus 1 session
    validation_label = np.random.randint(0, 4, 1) #only one session in the val set
    #list of tuples containing the train data as the fist element and the validation data as the second element 
    combinations=[]
    for i in range (len(train_val_data)):
        if shuffle==True:
            random.shuffle(train_val_data[i])
        train_data = train_val_data[:i] + train_val_data[i+1:]
        val_data=train_val_data[i]
        combinations.append((test_data, train_data, val_data)) #(test_data: list, train_data: list, val_data: ndarray)
    
    return combinations, train_label, validation_label

def leave_one_subject_out(session_data, normalization:str = None, shuffle: bool=True):
    subject_data_dict={}
    for key, value in session_data.items():#key is the subj, value is a dictionary 
        # fixed key we are dealing with a subject 
        new_value=np.concatenate(list(value.values())) #new value is the concatenation of a list of arrays representing the sessions for that subject
        if normalization =='z2':
            mean=np.mean(new_value)
            std=np.std(new_value)
            new_value=(new_value-mean)/std
        subject_data_dict.update({key: new_value})
    
    all_data=list(subject_data_dict.values())
    test_size = int(np.ceil(0.2 * len(all_data)))
    test_data=all_data[0:test_size]
    train_val_data=all_data[test_size:]
    if normalization=='z3':
        total=np.concatenate(train_val_data)
        mean= np.mean(total)
        std=np.std(total)
        for i in range(len(train_val_data)):
            train_val_data[i]=train_val_data[i]-mean
            train_val_data[i]=train_val_data[i]/std

        mean_test=np.mean(np.concatenate(test_data))
        std_test=np.std(np.concatenate(test_data))

        for i in range(len(test_data)):
            test_data[i] = (test_data[i] - mean_test) / std_test

    train_label = np.random.randint(0, 4, len(train_val_data)-1)
    validation_label = np.random.randint(0, 4, 1)
    combinations=[]
    for i in range (len(train_val_data)):
        if shuffle==True:
            random.shuffle(train_val_data[i])
        train_data = train_val_data[:i] + train_val_data[i+1:]
        val_data=train_val_data[i]
        combinations.append((test_data, train_data, val_data)) #list, list, ndarray
    
    return combinations, train_label, validation_label

combinations1, train_label1, validation_label1= leave_one_session_out(session_data, normalization='z3')
combinations2, train_label2, validation_label2= leave_one_subject_out(session_data, normalization='z3')

train_dataset = ds_time.EEG_Dataset(train_data, train_label, channels_to_set)
validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, channels_to_set)

# Get training config
train_config = ct.get_config_hierarchical_vEEGNet_training()

epochs = 2
path_to_save_model = 'model_weights_backup'
epoch_to_save_model = 1

# Update train config
train_config['epochs'] = epochs
train_config['path_to_save_model'] = path_to_save_model
train_config['epoch_to_save_model'] = epoch_to_save_model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get model

# Get number of channels and length of time samples
C = train_data.shape[2]
T = train_data.shape[3]

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']

# hvEEGNet creation
model = hvEEGNet.hvEEGNet_shallow(model_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataloader, loss function, optimizer and lr_scheduler

# Create dataloader
train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
loader_list             = [train_dataloader, validation_dataloader]

# Declare loss function
# This method return the PyTorch loss function required by the training function.
# The loss function for hvEEGNet is not directy implemented in PyTorch since it is a combination of different losses. So I have to create my own function to combine all the components.
loss_function = train_generic.get_loss_function(model_name = 'hvEEGNet_shallow', config = train_config)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(),
                              lr = train_config['lr'],
                              weight_decay = train_config['optimizer_weight_decay']
                              )

# (OPTIONAL) Setup lr scheduler
if train_config['use_scheduler'] :
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
else:
    lr_scheduler = None
    
# Move the model to training device (CPU/GPU)
model.to(train_config['device'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

model = train_generic.train(model, loss_function, optimizer,
                            loader_list, train_config, lr_scheduler, model_artifact = None)

#-------------------------------------------------
#RECONSTRUCTION
ch_to_plot='EEG FP2-REF'
x_eeg=data[subject_test]
x_r_eeg = model.reconstruct(x_eeg)

"""
daf=[]
for file in edf_files:
    daf.append({'name': file.split('_')[0], 'trial': file.split('_')[1], 'session': file.split('_')[2]})
df = pd.DataFrame(daf)
print(df)

trials_counts = df.groupby('name')['trial'].nunique().reset_index()
trials_counts.columns = ['name', 'unique_trial_count']
frequency_counts = trials_counts['unique_trial_count'].value_counts().reset_index()
# Rename the columns for clarity
frequency_counts.columns = ['unique_trial_count', 'frequency']
print(frequency_counts)

session_counts = df.groupby(['name', 'trial'])['session'].nunique().reset_index()
session_counts.columns = ['name', 'trial', 'unique_session_count']
frequency_counts = session_counts['unique_session_count'].value_counts().reset_index()
# Rename the columns for clarity
frequency_counts.columns = ['unique_session_count', 'frequency']
print(frequency_counts)"""
