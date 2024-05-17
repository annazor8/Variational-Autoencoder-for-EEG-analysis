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
directory_path = '/home/azorzetto/data1/TCParRidotto'
channels_to_set=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG EKG1-REF', 'EEG T1-REF', 'EEG T2-REF']
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
    raw_mne = mne.io.read_raw_edf(file_path)  # Load the EDF file
    raw_mne = raw_mne.reorder_channels(channels_to_set) #reorders the channels and drop the ones not contained in channels_to_set
    raw_mne=raw_mne.resample(250) #resample to standardize sampling frequency to 250 Hz
    epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=True, overlap=3) #divide the signal into fixed lenght epoch of 4s with 1 second of overlapping: the overlapping starts from the left side of previous epoch
    epoch_data = epochs_mne.get_data() #trasform the raw eeg into a 3d np array
    epoch_data = np.expand_dims(epoch_data, 1) # number of epochs for that signal x 1 x channels x time samples 

    # If session_data[sub_id][session] exists, concatenate
    if session_data[sub_id][session].size > 0:
        session_data[sub_id][session] = np.concatenate([session_data[sub_id][session], epoch_data], axis=-1)
    else:
        session_data[sub_id][session] = epoch_data
    
    #data.append(epoch_data)


#the input data is a 4D matrix representing either the training set or the test set of one or multiple subject(s)
def normalization_z3(data: NDArray):

    mean= np.mean(data)
    std=np.std(data)
    data=(data - mean)/std
    return data

#the normalization z2 is performed on each subject or session regardless the training/test set split. 
# In our case it's the same because the trinaing set is composed by only 1 subject for now
#in this case data should be a list of 4d arrays: each array is corresponding to a subject
def normalization_z2(data):
    for i, el in enumerate(data):
        norm_el=normalization_z3(el)
        if i==0:
            final=norm_el
        else:
            final=np.concatenate((final, norm_el), axis=0)
    return final

session_data_normalized = defaultdict(lambda: defaultdict(list))
for subj in session_data:
    for session, listdata in session_data[subj]:
        session_data_normalized[subj][session] = normalization_z3(data=listdata[0])

#leave one session out from the training for testing BUT we have to set the subject 
def leave_one_session_out(session_data: Dict[str, Dict[str, np.ndarray]], shuffle: bool = True): #-> np.ndarray, np.ndarray, np.ndarray
   list_dict_session=session_data.value()
   all_sessions=[]
   all_sessions = [el['key'] for el in list_dict_session]
    
    if number_of_sessions == 1:
        data_session=dict_sessions.values()[0]
        test_size = int(0.2 * data_session.shape[0])
        test_data=data_session[0:test_size,:,:,:]
        train_val_session=data_session[test_size:,:,:,:]
        if shuffle==True:
            train_val_session=random.shuffle(train_val_session)
        train_size=int(0.8 * train_val_session.shape[0])
        train_data=train_val_session[0:train_size,:,:,:]
        val_data=train_val_session[train_size,:,:,:]
    #if for that subject i have more than one session i can use one for test and the other(s) for training
    else: 
        test_session_index=random.randint(0, number_of_sessions)
        test_data=dict_sessions.pop(test_session_index) #pop removes the entry at that index from the dictionary and returns its value 
        train_val_session=np.concatenate(dict_sessions) #concatenate all the remaning arrays for training and validation 
        train_size = int(0.8 * train_val_session.shape[0])
        if shuffle==True:
            train_val_session=random.shuffle(train_val_session)
        train_data=train_val_session[0:train_size,:,:,:]
        val_data=train_val_session[train_size:,:,:,:]
    
    train_label = np.random.randint(0, 4, train_data.shape[0])
    validation_label = np.random.randint(0, 4, val_data.shape[0])
    return test_data, train_data, val_data, train_label, validation_label

def leave_one_subject_out(session_data, subject: str = None, shuffle: bool=True):
    list_subj=session_data.keys() #list of subjects name
    test_subject_index=np.random.randint(0, len(list_subj)) #random subject to keep as a test subject
    test_entry=session_data.pop(list_subj[test_subject_index]) #remove and return the {Dict[str, np.ndarray]}
    test_data=np.concatenate(test_entry.value()) #concatenates the array of the same subject but different sessions 
    train_val_data=np.concatenate(session_data.value().value())
    if shuffle==True:
        train_val_data=random.shuffle(train_val_data)
    train_size=int(0.8 * train_val_data.shape[0])
    train_data=train_val_data[0:train_size,:,:,:]
    val_data=train_val_data[train_size:,:,:,:]
    train_label = np.random.randint(0, 4, train_data.shape[0])
    validation_label = np.random.randint(0, 4, val_data.shape[0])
    return test_data, train_data,val_data, train_label, validation_label, list_subj[test_subject_index]

train_data=normalization_z2(dataset)

train_size = int(0.8 * dataset.shape[0])  # 80% for training
val_size = dataset.shape[0] - train_size  # 20% for validation

#split the dataset into training and validation, respectively 80% and 20%
train_data=dataset[0:train_size,:,:,:] #(1, 1, 22, 1000)
validation_data=dataset[train_size:,:,:,:] #(1, 1, 22, 1000)

#create random labels since they are not useful for the training part 
train_label = np.random.randint(0, 4, train_data.shape[0])
validation_label = np.random.randint(0, 4, validation_data.shape[0])

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
