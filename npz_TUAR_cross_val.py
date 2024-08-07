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
import gc

np.random.seed(43)
    
#directory_path='/home/azorzetto/dataset/01_tcp_ar' #dataset in local PC
directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar' #dataset in workstation

#for storing the reconstruction values
df_reconstruction = pd.DataFrame([], columns=[
    'Reconstruction error with no average_channels and no average_time_samples',
    'Reconstruction error with no average_channels and average_time_samples',
    'Reconstruction error with average_channels  and no average_time_samples',
    'Reconstruction error with average_channels  and average_time_samples'])

channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                    'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                    'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                    'EEG T1-REF', 'EEG T2-REF']
# List all files in the directory
all_files = os.listdir(directory_path)
# Filter out only EDF files
edf_files = [file for file in all_files if file.endswith('.edf')] #290 in totale

# data structure Dict[str, Dict[str, NDArray] --> Dict[subj_id, Dict[sess, NDArray]]
session_data: Dict[str, Dict[str, str]] = defaultdict(lambda: defaultdict(str))
#session_data: Dict[str, Dict[str, str]] = defaultdict(lambda: defaultdict(lambda: str))
all_session=[]
# Process each EDF file


for file_name in sorted(edf_files):
    file_path = os.path.join(directory_path, file_name)
    sub_id, session, time = file_name.split(".")[0].split(
        "_")  # split the filname into subject, session and time frame
    raw_mne = mne.io.read_raw_edf(file_path,
                                    preload=False)  # Load the EDF file: NB raw_mne.info['chs'] is the only full of information
    raw_mne.pick_channels(channels_to_set,
                            ordered=True)  # reorders the channels and drop the ones not contained in channels_to_set
    raw_mne.resample(250)  # resample to standardize sampling frequency to 250 Hz
    epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=False)  # divide the signal into fixed lenght epoch non overlapping
    del raw_mne
    epoch_data = epochs_mne.get_data(copy=False)  # trasform the raw eeg into a 3d np array
    del epochs_mne
    mean=np.mean(epoch_data)
    std = np.std(epoch_data)
    epoch_data = (epoch_data-mean) / std  # normalization for session
    del mean
    del std
    epoch_data = np.expand_dims(epoch_data, 1)  # number of epochs for that signal x 1 x channels x time samples

    folder_name = 'npzs'
    os.makedirs(folder_name, exist_ok=True)

    if session_data[sub_id][session]!= "":
        new_session = session + '_01'
        filename= sub_id + new_session + '.npz'
        file_path = os.path.join(folder_name, filename)
        np.savez(file_path, epoch_data)
        session_data[sub_id][new_session] = file_path
    else:
        filename=sub_id + session + '.npz'
        file_path = os.path.join(folder_name, filename)
        np.savez(file_path, epoch_data)
        session_data[sub_id][session] = file_path

    del epoch_data

print("------------------------------session data created---------------------------------------")
#leave one session out 
list_dict_session = session_data.values()  # the type is dict_values, is a list of paths
all_sessions= []  # initialize a list containing all sessions
for el in list_dict_session:
    all_sessions.extend(list(el.values()))

test_size = int(np.ceil(0.2 * len(all_sessions)))
test_data = all_sessions[0:test_size]
train_val_data = all_sessions[test_size:]

#list of tuples containing the train data as the fist element and the validation data as the second element
combinations = []
for i in range(0, len(train_val_data), 37):
    # do not make shuffle(train_data[i]) because the temporal sequence of the layers in the 3d matrix is important to be preserved
    train_data = train_val_data[:i] + train_val_data[i + 37:]  # concatenate the two lists with the + operator
    val_data = train_val_data[i:i + 37]
    combinations.append(
        (train_data, val_data))

del train_data

batch_size = 10
print("-------------------------loo done------------------------------------------")
for indx, combo in enumerate(combinations):  # 220 is the max number of combinations, with a slide of 37 there are 6 combinations
    train_data_list_path: list = combo[0]
    validation_data_list_path: list = combo[1]
    train_data_list=[]
    validation_data_list=[]

    train_data =None
    for i in range(0, len(train_data_list_path), batch_size):
        batch_paths = train_data_list_path[i:i + batch_size]
        batch_data = [np.load(filepath, mmap_mode='r')['arr_0'] for filepath in batch_paths]
    
        if train_data is None:
            train_data = np.concatenate(batch_data)
        else:
            train_data = np.concatenate((train_data, np.concatenate(batch_data)))
        
    #train_data=train_data[0:(train_data.shape(0)/2),:,:,:]
    del batch_data
    del train_data_list_path
    """for filepath in validation_data_list_path:
        data=np.load(filepath, mmap_mode='r')['arr_0']
        validation_data_list.append(data)  

    del validation_data_list_path
    validation_data = np.concatenate(validation_data_list)
    del validation_data_list"""
    
    # Load validation data in batches
    for i in range(0, len(validation_data_list_path), batch_size):
        batch_paths = validation_data_list_path[i:i + batch_size]
        batch_data = [np.load(filepath, mmap_mode='r')['arr_0'] for filepath in batch_paths]
        validation_data_batch = np.concatenate(batch_data)
        del batch_data
        validation_data_list.append(validation_data_batch)
        del validation_data_batch  # Clear memory for the next batch

    del validation_data_list_path  # Free memory

    # Concatenate all validation data batches
    validation_data = np.concatenate(validation_data_list)
    del validation_data_list 
    

    print("------------------------------session data created---------------------------------------")

    print("train data shape")
    print(train_data.shape)    #(55813, 1, 22, 1000)
    print(validation_data.shape) #(10845, 1, 22, 1000)
    train_data=train_data[0:500, :,:,:]
    validation_data=validation_data[0:125, :,:,:]
    train_label: np.ndarray = np.random.randint(0, 4, train_data.shape[0])
    validation_label: np.ndarray = np.random.randint(0, 4, validation_data.shape[0])
    train_dataset = ds_time.EEG_Dataset(train_data, train_label, channels_to_set)
    validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, channels_to_set)
    print("EEG_Dataset function called")
    
     # Get number of channels and length of time samples
    C = train_data.shape[2]
    T = train_data.shape[3]
    del train_data
    del validation_data

    train_config = ct.get_config_hierarchical_vEEGNet_training()
    epochs = 80
    # path_to_save_model = 'model_weights_backup'
    path_to_save_model = 'model_weights_backup_tuple_{}'.format(
        indx)  # the folder is model wights backup_iterationOfTheTuple and inside we have one file for each epoch
    os.makedirs(path_to_save_model, exist_ok=True)
    epoch_to_save_model = 1

    # Update train config
    train_config['epochs'] = epochs
    train_config['path_to_save_model'] = path_to_save_model
    train_config['epoch_to_save_model'] = epoch_to_save_model
    train_config['log_dir'] = './logs'
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

    gc.collect()

    df_tmp = pd.DataFrame([], columns=['Reconstruction error with no average_channels and no average_time_samples',
                                        'Reconstruction error with no average_channels and average_time_samples',
                                        'Reconstruction error with average_channels  and no average_time_samples',
                                        'Reconstruction error with average_channels  and average_time_samples'])
    for x_eeg_ in test_data:
        device = train_config['device']
        x_eeg = x_eeg_.astype(np.float32)
        x_eeg = torch.from_numpy(x_eeg)
        x_eeg = x_eeg.to(device)
        model.to(device)
        x_r_eeg = model.reconstruct(x_eeg)
        recon_error_avChannelsF_avTSF, recon_error_avChannelsF_avTST, recon_error_avChannelsT_avTSF, recon_error_avChannelsT_avTST = reconstruction_metrics(
            x_eeg, x_r_eeg, device)
        new_row = {
            'Reconstruction error with no average_channels and no average_time_samples': recon_error_avChannelsF_avTSF,
            'Reconstruction error with no average_channels and average_time_samples': recon_error_avChannelsF_avTST,
            'Reconstruction error with average_channels  and no average_time_samples': recon_error_avChannelsT_avTSF,
            'Reconstruction error with average_channels  and average_time_samples': recon_error_avChannelsT_avTST}
        df_tmp.append(new_row, ignore_index=True)

    row_test_mean = df_tmp.mean(axis=0)
    df_reconstruction.append(row_test_mean,
                            ignore_index=True)  # each row corresponds to a different combination of train/validation to perform on a test set
    
    print('end tuple {}'.format(indx))