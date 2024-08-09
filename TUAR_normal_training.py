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
from tuar_training_utils import get_data_TUAR
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
edf_files = [file for file in all_files if file.endswith('.edf')]
start_index=0
end_index=20
all_sessions = []
    # dataend_index structure Dict[str, Dict[str, NDArray] --> Dict[subj_id, Dict[sess, NDArray]]
session_data: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.array([])))
    # Process each EDF file
if end_index == None:
    end_index=len(edf_files) -1

subj_list=[]

for file_name in sorted(edf_files)[start_index:end_index]:
    file_path = os.path.join(directory_path, file_name)
    sub_id, session, time = file_name.split(".")[0].split(
        "_")  # split the filname into subject, session and time frame
    if sub_id in subj_list:
        continue
    else:
        subj_list.append(sub_id)
        raw_mne = mne.io.read_raw_edf(file_path,
                                        preload=False)  # Load the EDF file: NB raw_mne.info['chs'] is the only full of information
        raw_mne.pick_channels(channels_to_set,
                                ordered=True)  # reorders the channels and drop the ones not contained in channels_to_set
        raw_mne.resample(250)  # resample to standardize sampling frequency to 250 Hz
        epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=False)  # divide the signal into fixed lenght epoch of 4s with 1 second of overlapping: the overlapping starts from the left side of previous epoch
        del raw_mne
        epoch_data = epochs_mne.get_data(copy=False)  # trasform the raw eeg into a 3d np array
        del epochs_mne
        mean=np.mean(epoch_data)
        std = np.std(epoch_data)
        epoch_data = (epoch_data-mean) / std  # normalization for session
        del mean
        del std
        epoch_data = np.expand_dims(epoch_data, 1)  # number of epochs for that signal x 1 x channels x time samples
    # initialize a list containing all sessions
        all_sessions.append(epoch_data)

dataset=np.concatenate(all_sessions)
print(dataset.shape)

print("complete dataset")
Calculate_statistics(directory_path, start_index=0, end_index=30)

test_size = int(np.ceil(0.2 * len(all_sessions)))
test_data = np.concatenate(all_sessions[0:test_size])
print("test set")
Calculate_statistics(directory_path, start_index=0, end_index=test_size)
validation_data = np.concatenate(all_sessions[test_size:2*test_size])
print("validation set")
Calculate_statistics(directory_path, start_index=test_size, end_index=2*test_size)
train_data=np.concatenate(all_sessions[2*test_size:])
print("train set")
Calculate_statistics(directory_path, start_index=2*test_size, end_index=None)

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
path_to_save_model = 'model_weights_backup3' # the folder is model wights backup_iterationOfTheTuple and inside we have one file for each epoch
os.makedirs(path_to_save_model, exist_ok=True)
epoch_to_save_model = 1

# Update train config
train_config['epochs'] = epochs
train_config['path_to_save_model'] = path_to_save_model
train_config['epoch_to_save_model'] = epoch_to_save_model
train_config['log_dir'] = './logs3'
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