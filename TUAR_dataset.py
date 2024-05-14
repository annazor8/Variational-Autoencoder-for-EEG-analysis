import mne
from mne.io import Raw
import torch

from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm
import os
import mne
import numpy as np

# Directory containing the EDF files
directory_path = '/home/azorzetto/data1/TCParRidotto'
channels_to_set=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG EKG1-REF', 'EEG T1-REF', 'EEG T2-REF']
# List all files in the directory
all_files = os.listdir(directory_path)

# Filter out only EDF files
edf_files = [file for file in all_files if file.endswith('.edf')]

data=[] #I create a tuple where to store the arrays
# Process each EDF file
for file_name in edf_files:
    file_path = os.path.join(directory_path, file_name)
    try:
        # Load the EDF file
        tmp = mne.io.read_raw_edf(file_path)
        #common channel set
        ch = tmp.ch_names
        ch_drop = [cc for cc in ch if cc not in channels_to_set]
        tmp = tmp.drop_channels(ch_drop)
        tmp = tmp.reorder_channels(channels_to_set)
        #resample to standardize sampling frequency to 250 Hz
        tmp=tmp.resample(250)
        # Get the data as a NumPy array (if needed)
        tmp_array = tmp.get_data()
        data.append(tmp_array)

    except Exception as e:
        print(f"Failed to load {file_name}: {e}")

dataset=np.empty((3,1,22,1000))
i=0
for el in data:

    start = np.random.randint(0, (el.shape[1])-1000)
    tmp1=el[:,start:start+1000]
    tmp1 = tmp1.reshape(1, tmp1.shape[0], tmp1.shape[1])
    dataset[i,:,:,:] = tmp1
    i=i+1

# Check the shape of the final dataset
print("Final dataset shape:", dataset.shape)

np.random.shuffle(dataset)
train_data=dataset[0:2,:,:,:] #(1, 1, 22, 1000)
validation_data=dataset[2:,:,:,:] #(1, 1, 22, 1000)
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