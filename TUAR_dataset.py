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

np.random.seed(43) 
# Directory containing the EDF files
directory_path = '/home/azorzetto/data1/TCParRidotto'
channels_to_set=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG EKG1-REF', 'EEG T1-REF', 'EEG T2-REF']
# List all files in the directory
all_files = os.listdir(directory_path)

# Filter out only EDF files
edf_files = [file for file in all_files if file.endswith('.edf')]

data=[] #I create a list where to store the arrays. Each array corresponds to an EDF file channel x time_sampel
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
        #tmp_array = tmp.get_data()
        #data.append(tmp_array)
        epochs = mne.make_fixed_length_epochs(tmp, duration=4, preload=True, overlap=1)
        tmp1=epochs.get_data()
        tmp1=tmp1.reshape(tmp1.shape[0],1,tmp1.shape[1],tmp1.shape[2])
        np.random.shuffle(tmp1) #shuffle along the first dimension, so shuffle the trials 
        data.append(tmp1)
    except Exception as e:
        print(f"Failed to load {file_name}: {e}")


#the input data is a 4D matrix representing either the training set or the validation set of one or multiple subject(s)
def normalization_z3(data):

    mean= np.mean(data)
    std=np.std(data)
    data=(data - mean)/std
    return data

#the normalization z2 is performed on each subject or session regardless the training/test set split. In our case it's the same because the trinaing se tis composed by only 1 subject for now
#in this case data should be a list of 4d arrays: each array is corresponding to a subject
def normalization_z2(data):
    for i, el in enumerate(data):
        norm_el=normalization_z3(el)
        if i==0:
            final=norm_el
        else:
            final=np.concatenate((final, norm_el), axis=0)
    return final
#set the subject on whose eeg to train
subject_test=0
subject_train=1,2
dataset=[data[i] for i in subject_train]
#normalize using the z3 normalization: all the elements in the training set 
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