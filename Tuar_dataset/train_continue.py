import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm

train_session=0
model_epoch=80

data = np.load('/home/azorzetto/data1/Train/train{}/dataset.npz'.format(train_session))
channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                       'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                       'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                       'EEG T1-REF', 'EEG T2-REF']
test_data=data['test_data']
train_data=data['train_data']
validation_data=data['validation_data']
train_label=data['train_label']
validation_label=data['validation_label']

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
path_to_save_model = '/home/azorzetto/data1/Train/train{}/model_weights_backup_3cont'.format(train_session) # the folder is model wights backup_iterationOfTheTuple and inside we have one file for each epoch
os.makedirs(path_to_save_model, exist_ok=True)
epoch_to_save_model = 1

# Update train config
train_config['epochs'] = epochs
train_config['path_to_save_model'] = path_to_save_model
train_config['epoch_to_save_model'] = epoch_to_save_model
train_config['log_dir'] = './logs3cont'
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
model.load_state_dict(torch.load(
    '/home/azorzetto/data1/Train/train{}/model_weights_backup_2cont/model_epoch{}.pth'.format(train_session, model_epoch), 
    map_location= torch.device('cpu'), 
    weights_only=True
    ))
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
