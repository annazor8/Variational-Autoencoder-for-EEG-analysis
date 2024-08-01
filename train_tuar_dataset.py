import torch
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm
import os
import numpy as np
import pandas as pd 
from tuar_training_utils import reconstruction_metrics, get_data_TUAR, leave_one_session_out
from torch.utils.data import DataLoader

np.random.seed(43)

# Extract combinations1 and test_data1 from the loaded data
df_reconstruction = pd.DataFrame([], columns=[
    'Reconstruction error with no average_channels and no average_time_samples',
    'Reconstruction error with no average_channels and average_time_samples',
    'Reconstruction error with average_channels  and no average_time_samples',
    'Reconstruction error with average_channels  and average_time_samples'])

channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                    'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                    'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                    'EEG T1-REF', 'EEG T2-REF']

directory_path='/home/azorzetto/dataset/01_tcp_ar' #dataset in local PC
session_data=get_data_TUAR(directory_path)
print("loaded dataset")

combinations1, test_data1 = leave_one_session_out(session_data)  # NB combinations[0][0] is a list, combinations[0][1] is an array
for indx, combo in enumerate(combinations1):  # 220 is the max number of combinations
    train_data_list: list = combo[0]
    validation_data_list: list = combo[1]
    train_data = np.concatenate(train_data_list)
    validation_data = np.concatenate(validation_data_list)
    train_label: np.ndarray = np.random.randint(0, 4, train_data.shape[0])
    validation_label: np.ndarray = np.random.randint(0, 4, validation_data.shape[0])
    train_dataset = ds_time.EEG_Dataset(train_data, train_label, channels_to_set)
    validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, channels_to_set)

    train_config = ct.get_config_hierarchical_vEEGNet_training()
    epochs = 80
    # path_to_save_model = 'model_weights_backup'
    path_to_save_model = 'model_weights_backup_{}'.format(
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

    # Get number of channels and length of time samples
    C = train_data.shape[2]
    T = train_data.shape[3]
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

    # Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True,
                                    num_workers=6, drop_last=True)  # the batch is composed by a single sample, the sessions should be shuffled
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=8, shuffle=True,
                                        num_workers=6, drop_last=True)  # shuffle = False to be kept because the order of the trials in a session is important to be maintained
    loader_list = [train_dataloader, validation_dataloader]

    train_generic.train(model=model, loss_function=loss_function, optimizer=optimizer,
                                                                            loader_list=loader_list, train_config=train_config, lr_scheduler=lr_scheduler,
                                                                            model_artifact=None)

    
    df_tmp = pd.DataFrame([], columns=['Reconstruction error with no average_channels and no average_time_samples',
                                        'Reconstruction error with no average_channels and average_time_samples',
                                        'Reconstruction error with average_channels  and no average_time_samples',
                                        'Reconstruction error with average_channels  and average_time_samples'])
    for x_eeg_ in test_data1:
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