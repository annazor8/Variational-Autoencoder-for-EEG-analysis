import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm
import numpy as np
import pandas as pd 
from utils import reconstruction_metrics, get_data_TUAR, leave_one_session_out
from torch.utils.data import DataLoader
import pickle
np.random.seed(43)

# Extract combinations1 and test_data1 from the loaded data


channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                    'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                    'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                    'EEG T1-REF', 'EEG T2-REF']

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'E1', 'E2']

#directory_path='/home/azorzetto/dataset/01_tcp_ar' #dataset in local PC
directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar'
session_data=get_data_TUAR(directory_path, start_index=10, end_index=20)
print("loaded dataset")
os.makedirs('dataset_cross_val', exist_ok=True)

combinations1, test_data1 = leave_one_session_out(session_data)  # NB combinations[0][0] is a list, combinations[0][1] is an array

file_name='test_data.npy'
file_path = os.path.join('dataset_cross_val', file_name)

np.savez_compressed(file_path, np.concatenate(test_data1))

for indx, combo in enumerate(combinations1):  # 220 is the max number of combinations
    
    train_data_list: list = combo[0]
    validation_data_list: list = combo[1]

    train_data = np.concatenate(train_data_list)
    validation_data = np.concatenate(validation_data_list)
    del train_data_list
    del validation_data_list

    train_label: np.ndarray = np.random.randint(0, 4, train_data.shape[0])
    validation_label: np.ndarray = np.random.randint(0, 4, validation_data.shape[0])

    train_dataset = ds_time.EEG_Dataset(train_data, train_label, channels_to_set)
    validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, channels_to_set)
    # Get number of channels and length of time samples
    C = train_data.shape[2]
    T = train_data.shape[3]
    
    file_name='dataset_iteration_{}.npz'.format(indx)
    file_path = os.path.join('dataset_cross_val', file_name)

    #save as npz for reproducibility
    np.savez_compressed(file_path, validation_data=validation_data, train_data=train_data, train_label=train_label, validation_label=validation_label)
    del train_data
    del validation_data

    train_config = ct.get_config_hierarchical_vEEGNet_training()

    # Update train config
    epochs = 80
    train_config['epochs'] = epochs

    # path_to_save_model = 'model_weights_backup'
  
    path_to_save_model = 'model_weights_backup_iteration_{}'.format(indx)  # Folder name
    file_path = os.path.join('dataset_cross_val', path_to_save_model)  # Full path
    os.makedirs(file_path, exist_ok=True)
    train_config['path_to_save_model'] = path_to_save_model

    epoch_to_save_model = 5
    train_config['epoch_to_save_model'] = epoch_to_save_model

    log_dir_path = os.path.join('dataset_cross_val', 'logs{}'.format(indx))  # Combine with 'logs' folder

    # Update train_config with the new log directory path
    train_config['log_dir'] = log_dir_path

    # Create the logs directory within the "dataset_cross_val" directory
    os.makedirs(train_config['log_dir'], exist_ok=True)


 
    # Get model config
    model_config = cm.get_config_hierarchical_vEEGNet(C, T)

    # If the model has also a classifier add the information to training config
    train_config['measure_metrics_during_training'] = model_config['use_classifier']
    train_config['use_classifier'] = model_config['use_classifier']

    train_config['early_stopping'] = False #if you want to activate the early stopping
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

    
    results = [] #list containing the dictionaries
    av_reconstruction_error=[]
    i=0
    to_save_eeg=[]
    for i in range(test_data1.shape[0]-1):
    #for i in range(10):
        x_eeg_=test_data1[i]
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
        i=i+1

    filename='resconstruction_error_iteration_{}.pkl'.format(indx)
    file_path = os.path.join('dataset_cross_val', filename)
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)

    to_save_eeg=np.concatenate(to_save_eeg)

    filename='reconstructed_eeg_iteration_{}.npz'.format(indx)
    file_path = os.path.join('dataset_cross_val', filename)
    np.savez_compressed(file_path, x_r_eeg=to_save_eeg)

    df_reconstuction_error = pd.DataFrame(av_reconstruction_error)

    filename='mean_reconstruction_errors_iteration_{}.csv'.format(indx)
    file_path = os.path.join('dataset_cross_val', filename)
    df_reconstuction_error.to_csv(file_path, index=False)

    print('end tuple {}'.format(indx))