
from collections import defaultdict
import torch
from typing import Dict, List, Tuple
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic
from library.config import config_training as ct
from library.config import config_model as cm
import os
import mne
import numpy as np
import pandas as pd 
from tuar_training_utils import leave_one_session_out, reconstruction_metrics
from torch.utils.data import DataLoader

#Inside the module dtw analysis of the library there is a function that computed the dtw between two tensor.
# Read the function documentation to more info about the computation and the input parameters.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataloader, loss function, optimizer and lr_scheduler

"""
average_train_loss=df_train_loss.mean(axis=1)
average_val_loss=df_val_loss.mean(axis=1)
df_reconstruction.to_csv(path_or_buf='/home/azorzetto/data1/Variational-Autoencoder-for-EEG-analysis/performances/Reconstruction_performances.csv', index=False)
average_train_loss.to_csv(path_or_buf='/home/azorzetto/data1/Variational-Autoencoder-for-EEG-analysis/performances/training_performances.csv', index=False)
average_val_loss.to_csv(path_or_buf='/home/azorzetto/data1/Variational-Autoencoder-for-EEG-analysis/performances/validation_performances.csv', index=False)
    
def plot_ORIGINAL_vs_RECONSTRUCTED(ch_to_plot : str, channels_to_set : list, x_eeg : np.ndarray, x_r_eeg : np.ndarray, trial : int = 3):
    t = torch.linspace(2, 6, T)
    idx = channels_to_set.index(ch_to_plot)
    # Plot the original and reconstructed signal
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(1, 1, figsize = (12, 8))

    ax.plot(t, x_eeg.squeeze()[trial] [idx], label = 'Original EEG', color = 'red', linewidth = 2)
    ax.plot(t, x_r_eeg.squeeze()[trial] [idx], label = 'Reconstructed EEG', color = 'green', linewidth = 1)

    ax.legend()
    ax.set_xlim([2, 4]) # Note that the original signal is 4s long. Here I plot only 2 second to have a better visualization
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r"Amplitude [$\mu$V]")
    ax.set_title("Ch. {}".format(ch_to_plot))
    ax.grid(True)

    fig.tight_layout()
    base_path = Path(__file__).resolve(strict=True).parent

    print(base_path)
    plt.savefig(base_path / "signal.png")
    fig.show()"""

# train_data_list : list=combinations1[0][0]
# validation_data_list : list=combinations1[0][1]
# train_data=np.concatenate(train_data_list)
# validation_data=np.concatenate(validation_data_list)
# train_label : np.ndarray= np.random.randint(0, 4, train_data.shape[0])
# validation_label : np.ndarray= np.random.randint(0, 4, validation_data.shape[0])
# train_dataset = ds_time.EEG_Dataset(train_data, train_label, channels_to_set)
# validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, channels_to_set)
#
# train_config = ct.get_config_hierarchical_vEEGNet_training()
#
# epochs = 2
#     #path_to_save_model = 'model_weights_backup'
# path_to_save_model = 'model_weights_backup_{}'.format(1) #the folder is model wights backup_iterationOfTheTuple and inside we have one file for each epoch
# os.makedirs(path_to_save_model, exist_ok=True)
# epoch_to_save_model = 1
#
#     # Update train config
# train_config['epochs'] = epochs
# train_config['path_to_save_model'] = path_to_save_model
# train_config['epoch_to_save_model'] = epoch_to_save_model
#    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     # Get model
#
#     # Get number of channels and length of time samples
# C  = train_data.shape[2]
# T  = train_data.shape[3]
#     # Get model config
# model_config = cm.get_config_hierarchical_vEEGNet(C, T)
#
#     # If the model has also a classifier add the information to training config
# train_config['measure_metrics_during_training'] = model_config['use_classifier']
# train_config['use_classifier'] = model_config['use_classifier']
#     # hvEEGNet creation
# model = hvEEGNet.hvEEGNet_shallow(model_config) #new model is instantiated for each iteration of the loop.
#     # Declare loss function
#         # This method return the PyTorch loss function required by the training function.
#         # The loss function for hvEEGNet is not directy implemented in PyTorch since it is a combination of different losses. So I have to create my own function to combine all the components.
# loss_function = train_generic.get_loss_function(model_name = 'hvEEGNet_shallow', config = train_config)
#
#     # Create optimizer
# optimizer = torch.optim.AdamW(model.parameters(),
#                                 lr = train_config['lr'],
#                                 weight_decay = train_config['optimizer_weight_decay']
#                                 )
#     # (OPTIONAL) Setup lr scheduler
# if train_config['use_scheduler'] :
#     lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
# else:
#     lr_scheduler = None
#
#     # Move the model to training device (CPU/GPU)
# model.to(train_config['device'])
#
#     # Create dataloader
# train_dataloader = DataLoader(dataset=train_dataset,batch_size = 6, shuffle = True, num_workers=6) #the batch is composed by a single sample, the sessions should be shuffled
# validation_dataloader= DataLoader(dataset=validation_dataset, batch_size =6, shuffle = True, num_workers=6)#shuffle = False to be kept because the order of the trials in a session is important to be maintained
# loader_list             = [train_dataloader, validation_dataloader]
#
#
# train_loss_list, validation_loss_list, epoch_list = train_generic.train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler, model_artifact = None)
#
# train_loss_list = [loss.detach().numpy() for loss in train_loss_list]
# validation_loss_list = [loss.detach().numpy() for loss in validation_loss_list]
#
#
# df_train_loss.insert(loc=df_train_loss.shape[1], column='training loss', value= train_loss_list,allow_duplicates= True)
# df_val_loss.insert(loc=df_val_loss.shape[1], column='validation loss', value= validation_loss_list,allow_duplicates= True)
# df_tmp=pd.DataFrame([], columns=['Reconstruction error with no average_channels and no average_time_samples', 'Reconstruction error with no average_channels and average_time_samples', 'Reconstruction error with average_channels  and no average_time_samples', 'Reconstruction error with average_channels  and average_time_samples'])
# for x_eeg_ in test_data1:
#     device = train_config['device']
#     x_eeg = x_eeg_.astype(np.float32)
#     x_eeg = torch.from_numpy(x_eeg)
#     x_eeg = x_eeg.to(device)
#     model.to(device)
#     x_r_eeg= model.reconstruct(x_eeg)
#     recon_error_avChannelsF_avTSF, recon_error_avChannelsF_avTST, recon_error_avChannelsT_avTSF,recon_error_avChannelsT_avTST=reconstruction_metrics(x_eeg, x_r_eeg, device)
#     new_row={'Reconstruction error with no average_channels and no average_time_samples' : recon_error_avChannelsF_avTSF, 'Reconstruction error with no average_channels and average_time_samples' : recon_error_avChannelsF_avTST, 'Reconstruction error with average_channels  and no average_time_samples' : recon_error_avChannelsT_avTSF, 'Reconstruction error with average_channels  and average_time_samples' : recon_error_avChannelsT_avTST}
#     df_tmp.append(new_row, ignore_index=True)
#
# row_test_mean=df_tmp.mean(axis=0)
# df_reconstruction.append(row_test_mean, ignore_index=True) #each row corresponds to a different combination of train/validation to perform on a test set
    
"""recon_error = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device, average_channels = False, average_time_samples = False)
print("Reconstruction error with average_channels = False and average_time_samples = False. Shape of the output tensor = {}\n{}\n".format(recon_error.shape, recon_error))

recon_error = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device, average_channels = False, average_time_samples = True)
print("Reconstruction error with average_channels = False and average_time_samples = True. Shape of the output tensor = {}\n{}\n".format(recon_error.shape, recon_error))

recon_error = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device, average_channels = True, average_time_samples = False)
print("Reconstruction error with average_channels = True  and average_time_samples = False. Shape of the output tensor = {}\n{}\n".format(recon_error.shape, recon_error))

recon_error = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device, average_channels = True, average_time_samples = True)
print("Reconstruction error with average_channels = True  and average_time_samples = True. Shape of the output tensor = {}\n{}\n".format(recon_error.shape, recon_error))"""


def main():
    np.random.seed(43)

    # df_train_loss = pd.DataFrame([])
    # df_val_loss = pd.DataFrame([])
    # df_reconstruction = pd.DataFrame([], columns=[
    #     'Reconstruction error with no average_channels and no average_time_samples',
    #     'Reconstruction error with no average_channels and average_time_samples',
    #     'Reconstruction error with average_channels  and no average_time_samples',
    #     'Reconstruction error with average_channels  and average_time_samples'])
    #directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar'
    directory_path='/home/azorzetto/dataset/01_tcp_ar'

    #directory_path = '/home/lmonni/Documents/01_tcp_ar'
    channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                       'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                       'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                       'EEG T1-REF', 'EEG T2-REF']
    # List all files in the directory
    all_files = os.listdir(directory_path)
    # Filter out only EDF files
    edf_files = [file for file in all_files if file.endswith('.edf')][0:20]

    # data structure Dict[str, Dict[str, NDArray] --> Dict[subj_id, Dict[sess, NDArray]]
    session_data: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.array([])))
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
        epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=False,
                                                  overlap=3)  # divide the signal into fixed lenght epoch of 4s with 1 second of overlapping: the overlapping starts from the left side of previous epoch
        del raw_mne
        epoch_data = epochs_mne.get_data(copy=False)  # trasform the raw eeg into a 3d np array
        del epochs_mne
        mean=np.mean(epoch_data)
        std = np.std(epoch_data)
        epoch_data = (epoch_data-mean) / std  # normalization for session
        epoch_data = np.expand_dims(epoch_data, 1)  # number of epochs for that signal x 1 x channels x time samples
        # If session_data[sub_id][session] exists, concatenate
        all_session.append(epoch_data)
        if session_data[sub_id][session].size > 0:
            new_session = session + '_01'
            session_data[sub_id][new_session] = epoch_data
        else:
            session_data[sub_id][session] = epoch_data

    global_array=np.concatenate(all_session)
    global_min=np.min(global_array)
    global_max=np.max(global_array)
    
    combinations1, test_data1 = leave_one_session_out(
        session_data, global_min, global_max)  # NB combinations[0][0] is a list, combinations[0][1] is an array

    # combinations2,test_data2, train_label2, validation_label2= leave_one_subject_out(session_data, number_of_trials=2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Dataloader, loss function, optimizer and lr_scheduler


    df_train_loss = pd.DataFrame([])
    df_val_loss = pd.DataFrame([])
    df_reconstruction = pd.DataFrame([], columns=[
        'Reconstruction error with no average_channels and no average_time_samples',
        'Reconstruction error with no average_channels and average_time_samples',
        'Reconstruction error with average_channels  and no average_time_samples',
        'Reconstruction error with average_channels  and average_time_samples'])

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
        gamma_dtw=0.1
        epochs = 80
        # path_to_save_model = 'model_weights_backup'
        path_to_save_model = 'model_weights_backup_{}'.format(
            indx)  # the folder is model wights backup_iterationOfTheTuple and inside we have one file for each epoch
        os.makedirs(path_to_save_model, exist_ok=True)
        epoch_to_save_model = 1

        # Update train config
        train_config['epochs'] = epochs
        train_config['gamma_dtw'] = gamma_dtw
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

        #train_loss_list = [loss.detach().numpy() for loss in train_loss_list]
        #validation_loss_list = [loss.detach().numpy() for loss in validation_loss_list]

        #df_train_loss.insert(loc=df_train_loss.shape[1], column='training loss', value=train_loss_list, allow_duplicates=True)
        #df_val_loss.insert(loc=df_val_loss.shape[1], column='validation loss', value=validation_loss_list, allow_duplicates=True)
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


if __name__ == "__main__":
    main()