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
import os
import mne
import numpy as np
import random
import pandas as pd
from welford import Welford

np.random.seed(43) 
#Inside the module dtw analysis of the library there is a function that computed the dtw between two tensor.
# Read the function documentation to more info about the computation and the input parameters.
def reconstruction_metrics(x_eeg, x_r_eeg, device):
    recon_error_avChannelsF_avTSF = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device, average_channels = False, average_time_samples = False)
    recon_error_avChannelsF_avTST = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device, average_channels = False, average_time_samples = True)
    recon_error_avChannelsT_avTSF = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device, average_channels = True, average_time_samples = False)
    recon_error_avChannelsT_avTST = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device, average_channels = True, average_time_samples = True)
    return recon_error_avChannelsF_avTSF, recon_error_avChannelsF_avTST, recon_error_avChannelsT_avTSF,recon_error_avChannelsT_avTST

# Directory containing the EDF files
#directory_path='C:\Users\albin\OneDrive\Desktop\TCParRidotto'
#directory_path = r'C:\Users\albin\OneDrive\Desktop\TCParRidotto'

#directory_path = '/home/azorzetto/data1/TCParRidotto'
directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar'
channels_to_set=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
# List all files in the directory
all_files = os.listdir(directory_path)
# Filter out only EDF files
edf_files = [file for file in all_files if file.endswith('.edf')]

# data structure Dict[str, Dict[str, NDArray] --> Dict[subj_id, Dict[sess, NDArray]]
session_data: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.array([])))



# Stampa il risultato per verifica (opzionale)
# for sub_id, sessions in session_data.items():
#     for session, data in sessions.items():
#         print(f"Sub ID: {sub_id}, Session: {session}, Data shape: {data.shape}")

#leave one session out from the training for testing: we are loosing the subject information level

def leave_one_session_out_taining(session_data: Dict[str, Dict[str, np.ndarray]], number_of_trials:int =64): #-> np.ndarray, np.ndarray, np.ndarray
    #a list of the dictionaries [{session: arrays}, {session: arrays}, {session: arrays},...]
    #list of defautdict [{'session': array}]
    list_dict_session=session_data.values() #the type is dict_values
    all_sessions_complete=[] #initialize a list containing all sessions
    for el in list_dict_session:
        all_sessions_complete.extend(list(el.values()))
    all_sessions=[]
    for el in all_sessions_complete:
        i=random.randint(0, el.shape [0]-number_of_trials)
        all_sessions.append(el[i:i+number_of_trials,:,:,:])
    #il numero di trials minimo è 64
    trials_value=[]
    for el in all_sessions:
        trials_value.append(el.shape[0])

    test_size = int(np.ceil(0.2 * len(all_sessions)))
    test_data=all_sessions[0:test_size]
    train_val_data=all_sessions[test_size:]
    del all_sessions
    #list of tuples containing the train data as the fist element and the validation data as the second element 
    df_train_loss = pd.DataFrame([])
    df_val_loss = pd.DataFrame([])
    df_reconstruction=pd.DataFrame([], columns=['Reconstruction error with no average_channels and no average_time_samples', 'Reconstruction error with no average_channels and average_time_samples', 'Reconstruction error with average_channels  and no average_time_samples', 'Reconstruction error with average_channels  and average_time_samples'])
    
    for i in range (0, len(train_val_data), 4):
        indx=0
        #do not make shuffle(train_data[i]) because the temporal sequence of the layers in the 3d matrix is important to be preserved 
        train_data = train_val_data[:i] + train_val_data[i+4:] #concatenate the two lists with the + operator
        val_data=train_val_data[i:i+4]
        train_label : np.ndarray= np.random.randint(0, 4, train_data[0].shape[0])
        validation_label : np.ndarray= np.random.randint(0, 4, val_data[0].shape[0])
        train_dataset = ds_time.EEG_Dataset_list(train_data, train_label, channels_to_set)
        validation_dataset = ds_time.EEG_Dataset_list(val_data, validation_label, channels_to_set)

        train_config = ct.get_config_hierarchical_vEEGNet_training()

        epochs = 30
        #path_to_save_model = 'model_weights_backup'
        path_to_save_model = 'model_weights_backup_{}'.format(indx) #the folder is model wights backup_iterationOfTheTuple and inside we have one file for each epoch 
        os.makedirs(path_to_save_model, exist_ok=True)
        epoch_to_save_model = 1

        # Update train config
        train_config['epochs'] = epochs
        train_config['path_to_save_model'] = path_to_save_model
        train_config['epoch_to_save_model'] = epoch_to_save_model
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Get model

        # Get number of channels and length of time samples
        C  = train_data[0].shape[2]
        T  = train_data[0].shape[3]
        # Get model config
        model_config = cm.get_config_hierarchical_vEEGNet(C, T)

        # If the model has also a classifier add the information to training config
        train_config['measure_metrics_during_training'] = model_config['use_classifier']
        train_config['use_classifier'] = model_config['use_classifier']
        # hvEEGNet creation
        model = hvEEGNet.hvEEGNet_shallow(model_config) #new model is instantiated for each iteration of the loop.
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

        # Create dataloader
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size = None, shuffle = True, num_workers=10) #the batch is composed by a single sample, the sessions should be shuffled
        validation_dataloader= torch.utils.data.DataLoader(dataset=validation_dataset, batch_size =None, shuffle = False, num_workers=10)#shuffle = False to be kept because the order of the trials in a session is important to be maintained
        loader_list             = [train_dataloader, validation_dataloader]
    

        train_loss_list, validation_loss_list, epoch_list = train_generic.train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler, model_artifact = None)

        train_loss_list = [loss.detach().numpy() for loss in train_loss_list]
        validation_loss_list = [loss.detach().numpy() for loss in validation_loss_list]


        df_train_loss.insert(loc=df_train_loss.shape[1], column='training loss', value= train_loss_list,allow_duplicates= True)
        df_val_loss.insert(loc=df_val_loss.shape[1], column='validation loss', value= validation_loss_list,allow_duplicates= True)
        df_tmp=pd.DataFrame([], columns=['Reconstruction error with no average_channels and no average_time_samples', 'Reconstruction error with no average_channels and average_time_samples', 'Reconstruction error with average_channels  and no average_time_samples', 'Reconstruction error with average_channels  and average_time_samples'])
        for x_eeg_ in test_data:
            device = train_config['device']
            x_eeg = x_eeg_.astype(np.float32)
            x_eeg = torch.from_numpy(x_eeg)
            x_eeg = x_eeg.to(device)
            model.to(device)
            x_r_eeg= model.reconstruct(x_eeg)
            recon_error_avChannelsF_avTSF, recon_error_avChannelsF_avTST, recon_error_avChannelsT_avTSF,recon_error_avChannelsT_avTST=reconstruction_metrics(x_eeg, x_r_eeg, device)
            new_row={'Reconstruction error with no average_channels and no average_time_samples' : recon_error_avChannelsF_avTSF, 'Reconstruction error with no average_channels and average_time_samples' : recon_error_avChannelsF_avTST, 'Reconstruction error with average_channels  and no average_time_samples' : recon_error_avChannelsT_avTSF, 'Reconstruction error with average_channels  and average_time_samples' : recon_error_avChannelsT_avTST}
            df_tmp.append(new_row, ignore_index=True)

        row_test_mean=df_tmp.mean(axis=0) 
        df_reconstruction.append(row_test_mean, ignore_index=True) #each row corresponds to a different combination of train/validation to perform on a test set
    return df_train_loss,df_val_loss, df_reconstruction

df_train_loss,df_val_loss, df_reconstruction= leave_one_session_out_taining(session_data)