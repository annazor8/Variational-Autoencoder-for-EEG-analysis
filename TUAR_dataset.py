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
from welford import Welford 
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(43) 
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

# Process each EDF file
for file_name in sorted(edf_files):
    file_path = os.path.join(directory_path, file_name)
    sub_id, session, time = file_name.split(".")[0].split("_") #split the filname into subject, session and time frame
    raw_mne = mne.io.read_raw_edf(file_path, preload=True)  # Load the EDF file: NB raw_mne.info['chs'] is the only full of information
    raw_mne.pick_channels(channels_to_set, ordered=True) #reorders the channels and drop the ones not contained in channels_to_set
    raw_mne.resample(250) #resample to standardize sampling frequency to 250 Hz
    epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=True, overlap=3) #divide the signal into fixed lenght epoch of 4s with 1 second of overlapping: the overlapping starts from the left side of previous epoch
    epoch_data = epochs_mne.get_data() #trasform the raw eeg into a 3d np array
    std=np.std(epoch_data)
    epoch_data= epoch_data/std #normalization for session
    epoch_data = np.expand_dims(epoch_data, 1) # number of epochs for that signal x 1 x channels x time samples 

    # If session_data[sub_id][session] exists, concatenate
    if session_data[sub_id][session].size > 0:
        new_session=session + '_01'
        session_data[sub_id][new_session] = epoch_data
    else:
        session_data[sub_id][session] = epoch_data
    


#the normalization z2 is performed on each subject or session regardless the training/test set split. 
# the normalization z3 is performed based on the entrie traing set 

#leave one session out from the training for testing: we are loosing the subject information level

def leave_one_session_out(session_data: Dict[str, Dict[str, np.ndarray]], number_of_trials:int =64, shuffle: bool = True): #-> np.ndarray, np.ndarray, np.ndarray
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
    """trials_value=[]
    for el in all_sessions:
        trials_value.append(el.shape[0])

    plt.hist(trials_value, bins='auto', alpha=1.0, rwidth=0.85, color='black', edgecolor='black')
    # Aggiungere titoli e etichette
    plt.title('Istogramma del numero di trials')
    plt.xlabel('# trials')
    plt.ylabel('Frequenza')
    base_path = Path(__file__).resolve(strict=True).parent
    print(base_path)
    plt.savefig(base_path / "signal.png")
    counter = Counter(trials_value)
    most_common = counter.most_common(1)[0] 
    print(f"Il numero più frequente è {most_common[0]} con {most_common[1]} occorrenze.")"""
    if shuffle==True:
        random.shuffle(all_sessions) #shuffle because two adjacent sessions could belong to the same subject
    test_size = int(np.ceil(0.2 * len(all_sessions)))
    test_data=all_sessions[0:test_size]
    train_val_data=all_sessions[test_size:]
    #list of tuples containing the train data as the fist element and the validation data as the second element 
    combinations=[]
    for i in range (len(train_val_data)):
        #do not make shuffle(train_data[i]) because the temporal sequence of the layers in the 3d matrix is important to be preserved 
        train_data = train_val_data[:i] + train_val_data[i+1:] #concatenate the two lists with the + operator
        val_data=train_val_data[i]
        combinations.append((train_data, val_data)) #combinations is a list of tuples (train_data: list, val_data: ndarray)
    return combinations, test_data

def leave_one_subject_out(session_data, number_of_trials:int=64, shuffle: bool=True):
    #initialize an empty dictionary: Dict[str, Dict[str, NDArray]
    subject_data_dict: Dict[str,list] = defaultdict(lambda:  list)
    for subj, value in session_data.items():#key is the subj, value is a dictionary
        # fixed key we are dealing with a subject 
        new_value=list(value.values()) #lista degli array, ciascuno rappresentante una sessione per il soggetto key
        subject_data_dict.update({subj: new_value})
    
    subjects=list(subject_data_dict.keys()) #list of subjects  
    test_size = int(np.ceil(0.2 * len(subjects))) #train size is the 20% of the total size 
    
    if shuffle==True:
        random.shuffle(subjects) #metto un controllo sullo shuffle dei soggetti 
    
    test_data_complete=[] # prendo una sessione per ogni soggetto che uso nel test set
    for i in range(test_size):
        epoch_list=list(subject_data_dict[subjects[i]])
        if len(epoch_list)<2: #se ho solo una sessione per quel soggetto uso quella
            test_data_complete.extend(subject_data_dict[subjects[i]])
        else:
            j=random.randint(0, len(list(subject_data_dict[subjects[i]]))-1) #se ho più sessioni per quel soggetto ne scelgo una a caso
            el=list(subject_data_dict[subjects[i]])[j]
            test_data_complete.append(el)
    test_data=[]
    for el in test_data_complete:
        i=random.randint(0, el.shape [0]-number_of_trials)
        test_data.append(el[i:i+number_of_trials,:,:,:])

    train_val_data_complete=[]
    for k in range(test_size, len(subjects)):
        if len(list(subject_data_dict[subjects[k]]))<2:
            train_val_data_complete.extend(subject_data_dict[subjects[k]])
        else:
            j=random.randint(0, len(list(subject_data_dict[subjects[k]]))-1)
            el=list(subject_data_dict[subjects[k]])[j]
            train_val_data_complete.append(el)

    train_val_data=[]
    for el in train_val_data_complete:
        i=random.randint(0, el.shape [0]-number_of_trials)
        train_val_data.append(el[i:i+number_of_trials,:,:,:])
    combinations=[]
    for i in range (len(train_val_data)):
        train_data = train_val_data[:i] + train_val_data[i+1:]
        val_data=train_val_data[i]
        combinations.append((train_data, val_data))
    
    return combinations, test_data,  train_label, validation_label


combinations1,test_data1= leave_one_session_out(session_data, number_of_trials=1) #NB combinations[0][0] is a list, combinations[0][1] is an array
#combinations2,test_data2, train_label2, validation_label2= leave_one_subject_out(session_data, number_of_trials=2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataloader, loss function, optimizer and lr_scheduler
for indx, combo in enumerate(combinations1): #220 is the max number of combinations
    train_data=np.concatenate(combo[0])
    train_label = np.random.randint(0, 4, train_data[0])
    validation_label = np.random.randint(0, 4, 1)
    validation_data=combo[1]
    train_dataset = ds_time.EEG_Dataset(train_data, train_label, channels_to_set)
    validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, channels_to_set)
    # Get training config
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
    C = 22
    T = 1000 
    # Get model config
    model_config = cm.get_config_hierarchical_vEEGNet(C, T)

    # If the model has also a classifier add the information to training config
    train_config['measure_metrics_during_training'] = model_config['use_classifier']
    train_config['use_classifier'] = model_config['use_classifier']

    # hvEEGNet creation
    model = hvEEGNet.hvEEGNet_shallow(model_config)
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
    train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'])
    validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'])
    loader_list             = [train_dataloader, validation_dataloader]

 

    model = train_generic.train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler, model_artifact = None)
    print('end tuple')
#-------------------------------------------------
"""#RECONSTRUCTION
ch_to_plot='EEG FP2-REF'
x_eeg=data[subject_test]
x_r_eeg = model.reconstruct(x_eeg)


daf=[]
for file in edf_files:
    daf.append({'name': file.split('_')[0], 'trial': file.split('_')[1], 'session': file.split('_')[2]})
df = pd.DataFrame(daf)
print(df)
    Exception has occurred: AttributeError
'tuple' object has no attribute 'append'
  File "/home/azorzetto/data1/Variational-Autoencoder-for-EEG-analysis/TUAR_dataset.py", line 75, in leave_one_session_out
    all_sessions.append(el[i:i+number_of_trials,:,:,:])
  File "/home/azorzetto/data1/Variational-Autoencoder-for-EEG-analysis/TUAR_dataset.py", line 162, in <module>
    combinations1,test_data1, train_label1, validation_label1= leave_one_session_out(session_data, number_of_trials=1) #NB combinations[0][0] is a list, combinations[0][1] is an array
AttributeError: 'tuple' object has no attribute 'append'
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
combinations1,test_data1= leave_one_session_out(session_data, number_of_trials=1) #NB combinations[0][0] is a list, combinations[0][1] is an array
train_dataset=ds_time.EEG_Dataset_list(combinations1[0][0])
random_sampler = torch.utils.data.RandomSampler((len(train_dataset)))
batch_sampler = torch.utils.data.BatchSampler(random_sampler, batch_size = train_config['batch_size'], drop_last=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle = False, batch_sampler=batch_sampler)

validation_dataloader= torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
loader_list             = [train_dataloader, validation_dataloader]