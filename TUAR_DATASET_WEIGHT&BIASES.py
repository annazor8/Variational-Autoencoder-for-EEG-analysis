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
from library.training import wandb_training as wt

np.random.seed(43) 
# Directory containing the EDF files
#directory_path='C:\Users\albin\OneDrive\Desktop\TCParRidotto'
#directory_path = r'C:\Users\albin\OneDrive\Desktop\TCParRidotto'

#directory_path = '/home/azorzetto/data1/TCParRidotto'
directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar'
channels_to_set=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF', 'EEG T1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 
                 'EEG C4-REF', 'EEG T4-REF', 'EEG A2-REF','EEG T2-REF', 'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF']
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

def leave_one_session_out(session_data: Dict[str, Dict[str, np.ndarray]], number_of_trials:int =64): #-> np.ndarray, np.ndarray, np.ndarray
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

def leave_one_subject_out(session_data, number_of_trials:int=64):
    #initialize an empty dictionary: Dict[str, Dict[str, NDArray]
    subject_data_dict: Dict[str,list] = defaultdict(lambda:  list)
    for subj, value in session_data.items():#key is the subj, value is a dictionary
        # fixed key we are dealing with a subject 
        new_value=list(value.values()) #lista degli array, ciascuno rappresentante una sessione per il soggetto key
        subject_data_dict.update({subj: new_value})
    
    subjects=list(subject_data_dict.keys()) #list of subjects  
    test_size = int(np.ceil(0.2 * len(subjects))) #train size is the 20% of the total size 
    
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
    combinations : list=[]
    for i in range (len(train_val_data)):
        train_data : list = train_val_data[:i] + train_val_data[i+1:]
        val_data : np.ndarray=train_val_data[i]
        combinations.append((train_data, val_data))
    
    return combinations, test_data

number_of_trials=64
combinations1,test_data1= leave_one_session_out(session_data, number_of_trials=number_of_trials) #NB combinations[0][0] is a list, combinations[0][1] is an array
C = 22
T = 1000 
#combinations2,test_data2, train_label2, validation_label2= leave_one_subject_out(session_data, number_of_trials=2)
# Training parameters to change (for more info check the function get_config_hierarchical_vEEGNet_training)

epochs = 2
epoch_to_save_model = 1
project_name = "Example_project"                # Name of wandb project
name_training_run = "first_test_wandb"          # Name of the training run
model_artifact_name = "temporary_artifacts"     # Name of the artifact used to save the model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataloader, loss function, optimizer and lr_scheduler
for indx, combo in enumerate(combinations1): #220 is the max number of combinations
    path_to_save_model = 'model_weights_backup_{}'.format(indx) #the folder is model wights backup_iterationOfTheTuple and inside we have one file for each epoch 


    #combo is the tuple
    train_label = np.random.randint(0, 4, combo[0][0].shape[0]) #combo[0] is a list
    validation_label = np.random.randint(0, 4, 1)
    
    # Get model config
    model_config = cm.get_config_hierarchical_vEEGNet(C, T)

    # Get training config
    train_config = ct.get_config_vEEGNet_training()

    # Update training config
    train_config['epochs'] = epochs
    train_config['path_to_save_model'] = path_to_save_model
    train_config['epoch_to_save_model'] = epoch_to_save_model

    # Update training config (wandb)
    train_config['project_name'] = project_name
    train_config['name_training_run'] = name_training_run
    train_config['model_artifact_name'] = model_artifact_name

    # If the model has also a classifier add the information to training config
    train_config['measure_metrics_during_training'] = model_config['use_classifier']
    train_config['use_classifier'] = model_config['use_classifier']

    train_dataset=ds_time.EEG_Dataset_list(combo[0], train_label, channels_to_set) #combinations1[0][0] is a list of array corresponding to the train data 
    validation_dataset=ds_time.EEG_Dataset(combo[1], validation_label, channels_to_set) #combinations1[0][1] is an array corresponding to the validation array

    # Train the model
    model = wt.train_wandb_V2_TUAR('hvEEGNet_shallow', train_config, model_config, train_dataset, validation_dataset, number_of_trials)
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
