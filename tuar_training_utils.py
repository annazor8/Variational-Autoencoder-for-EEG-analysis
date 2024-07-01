from collections import defaultdict
from library.analysis import dtw_analysis
from typing import Dict, List, Tuple
import numpy as np
import random
import os
import mne


def reconstruction_metrics(x_eeg, x_r_eeg, device):
    recon_error_avChannelsF_avTSF = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device,
                                                                                        average_channels=False,
                                                                                        average_time_samples=False)
    recon_error_avChannelsF_avTST = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device,
                                                                                        average_channels=False,
                                                                                        average_time_samples=True)
    recon_error_avChannelsT_avTSF = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device,
                                                                                        average_channels=True,
                                                                                        average_time_samples=False)
    recon_error_avChannelsT_avTST = dtw_analysis.compute_recon_error_between_two_tensor(x_eeg, x_r_eeg, device,
                                                                                        average_channels=True,
                                                                                        average_time_samples=True)
    return recon_error_avChannelsF_avTSF, recon_error_avChannelsF_avTST, recon_error_avChannelsT_avTSF, recon_error_avChannelsT_avTST

def get_data_TUAR(directory_path:str)
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
    return session_data

def leave_one_session_out(session_data: Dict[str, Dict[str, np.ndarray]], global_min, global_max, new_min=-100, new_max=100):  # -> np.ndarray, np.ndarray, np.ndarray
    """
    Returns splits of sessions leaving one different session out in each fold
    Return:
        combinations: tuple list each one having two elements: train split and val split for each fold
        test_data: list containing hold-out sessions, used for test
    """
    # a list of the dictionaries [{session: arrays}, {session: arrays}, {session: arrays},...]
    # list of defautdict [{'session': array}]
    list_dict_session = session_data.values()  # the type is dict_values
    all_sessions_complete = []  # initialize a list containing all sessions
    for el in list_dict_session:
        all_sessions_complete.extend(list(el.values()))
    all_sessions = all_sessions_complete
    #for el in all_sessions_complete:
    #    el=((el - global_min) / ( global_max- global_min)) * (new_max - new_min) + new_min
    #    all_sessions.append(el)

    test_size = int(np.ceil(0.2 * len(all_sessions)))
    test_data = all_sessions[0:test_size]
    train_val_data = all_sessions[test_size:]
    # list of tuples containing the train data as the fist element and the validation data as the second element
    combinations = []
    for i in range(0, len(train_val_data), 4):
        # do not make shuffle(train_data[i]) because the temporal sequence of the layers in the 3d matrix is important to be preserved
        train_data = train_val_data[:i] + train_val_data[i + 4:]  # concatenate the two lists with the + operator
        val_data = train_val_data[i:i + 4]
        combinations.append(
            (train_data, val_data))  # combinations is a list of tuples (train_data: list, val_data: ndarray)
    return combinations, test_data


def leave_one_subject_out(session_data, global_min, global_max, number_of_trials: int = 64, shuffle: bool = True):
    # initialize an empty dictionary: Dict[str, Dict[str, NDArray]
    subject_data_dict: Dict[str, list] = defaultdict(lambda: list)
    for subj, value in session_data.items():  # key is the subj, value is a dictionary
        # fixed key we are dealing with a subject
        new_value = list(value.values())  # lista degli array, ciascuno rappresentante una sessione per il soggetto key
        subject_data_dict.update({subj: new_value})

    # Get training config
    subjects = list(subject_data_dict.keys())  # list of subjects
    test_size = int(np.ceil(0.2 * len(subjects)))  # train size is the 20% of the total size

    if shuffle == True:
        random.shuffle(subjects)  # metto un controllo sullo shuffle dei soggetti

    test_data_complete = []  # prendo una sessione per ogni soggetto che uso nel test set
    for i in range(test_size):
        epoch_list = list(subject_data_dict[subjects[i]])
        if len(epoch_list) < 2:  # se ho solo una sessione per quel soggetto uso quella
            test_data_complete.extend(subject_data_dict[subjects[i]])
        else:
            j = random.randint(0, len(list(
                subject_data_dict[subjects[i]])) - 1)  # se ho più sessioni per quel soggetto ne scelgo una a caso
            el = list(subject_data_dict[subjects[i]])[j]
            test_data_complete.append(el)
    test_data = []
    for el in test_data_complete:
        i = random.randint(0, el.shape[0] - number_of_trials)
        test_data.append(el[i:i + number_of_trials, :, :, :])

    train_val_data_complete = []
    for k in range(test_size, len(subjects)):
        if len(list(subject_data_dict[subjects[k]])) < 2:
            train_val_data_complete.extend(subject_data_dict[subjects[k]])
        else:
            j = random.randint(0, len(list(subject_data_dict[subjects[k]])) - 1)
            el = list(subject_data_dict[subjects[k]])[j]
            train_val_data_complete.append(el)

    train_val_data = []
    for el in train_val_data_complete:
        i = random.randint(0, el.shape[0] - number_of_trials)
        train_val_data.append(el[i:i + number_of_trials, :, :, :])
    combinations: list = []
    for i in range(len(train_val_data)):
        train_data: list = train_val_data[:i] + train_val_data[i + 1:]
        val_data: np.ndarray = train_val_data[i]
        combinations.append((train_data, val_data))

    # Get training config
    return combinations, test_data
