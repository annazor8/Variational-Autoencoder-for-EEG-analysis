from collections import defaultdict
from library.analysis import dtw_analysis
from typing import Dict, List, Tuple
import numpy as np
import random


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
    all_sessions = []
    for el in all_sessions_complete:
        el=((el - global_min) / ( global_max- global_min)) * (new_max - new_min) + new_min
        all_sessions.append(el)

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
