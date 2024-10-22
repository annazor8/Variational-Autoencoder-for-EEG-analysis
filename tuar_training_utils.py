from collections import defaultdict
from library.analysis import dtw_analysis
from typing import Dict
import numpy as np
import random
import os
import mne
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.typing import NDArray
from typing import Sequence, Union, List

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

def get_data_TUAR(directory_path:str, start_index : int =0, end_index : int = None):
    """
    Returns a doble level dictionary with {subject : {session : array}} where the array corresponds to the value of the session 
    with trials x 1 x 22 x 1000
    """
    
    channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                       'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                       'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                       'EEG T1-REF', 'EEG T2-REF']
    
    new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'E1', 'E2']
    # List all files in the directory
    all_files = os.listdir(directory_path)
    # Filter out only EDF files
    edf_files = [file for file in all_files if file.endswith('.edf')]

    # data structure Dict[str, Dict[str, NDArray] --> Dict[subj_id, Dict[sess, NDArray]]
    session_data: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.array([])))
    # Process each EDF file
    if end_index == None:
        end_index = len(edf_files) -1

    for file_name in sorted(edf_files)[start_index:end_index]:
        file_path = os.path.join(directory_path, file_name)
        sub_id, session, time = file_name.split(".")[0].split(
            "_")  # split the filname into subject, session and time frame
        raw_mne = mne.io.read_raw_edf(file_path,
                                      preload=False)  # Load the EDF file: NB raw_mne.info['chs'] is the only full of information
        raw_mne.pick_channels(channels_to_set,
                              ordered=True)  # reorders the channels and drop the ones not contained in channels_to_set
        rename_mapping = dict(zip(channels_to_set, new_channel_names))
        raw_mne.rename_channels(rename_mapping)
        raw_mne.resample(250)  # resample to standardize sampling frequency to 250 Hz
        epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=False)  # divide the signal into fixed lenght epoch of 4s with 1 second of overlapping: the overlapping starts from the left side of previous epoch
        del raw_mne
        epoch_data = epochs_mne.get_data(copy=False)  # trasform the raw eeg into a 3d np array
        del epochs_mne
        mean=np.mean(epoch_data)
        std = np.std(epoch_data)
        epoch_data = (epoch_data-mean) / std  # normalization for session
        del mean
        del std
        epoch_data = np.expand_dims(epoch_data, 1)  # number of epochs for that signal x 1 x channels x time samples
        # If session_data[sub_id][session] exists, concatenate
        if session_data[sub_id][session].size > 0:
            new_session = session + '_01'
            session_data[sub_id][new_session] = epoch_data
        else:
            session_data[sub_id][session] = epoch_data

        del epoch_data
    return session_data

def leave_one_session_out(session_data: Dict[str, Dict[str, np.ndarray]]):  # -> np.ndarray, np.ndarray, np.ndarray
    """
    Returns splits of sessions leaving one different session out in each fold
    Return:
        combinations: tuple list each one having two elements: train split and val split for each fold
        test_data: list containing hold-out sessions, used for test
    """
    # a list of the dictionaries [{session: arrays}, {session: arrays}, {session: arrays},...]
    # list of defautdict [{'session': array}]
    list_dict_session = session_data.values()  # the type is dict_values
    all_sessions = []  # initialize a list containing all sessions
    for el in list_dict_session:
        all_sessions.extend(list(el.values()))

    test_size = int(np.ceil(0.2 * len(all_sessions)))
    test_data = all_sessions[0:test_size]
    train_val_data = all_sessions[test_size:]
    # list of tuples containing the train data as the fist element and the validation data as the second element
    combinations = []
    for i in range(0, len(train_val_data), 2): #2 because 20% in validation and 2 sessions are the 20% of the total of 10 sessions
        # do not make shuffle(train_data[i]) because the temporal sequence of the layers in the 3d matrix is important to be preserved
        train_data = train_val_data[:i] + train_val_data[i + 2:]  # concatenate the two lists with the + operator
        val_data = train_val_data[i:i + 2]
        combinations.append(
            (train_data, val_data))  # combinations is a list of tuples (train_data: list, val_data: ndarray)
    return combinations, np.concatenate(test_data)


def leave_one_subject_out(session_data, number_of_trials : int =50):
    # initialize an empty dictionary: Dict[str, Dict[str, NDArray]
    subject_data_dict: Dict[str, list] = defaultdict(lambda: list)
    for subj, value in session_data.items():  # key is the subj, value is a dictionary
        # fixed key we are dealing with a subject
        new_value = list(value.values())  # lista degli array, ciascuno rappresentante una sessione per il soggetto key
        subject_data_dict.update({subj: new_value})

    # Get training config
    subjects = list(subject_data_dict.keys())  # list of subjects
    test_size = int(np.ceil(0.2 * len(subjects)))  # train size is the 20% of the total size

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

    """for el in test_data_complete:
        i = random.randint(0, el.shape[0] - number_of_trials)
        test_data.append(el[i:i + number_of_trials, :, :, :])"""

    train_val_data_complete = []
    for k in range(test_size, len(subjects)):
        if len(list(subject_data_dict[subjects[k]])) < 2:
            train_val_data_complete.extend(subject_data_dict[subjects[k]])
        else:
            j = random.randint(0, len(list(subject_data_dict[subjects[k]])) - 1)
            el = list(subject_data_dict[subjects[k]])[j]
            train_val_data_complete.append(el)
    train_val_data = train_val_data_complete #remove this line if I use the number of trials 
    """train_val_data = []
    for el in train_val_data_complete:
        i = random.randint(0, el.shape[0] - number_of_trials)
        train_val_data.append(el[i:i + number_of_trials, :, :, :])"""
    combinations: list = []
    for i in range(len(train_val_data)):
        train_data: list = train_val_data[:i] + train_val_data[i + 1:]
        val_data: np.ndarray = train_val_data[i]
        combinations.append((train_data, val_data))

    # Get training config
    return combinations, test_data

def plot_ORIGINAL_vs_RECONSTRUCTED(ch_to_plot : str, channels_to_set : list, x_eeg : np.ndarray, x_r_eeg : np.ndarray, trial : int = 3, T : int = 1000, name : str = "signal.png"):
    t = torch.linspace(0, 4, T)
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
    plt.savefig(base_path / name)
    fig.show()

SCALE_FOR_MICROVOLTS= 1e-6

def create_raw_mne(
    eeg: NDArray,
    eeg_ch_list: Sequence[str],
    eeg_srate: float,
    scale_microvolts: bool = True,
    ch_types: Union[str, List[str]] = "eeg",
) -> mne.io.RawArray:
    """
    Create mne object starting from data, eeg_ch_list, eeg_srate,

    Args:
        eeg: eeg multichannel timeseris
        eeg_ch_list: channel labels
        eeg_srate: sampling rate
        scale_microvolts: flag for scaling signals to uVolt (* 1e-6, default mne). Defaults to True.
        ch_types: type of signals, could be either a string (e.g., "eeg") or a list with same len of
            ch_list. Defaults to "eeg".

    Returns:
        mne.io.RawArray: the mne raw object
    """
    info = mne.create_info(
        ch_names=eeg_ch_list, ch_types=ch_types, sfreq=eeg_srate, verbose=False  # type: ignore
    )
    if isinstance(ch_types, list):
        scale = np.array([SCALE_FOR_MICROVOLTS if chtype == "eeg" else 1 for chtype in ch_types])
    else:
        scale = SCALE_FOR_MICROVOLTS
    #info.set_montage(montage, match_alias=False, match_case=False, verbose=False)
    data_ = scale * eeg if scale_microvolts else eeg
    return mne.io.RawArray(data_, info, verbose=False)

def normalize_to_range(x, min_val=-100, max_val=100, alpha=-1, beta=1):
    """
    Normalizza un valore x dall'intervallo [min_val, max_val] all'intervallo [-1, 1]

    Args:
        param x: Il valore da normalizzare.
        param min_val: Il valore minimo dell'intervallo originale. (default -100)
        param max_val: Il valore massimo dell'intervallo originale. (default 100)
        return: Il valore normalizzato nell'intervallo [-1, 1]

    Returns:
        an array of the same shape of the input whose values are rescaled in the min-max range 
    """
    return (beta - alpha) * (x - min_val) / (max_val - min_val) + alpha

def statistics_clean_eeg(x_eeg, x_artifactual):
    """
    calcola media e std usando solo i valori non artefattuali degli EEG
    Args:
        x_eeg: eeg session trials x channels x time samples 
        x_artifactual: binary  3d array of the sami shape of x_eeg that has 0 if the corresponding vale cor that channel and tima sample is clean, 1 if it is artifactual 
    Returns:
        mean : the mean value computed only on the clean egg values 
        std : the std value computed only on the clean egg values
    """
    clean_mask = (x_artifactual == 0)
    clean_values = x_eeg[clean_mask]
    mean=np.mean(clean_values)
    std=np.std(clean_values)
    return mean, std
