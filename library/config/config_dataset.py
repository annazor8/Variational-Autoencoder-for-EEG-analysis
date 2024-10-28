"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the config related to dataset download and preprocess
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_moabb_dataset_config(subjects_list = [1,2,3,4,5,6,7,8,9], use_stft_representation = False):
    """
    Configuration for the download of dataset 2a through moabb
    """

    dataset_config = dict(
        # Frequency filtering settings
        filter_data = False,    # If True filter the data
        filter_type = 0,        # 0 Bandpass, 1 lowpass, 2 highpass (used only if filter_data is True)
        fmin = 0.5,             # Used in bandpass and highpass (used only if filter_data is True)
        fmax = 50,              # Used in bandpass and lowpass (used only if filter_data is True)
        filter_method = 'iir',  # Filter settings (used only if filter_data is True)
        iir_params = dict(ftype = 'cheby2', order = 20, rs = 30), # Filter settings (used only if filter_data is True and filter_method is iir)

        # Resampling settings
        resample_data = False,  # If true resample the data
        resample_freq = 128,    # New sampling frequency (used only if resample_data is True)

        # Trial segmentation
        trial_start = 2,    # Time (in seconds) when the trial starts. Keep this value
        trial_end = 6,      # Time (in seconds) when the trial end. Keep this value
        use_moabb_segmentation = False,

        # Split in train/test/validation
        seed_split = 42,                        # Seed for the random function used for split the dataset. Used for reproducibility
        percentage_split_train_test = -1,       # For ALL the data select the percentage for training and for test. -1 means to use the original division in train and test data
        percentage_split_train_validation = 0.9, # For ONLY the training data select the percentage for train and for validation

        # Other
        n_classes = 4,                  # Number of labels. For datset 2a is equal to 4. (IGNORE)
        subjects_list = subjects_list,  # List of the subjects of dataset 2a to download

        # Stft settings (IGNORE)(NOT USED)
        use_stft_representation = use_stft_representation,
        channels_list = ['C3', 'Cz', 'C4'], # List of channel to transform with STFT. Ignore.
        normalize = 0, # If different to 0 normalize the data during the dataset creation. Ignore and kept to 0
        train_trials_to_keep = None, # Boolean list with the same length of the training set, BEFORE THE DIVISION with training and validation, that indicate with trial kept for the training.
        # normalization_type = 1, # 0 = no normalization, 1 = ERS normalization (NOT IMPLEMENTED)
    )
    
    if dataset_config['use_stft_representation']: 
        dataset_config['stft_parameters'] = get_config_stft()
    else:
        del dataset_config['channels_list']

    return dataset_config

def get_config_stft():
    config = dict(
        sampling_freq = 250,
        nperseg = 50,
        noverlap = 40,
        # window = ('gaussian', 1),
        window = 'hann',
    )

    return config

def get_artifact_dataset_config(type_dataset, folder_to_save = 'v2'):
    dataset_config = dict(
        # Version
        type_dataset = type_dataset,
        # Frequency filtering settings
        filter_data = True,
        fmin = 0,
        fmax = 125,
        # Resampling settings
        resample_data = True,
        resample_freq = 256,
        # Other
        folder_to_save = folder_to_save
    )

    return dataset_config

def get_preprocessing_config():
    dataset_config = dict(
        channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                       'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF','EEG T3-REF', 'EEG T4-REF',
                       'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                       'EEG T1-REF', 'EEG T2-REF'], 
        split_mapping={'FP1': 'Fp1', 'FP2':'Fp2', 'F3':'F3', 'F4':'F4', 'C3':'C3', 'C4':'C4', 'P3':'P3', 'P4':'P4', 'O1':'O1', 'O2':'O2', 'F7':'F7', 'T3':'T3', 'T4':'T4', 'T5':'T5', 'T6':'T6', 'A1':'A1', 'A2':'A2', 'FZ':'Fz', 'CZ':'Cz', 'PZ':'Pz', 'T1':'T1', 'T2':'T2'},
        new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2'],
        start_index=0,
        end_index=None,
        notch_filter=True,
        sfreq = 250,
        band_pass_filter=True,
        notch_filter=True,
        l_freq=0.5,
        h_freq=100,
        monopolar_reference=False,
        ref_channel=['EEG A1-REF'],
        z_score_session_wise=True,
        min_max_normalization=False,
        min_val=-50,
        max_val=50,
        z_score_cleand=False,
        only_cleand_trials=False #to retain only the clean trials and discard the artifactual trials
    )

    return dataset_config
