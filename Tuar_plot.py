import mne
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage



#now i'm using the standard_alphabetic name 
def electrodes_name(standard_montage_name : str):
    montage = make_standard_montage(standard_montage_name)
    electrode_names = sorted(montage.ch_names)
    # Print the electrode names alphabetically
    for name in electrode_names:
        print(name)
#/home/azorzetto/train6/dataset/test_data.npy
data = np.load('/home/azorzetto/train6/dataset.npz')
test_data=data['test_data']
directory_path='/home/azorzetto/dataset/01_tcp_ar'

channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                       'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                       'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                       'EEG T1-REF', 'EEG T2-REF']

print(mne.channels.get_builtin_montages(descriptions=True))
electrodes_name("standard_postfixed")

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'P5', 'P6']
# Create a mapping from old names to new names
rename_mapping = dict(zip(channels_to_set, new_channel_names))
    # List all files in the directory
all_files = os.listdir(directory_path)
    # Filter out only EDF files
edf_files = [file for file in all_files if file.endswith('.edf')]



# Loop through each EDF file
for file_name in sorted(os.listdir(directory_path)):
    if file_name.endswith('.edf'):
        file_path = os.path.join(directory_path, file_name)
        
        # Split the filename into subject, session, and time frame
        sub_id, session, time = file_name.split(".")[0].split("_")

        # Load the EDF file
        raw_mne = mne.io.read_raw_edf(file_path, preload=True)  #instance of the Raw class, specifically a subclass of Raw tailored for EDF files, called RawEDF
        raw_mne.plot(scalings='auto', block=True)


        print(raw_mne.info['ch_names'])
        raw_mne.rename_channels(rename_mapping)
        print(raw_mne.info['ch_names'])
        # Pick and reorder channels
        raw_mne.pick_channels(new_channel_names, ordered=True)
        print(raw_mne.info['ch_names'])

        
        # Resample the data to 250 Hz
        raw_mne.resample(250)
     
        montage = make_standard_montage('standard_alphabetic')
        
        # Apply the montage to your raw data
        raw_mne.set_montage(montage)
        raw_mne.plot(scalings='auto', block=True)
       
        # Get the data and sampling frequency
        spectrum=raw_mne.compute_psd(fmax=70)
        fig = spectrum.plot(picks="data", exclude="bads", amplitude=False)
        plt.show()
        fig.legend()
        fig.suptitle(f'EEG Data Visualization {sub_id} {session} {time}', fontsize=20)

        plt.show()

