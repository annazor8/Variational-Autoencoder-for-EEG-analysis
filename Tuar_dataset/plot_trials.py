import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import torch 
from pathlib import Path
import mpld3
import mne
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library.config import config_model as cm
from library.model import hvEEGNet
from utils import normalize_to_range

def plot_trials(path_to_dataset, path_to_save_img, path_to_model, channel_names: list, type_data : str, from_trial: int, post_processing : bool = False, original_signal_normalize : bool = False, reconstructed_signal_normalize : bool = False, min : int =-1 , max : int =1):
    """

    Plots the trials of the train or the test set 

    Args:
        train_session: is the number of the train session into consideration
        model_epoch: is the epoch number at which the model is loaded 
        type_data: define the selection of train or test data
        post_processing: is a bool variable that define if a band pass or a notch filter is applied on the RECONSTRUCTED variable 
        original_signal_normalize: is a bool variable that define if a min-max normalization is applied on the original signal
        reconstructed_signal_normalize: is a bool variable that define if a min-max normalization is applied on the reconstructed signal
        min: if original_signal_normalize or reconstructed_signal_normalize are set to True, define the min value of the min-max range
        max: if original_signal_normalize or reconstructed_signal_normalize are set to True, define the max value of the min-max range
    """

    #load the test data
    data = np.load(path_to_dataset)
    test_data=data[type_data]

    #load the Reconstruction error with average_channels  and average_time_samples
    model_config = cm.get_config_hierarchical_vEEGNet(22, 1000)
    model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
    model.load_state_dict(torch.load(path_to_model, map_location= torch.device('cpu'), weights_only=True))
    model.eval()

    rec_array=[]
    for i in range(from_trial, test_data.shape[0]):
        x=test_data[i,:,:,:]
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x=torch.unsqueeze(x, 1)
        x_eeg_rec=model.reconstruct(x)
        rec_array.append(x_eeg_rec)
        t = torch.linspace(0, 4, 1000).numpy()  # Convert to NumPy array if needed
        trial_eeg=np.squeeze(x)
        x_eeg_rec=np.squeeze(x_eeg_rec)

        #----------------------if you want to normalize the data in a min-max range
        if original_signal_normalize== True:
            trial_eeg=normalize_to_range(trial_eeg, min, max)
        if reconstructed_signal_normalize==True:
            x_eeg_rec=normalize_to_range(x_eeg_rec, min, max)

        #----------------------if you want to apply a bandpass filter or a notch filter in post-processing
        if post_processing == True:
            info = mne.create_info(ch_names=channel_names, sfreq=250, ch_types='eeg')
            raw_reconstructed_test = mne.io.RawArray(x_eeg_rec, info) #I create a raw mne starting from the reconstructed values
            raw_reconstructed_test.filter(l_freq=0.5, h_freq=60)
            #raw_reconstructed_test.notch_filter(freqs=60, picks='all', method='spectrum_fit') #I apply a notch filter
            x_eeg_rec=raw_reconstructed_test.get_data()

        font_properties = FontProperties(weight='bold', size=16)
        for idx_ch, ch in enumerate(channel_names):
            # Plot the original and reconstructed signal
            fig, ax = plt.subplots(figsize = (18, 10))  # Adjust figsize for better visibility
            ax.plot(t, trial_eeg.squeeze()[idx_ch].squeeze(), label=f'Original EEG - channel {ch}', color='black', linewidth=1)
            ax.plot(t, (x_eeg_rec.squeeze()[idx_ch].squeeze()), label=f'Reconstructed EEG - channel {ch}', color='red', linewidth=1)

            ax.legend(loc='upper right', prop=font_properties)
            ax.set_xlim([0, 2])
            ax.set_xlabel('Time [s]', fontsize=20, fontweight='bold')
            ax.set_ylabel(r"Amplitude [$\mu$V]", fontsize=20, fontweight='bold')
            ax.grid(True)
            # Adjust layout
            plt.tight_layout()
            plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=10)  # Maggiori dimensioni per tick principali
            plt.tick_params(axis='both', which='minor', labelsize=12, width=1, length=5)   # Tick minori, se presenti
    
        # Imposta i tick label in grassetto
            for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                tick.set_fontsize(16)        # Imposta dimensione del font per i tick
                tick.set_fontweight('bold')  # Imposta il font in grassett
            output_dir = Path(path_to_save_img)
            output_dir.mkdir(parents=True, exist_ok=True)
            png_path = output_dir / f'trial_{i}_channel_{ch}.png'
            html_path = output_dir / f'trial_{i}_channel_{ch}.html'
            plt.savefig(png_path, dpi=300)
            mpld3.save_html(plt.gcf(), str(html_path))
            plt.close()

#train_session="Shuffle9"
#train_session='_jrj2'
channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']

train_session=13
model_epoch=66

from_trial=0
path_to_dataset=f'/home/azorzetto/train{train_session}/dataset.npz'
type_data= "train_data" #or 'test_data'

path_to_model=f'/home/azorzetto/train{train_session}/model_weights_backup{train_session}/model_epoch{model_epoch}.pth'
path_to_save_img=f'/home/azorzetto/train{train_session}/img_train_epoch_{model_epoch}'

post_processing=False
original_signal_normalize=False
reconstructed_signal_normalize=False

plot_trials(path_to_dataset=path_to_dataset, path_to_save_img=path_to_save_img, path_to_model=path_to_model, channel_names= channel_names, type_data=type_data, from_trial=from_trial, post_processing= False,  original_signal_normalize= False, reconstructed_signal_normalize= False)
