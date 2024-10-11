"Questa era una prova per vedere come cambiava il plot dopo che avevo modificato il reference al canale A1"
import numpy as np
import mne
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import mpld3
import torch
channels_to_set = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                       'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T4-REF',
                       'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                       'EEG T1-REF', 'EEG T2-REF']
raw_mne = mne.io.read_raw_edf("/home/azorzetto/dataset/01_tcp_ar/aaaaabbn_s009_t000.edf",
                                    preload=True)  # Load the EDF file: NB raw_mne.info['chs'] is the only full of information
#raw_mne.plot()

raw_mne.pick_channels(channels_to_set,
                        ordered=True)  # reorders the channels and drop the ones not contained in channels_to_set

raw_mne.set_eeg_reference('average')

raw_mne.filter(l_freq=0.5, h_freq=70)
raw_mne.notch_filter(freqs=60, picks='all', method='spectrum_fit')
raw_mne.resample(250)  # resample to standardize sampling frequency to 250 Hz
raw_mne.set_eeg_reference(ref_channels=['EEG A1-REF'])

epochs_mne = mne.make_fixed_length_epochs(raw_mne, duration=4, preload=False, reject_by_annotation=False)  # divide the signal into fixed lenght epoch of 4s
del raw_mne
epoch_data = epochs_mne.get_data(copy=False)  # trasform the raw eeg into a 3d np array
epoch_data=epoch_data*1e6

mean=np.mean(epoch_data)
std = np.std(epoch_data)
epoch_data = (epoch_data-mean) / std  # normalization for session
t = torch.linspace(0, 4, 1000).numpy()

for i in range(40, epoch_data.shape[0]):
    for idx_ch, ch in enumerate(channels_to_set):
            # Plot the original and reconstructed signal
            fig, ax = plt.subplots(figsize = (20, 12))  # Adjust figsize for better visibility
            ax.plot(t, epoch_data[i].squeeze()[idx_ch].squeeze(), label=f'Original EEG - channel {ch}', color='black', linewidth=1)

            ax.legend(loc='upper right')
            ax.set_xlim([0, 2])
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(r"Amplitude [$\mu$V]")
            ax.grid(True)
            # Adjust layout
            plt.tight_layout()
            output_dir = Path(f'/home/azorzetto/prova1_A1_REF')
            output_dir.mkdir(parents=True, exist_ok=True)
            png_path = output_dir / f'trial_{i}_channel_{ch}.png'
            html_path = output_dir / f'trial_{i}_channel_{ch}.html'
            plt.savefig(png_path, dpi=300)
            mpld3.save_html(plt.gcf(), str(html_path))
            plt.close()