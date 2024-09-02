import numpy as np
from tuar_training_utils import create_raw_mne
import mne
import mpld3

dataset = np.load('/home/azorzetto/trainShuffle_jrj/dataset.npz')
train_data = dataset['train_data']
test_data= dataset['test_data']
ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
dataset_reconstructed = np.load('/home/azorzetto/trainShuffle_jrj/reconstructed_eeg.npz') ['x_r_eeg']
test_data_2d= np.squeeze(test_data[0, :, :])
for i in range(1, test_data.shape[0]):
    test_data_2d=np.concatenate((test_data_2d, np.squeeze(test_data[i,:,:])), axis=1)

test_data_2d_reconstructed= np.squeeze(dataset_reconstructed[0, :, :])
for i in range(1, dataset_reconstructed.shape[0]):
    test_data_2d_reconstructed=np.concatenate((test_data_2d_reconstructed, np.squeeze(dataset_reconstructed[i,:,:])), axis=1)
eeg_srate=250

info = mne.create_info(ch_names=ch_names, sfreq=eeg_srate, ch_types='eeg')
raw_original_test = mne.io.RawArray(test_data_2d, info)
raw_reconstructed_test= mne.io.RawArray(test_data_2d_reconstructed, info)

spectrum_original=raw_original_test.compute_psd(fmax=70)
fig_spectrum1= spectrum_original.plot(picks="data", exclude="bads", amplitude=False)
mpld3.save_html(fig_spectrum1, "/home/azorzetto/PSD_plot/original_eeg.html") 
fig_spectrum1.savefig("/home/azorzetto/PSD_plot/original_eeg.png", format='png')
spectrum_reconstructed=raw_reconstructed_test.compute_psd(fmax=70)
fig_spectrum2= spectrum_reconstructed.plot(picks="data", exclude="bads", amplitude=False)
fig_spectrum2.savefig("/home/azorzetto/PSD_plot/reconstructed_eeg.png", format='png')
mpld3.save_html(fig_spectrum2, "/home/azorzetto/PSD_plot/reconstructed_eeg.html")    
print()
