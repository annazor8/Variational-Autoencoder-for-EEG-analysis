import numpy as np
from utils import create_raw_mne
import mne
import mpld3
import matplotlib.pyplot as plt

channel=['Cz']
dataset = np.load('/home/azorzetto/train8/dataset.npz')
train_data = dataset['train_data']
test_data= dataset['test_data']
ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
eeg_srate=250
info = mne.create_info(ch_names=ch_names, sfreq=eeg_srate, ch_types='eeg')
power_original_array=[]
power_reconstructed_array=[]

dataset_reconstructed = np.load('/home/azorzetto/train8/reconstructed_eeg.npz') ['x_r_eeg']
if np.isnan(dataset_reconstructed).any():
    print("Data contains NaNs, which may cause PSD calculation issues.")

test_data_2d= np.squeeze(test_data[1, :, :])
raw_original_test = mne.io.RawArray(test_data_2d, info)
spectrum_original=raw_original_test.compute_psd(picks=channel, fmin=1, fmax=40, )
power_original_array.append(np.array(spectrum_original.get_data()))

for i in range(2, test_data.shape[0]):
    raw_original_test = mne.io.RawArray(np.squeeze(test_data[i,:,:]), info)
    spectrum_original=raw_original_test.compute_psd(picks=channel, fmin=1, fmax=40)
    power_original_array.append(np.array(spectrum_original.get_data()))
    test_data_2d=np.concatenate((test_data_2d, np.squeeze(test_data[i,:,:])), axis=1)


test_data_2d_reconstructed = None  # Start with None to identify the first valid trial

for i in range(dataset_reconstructed.shape[0]):
    trial = dataset_reconstructed[i, :, :]  # Extract the i-th trial

    if np.isnan(trial).any():
        # Skip the trial if it contains NaNs
        continue

    if test_data_2d_reconstructed is None:
        # Initialize with the first valid trial
        test_data_2d_reconstructed = np.squeeze(trial)
        raw_reconstructed_test= mne.io.RawArray(test_data_2d_reconstructed, info)
        spectrum_rec=raw_reconstructed_test.compute_psd(picks=channel, fmin=1, fmax=40)
        power_reconstructed_array.append(np.array(spectrum_rec.get_data()))
    else:
        # Concatenate valid trials along axis 1
        test_data_2d_reconstructed = np.concatenate((test_data_2d_reconstructed, np.squeeze(trial)), axis=1)
        raw_reconstructed_test= mne.io.RawArray(np.squeeze(trial), info)
        spectrum_rec=raw_reconstructed_test.compute_psd(picks=channel, fmin=1, fmax=40)
        power_reconstructed_array.append(np.array(spectrum_rec.get_data()))

info = mne.create_info(ch_names=channel, ch_types = ['eeg'], sfreq=250)
average_original_spectrum= mne.io.RawArray(np.mean(power_original_array, axis=0), info)
average_reconstructed_spectrum=mne.io.RawArray(np.mean(power_reconstructed_array, axis=0), info)

average_original_spectrum = average_original_spectrum.compute_psd(picks=channel, fmin=1, fmax=40)
freqs = average_original_spectrum.freqs
psd_values = average_original_spectrum.get_data()
plt.figure()
plt.plot(freqs, psd_values.T)  # Trasponi psd_values per allinearlo con freqs
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Densità Spettrale di Potenza (µV²/Hz)')
plt.title('Spettro Lineare')
plt.show()
average_reconstructed_spectrum = average_reconstructed_spectrum.compute_psd(picks=channel, fmin=1, fmax=40)
fig_spectrum1= average_original_spectrum.plot(picks=channel, exclude="bads",dB=False, xscale='linear', average=False,spatial_colors=False, color='black')
mpld3.save_html(fig_spectrum1, "/home/azorzetto/train8/PSD_plot/original_eeg{}.html".format(channel[0])) 
fig_spectrum1.savefig("/home/azorzetto/train8/PSD_plot/original_eeg{}.png".format(channel[0]), format='png')

fig_spectrum2= average_reconstructed_spectrum.plot(picks=channel, exclude="bads", dB=False, xscale='linear', average=False, spatial_colors=False, color='red')
fig_spectrum2.savefig("/home/azorzetto/train8/PSD_plot/reconstructed_eeg{}.png".format(channel[0]), format='png')
mpld3.save_html(fig_spectrum2, "/home/azorzetto/train8/PSD_plot/reconstructed_eeg{}.html".format(channel[0]))


first_channel_original = test_data_2d[7, :] #18=Cz e 7=P4

# Plot the first channel
plt.figure(figsize=(10, 4))
plt.plot(first_channel_original, label='P4', linewidth=0.5, color='black')
plt.xlim((100980, 102020))
plt.ylim((-2.5,2.5))
plt.xlabel('Time Points')
plt.ylabel('Amplitude')
plt.title('Plot of the P4 Channel')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/azorzetto/train8/original_test_eeg_TIME_P4.png", format='png')

first_channel_reconstructed=test_data_2d_reconstructed[7, :]

plt.figure(figsize=(10, 4))
plt.plot(first_channel_reconstructed, label='P4',linewidth=0.5, color='red' )
plt.xlim((100980, 102020))
plt.ylim((-2.5,2.5))
plt.xlabel('Time Points')
plt.ylabel('Amplitude')
plt.title('Plot of the P4 Channel')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/azorzetto/train8/reconstructed_test_eeg_TIME_P4.png", format='png')
eeg_srate=250
info = mne.create_info(ch_names=ch_names, sfreq=eeg_srate, ch_types='eeg')
raw_original_test = mne.io.RawArray(test_data_2d, info)

raw_reconstructed_test= mne.io.RawArray(test_data_2d_reconstructed, info)
raw_reconstructed_test.notch_filter(freqs=60, picks='all', method='spectrum_fit')
raw_reconstructed_test_array=raw_reconstructed_test.get_data()
filtered_reconstructed=raw_reconstructed_test_array[7, :]

plt.figure(figsize=(10, 4))
plt.plot(filtered_reconstructed, label='P4',linewidth=0.5, color='red' )
plt.xlim((100980, 102020))
plt.ylim((-2.5,2.5))
plt.xlabel('Time Points')
plt.ylabel('Amplitude')
plt.title('Plot of the P4 Channel with notch filter applied')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/azorzetto/train8/reconstructed_test_eeg_TIME_P4_notch_filter.png", format='png')

"""spectrum_original=raw_original_test.compute_psd(fmax=70)
fig_spectrum1= spectrum_original.plot(picks=['Cz'], exclude="bads",dB=False, xscale='linear', average=False)
mpld3.save_html(fig_spectrum1, "/home/azorzetto/train8/PSD_plot/original_eegCz.html") 
fig_spectrum1.savefig("/home/azorzetto/train8/PSD_plot/original_eegCz.png", format='png')
spectrum_reconstructed=raw_reconstructed_test.compute_psd(fmax=70)
fig_spectrum2= spectrum_reconstructed.plot(picks=['Cz'], exclude="bads", dB=False, xscale='linear', average=False)
fig_spectrum2.savefig("/home/azorzetto/train8/PSD_plot/reconstructed_eegCz.png", format='png')
mpld3.save_html(fig_spectrum2, "/home/azorzetto/train8/PSD_plot/reconstructed_eegCz.html")    
print()"""

