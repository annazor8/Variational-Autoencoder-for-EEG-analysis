import numpy as np
from tuar_training_utils import create_raw_mne
import mne
import mpld3
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import stft, welch
from library.config import config_model as cm
from library.model import hvEEGNet
import torch 
channel='Fp1'
dataset = np.load('/home/azorzetto/train8/dataset.npz')
train_data = dataset['train_data']
test_data= dataset['test_data']
ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
eeg_srate=250
info = mne.create_info(ch_names=ch_names, sfreq=eeg_srate, ch_types='eeg')
power_original_array=[]
power_reconstructed_array=[]

model_config = cm.get_config_hierarchical_vEEGNet(22, 1000)
model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
model.load_state_dict(torch.load('/home/azorzetto/train8/model_weights_backup8/model_epoch80.pth', map_location= torch.device('cpu')))

#dataset_reconstructed = np.load('/home/azorzetto/train8/reconstructed_eeg.npz') ['x_r_eeg']
"""if np.isnan(dataset_reconstructed).any():
    print("Data contains NaNs, which may cause PSD calculation issues.")"""

info = mne.create_info(ch_names=ch_names, sfreq=eeg_srate, ch_types='eeg')

dataset_reconstructed=[]
for j in range(test_data.shape[0]):
    x_eeg=test_data[j,:,:,:]
    x_eeg = x_eeg.astype(np.float32)
    x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
    model.eval()
    dataset_reconstructed.append(model.reconstruct(x_eeg))

dataset_reconstructed=np.concatenate(dataset_reconstructed)
test_data_2d= np.squeeze(test_data[1, :, :])
f, x_eeg_psd = welch(test_data_2d, fs = 250, nperseg = 256)

power_original_array.append(x_eeg_psd)

for i in range(2, test_data.shape[0]):
    f, spectrum_original= welch(np.squeeze(test_data[i, :, :]), fs = 250, nperseg = 256)
    power_original_array.append(spectrum_original)
    test_data_2d=np.concatenate((test_data_2d, np.squeeze(test_data[i,:,:])), axis=1)


test_data_2d_reconstructed = None  # Start with None to identify the first valid trial

test_data_2d_reconstructed_arr=[]
for i in range(dataset_reconstructed.shape[0]):
    trial = dataset_reconstructed[i, :, :]  # Extract the i-th trial
    if np.isnan(trial).any():
        # Skip the trial if it contains NaNs
        continue

    else:
        # Concatenate valid trials along axis 1
        #raw_reconstructed_test = mne.io.RawArray(np.squeeze(trial), info)
        #raw_reconstructed_test.notch_filter(freqs=60, picks='all', method='spectrum_fit')
        #test_data_2d_reconstructed=raw_reconstructed_test.get_data()
        test_data_2d_reconstructed=np.squeeze(trial)
        f, spectrum_rec=welch(test_data_2d_reconstructed, fs = 250, nperseg = 256)
        power_reconstructed_array.append(spectrum_rec)
        #test_data_2d_reconstructed_arr.append(test_data_2d_reconstructed)
        test_data_2d_reconstructed_arr.append(np.squeeze(trial))


average_original_spectrum=np.mean(power_original_array, axis=0)
average_reconstructed_spectrum=np.mean(power_reconstructed_array, axis=0)

index_ch=ch_names.index(channel)

fig_spectrum1=plt.figure(figsize=(10, 6))
plt.plot(f, average_original_spectrum[index_ch,:], color="black")
plt.xlabel('Frequenza [Hz]')
plt.ylabel('PSD')
plt.title('mean PSD of the trials of the original signal- channel {}'.format(channel))
mpld3.save_html(fig_spectrum1, "/home/azorzetto/train8/Prova_con_model_eval()/PSD_no_notch_with_reconstruction/original_eeg_linear{}.html".format(channel)) 
fig_spectrum1.savefig("/home/azorzetto/train8/Prova_con_model_eval()/PSD_no_notch_with_reconstruction/original_eeg_linear{}.png".format(channel), format='png')

fig_spectrum2=plt.figure(figsize=(10, 6))
plt.plot(f, average_reconstructed_spectrum[index_ch,:], color="red")
plt.xlabel('Frequenza [Hz]')
plt.ylabel('PSD')
plt.title('mean PSD of the trials of the reconstructed signal- channel {}'.format(channel))
plt.show
fig_spectrum2.savefig("/home/azorzetto/train8/Prova_con_model_eval()/PSD_no_notch_with_reconstruction/reconstructed_eeg_linear{}.png".format(channel), format='png')
mpld3.save_html(fig_spectrum2, "/home/azorzetto/train8/Prova_con_model_eval()/PSD_no_notch_with_reconstruction/reconstructed_eeg_linear{}.html".format(channel))


first_channel_original = test_data_2d[index_ch, :] #18=Cz e 7=P4
first_channel_reconstructed=np.concatenate(test_data_2d_reconstructed_arr, axis=1)[index_ch, :]
# Plot the first channel
plt.figure(figsize=(10, 4))
plt.plot(first_channel_original, label=channel, linewidth=0.5, color='black')
plt.xlim((100980, 102020))
plt.ylim((-2.5,2.5))
plt.xlabel('Time Points')
plt.ylabel('Amplitude')
plt.title('Plot of the {} Channel'.format(channel))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/azorzetto/train8/Prova_con_model_eval()/time_domain_rec_WITHOUT_notch/original_test_eeg_TIME_{}.png".format(channel), format='png')

#first_channel_reconstructed=test_data_2d_reconstructed[index_ch, :]

plt.figure(figsize=(10, 4))
plt.plot(first_channel_reconstructed, label=channel,linewidth=0.5, color='red' )
plt.xlim((100980, 102020))
plt.ylim((-2.5,2.5))
plt.xlabel('Time Points')
plt.ylabel('Amplitude')
plt.title('Plot of the  Reconstructed signal WITH notch- {} Channel'.format(channel))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/azorzetto/train8/Prova_con_model_eval()/time_domain_rec_WITHOUT_notch/reconstructed_test_eeg_TIME_{}.png".format(channel), format='png')
eeg_srate=250
info = mne.create_info(ch_names=ch_names, sfreq=eeg_srate, ch_types='eeg')
raw_original_test = mne.io.RawArray(test_data_2d, info)

"""raw_reconstructed_test= mne.io.RawArray(test_data_2d_reconstructed, info)
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
plt.savefig("/home/azorzetto/train8/reconstructed_test_eeg_TIME_P4_notch_filter.png", format='png')"""

"""spectrum_original=raw_original_test.compute_psd(fmax=70)
fig_spectrum1= spectrum_original.plot(picks=['Cz'], exclude="bads",dB=False, xscale='linear', average=False)
mpld3.save_html(fig_spectrum1, "/home/azorzetto/train8/PSD_plot/original_eegCz.html") 
fig_spectrum1.savefig("/home/azorzetto/train8/PSD_plot/original_eegCz.png", format='png')
spectrum_reconstructed=raw_reconstructed_test.compute_psd(fmax=70)
fig_spectrum2= spectrum_reconstructed.plot(picks=['Cz'], exclude="bads", dB=False, xscale='linear', average=False)
fig_spectrum2.savefig("/home/azorzetto/train8/PSD_plot/reconstructed_eegCz.png", format='png')
mpld3.save_html(fig_spectrum2, "/home/azorzetto/train8/PSD_plot/reconstructed_eegCz.html")    
print()"""

