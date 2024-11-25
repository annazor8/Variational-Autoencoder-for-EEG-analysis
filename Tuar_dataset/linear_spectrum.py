import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import mne
import mpld3
import matplotlib.pyplot as plt
from scipy.signal import welch
from library.model import hvEEGNet
import torch
from library.config import config_model as cm

def plot_average_test_PSD(ch_names: list, test_data, path_to_model : str, eeg_srate : int, optuput_path_folder : str, notch_filter : bool = False, time_slots : list = [(100980, 102020)]):
    plt.switch_backend('TkAgg')

    #-------------------------------------------------------------------------------------------------------------
    #reconstruction of the trials 
    model_config = cm.get_config_hierarchical_vEEGNet(22, 1000)
    model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
    model.load_state_dict(torch.load(path_to_model, map_location= torch.device('cpu')))

    dataset_reconstructed=[]
    for j in range(test_data.shape[0]):
        x_eeg=test_data[j,:,:,:]
        x_eeg = x_eeg.astype(np.float32)
        x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
        model.eval()
        dataset_reconstructed.append(model.reconstruct(x_eeg)) #ogni elemento è ([1, 1, 22, 1000])

    dataset_reconstructed=np.concatenate(dataset_reconstructed) #trailsx1x22x1000

    #-------------------------------------------------------------------------------------------------------------
    #computation of the spectrum for the orginial trials and the reconstructed trials

    test_data_2d= np.squeeze(test_data[0, :, :]) #arry for the 2d signal CxT where i concatenate the trials of 4 seconds
    f, x_eeg_psd = welch(test_data_2d, fs = 250, nperseg = 500, noverlap=250)

    #power and concatenation of the original trials
    power_original_array=[]
    power_original_array.append(x_eeg_psd)

    for i in range(1, test_data.shape[0]):
        f, spectrum_original= welch(np.squeeze(test_data[i, :, :]), fs = 250, nperseg = 500, noverlap=250)
        power_original_array.append(spectrum_original)
        test_data_2d=np.concatenate((test_data_2d, np.squeeze(test_data[i,:,:])), axis=1)


    test_data_2d_reconstructed = None  # Start with None to identify the first valid trial

    test_data_2d_reconstructed_arr=[]
    power_reconstructed_array=[]

    for i in range(dataset_reconstructed.shape[0]):
        trial = dataset_reconstructed[i, :, :]  # Extract the i-th trial
        if np.isnan(trial).any():
            # Skip the trial if it contains NaNs
            continue
        else:
            if notch_filter==True: #If I want to apply a notch filter in the reconstructed signal
                info = mne.create_info(ch_names=ch_names, sfreq=eeg_srate, ch_types='eeg')
                raw_reconstructed_test = mne.io.RawArray(np.squeeze(trial), info) #I create a raw mne starting from the reconstructed values
                raw_reconstructed_test.filter(l_freq=0, h_freq=50)
                #raw_reconstructed_test.notch_filter(freqs=60, picks='all', method='spectrum_fit') #I apply a notch filter
                test_data_2d_reconstructed=raw_reconstructed_test.get_data()
                f, spectrum_rec=welch(test_data_2d_reconstructed, fs = 250, nperseg = 500, noverlap=250)
                power_reconstructed_array.append(spectrum_rec)
                test_data_2d_reconstructed_arr.append(test_data_2d_reconstructed)

            elif notch_filter==False:
                test_data_2d_reconstructed=np.squeeze(trial)
                f, spectrum_rec=welch(test_data_2d_reconstructed, fs = 250, nperseg = 500, noverlap=250)
                power_reconstructed_array.append(spectrum_rec)
                test_data_2d_reconstructed_arr.append(np.squeeze(trial)) #this is done to avoid NaN values

    #-------------------------------------------------------------------------------------------------------------
    #mean and std values of the original and the reconstructed spectrum
    average_original_spectrum=np.mean(power_original_array, axis=0) #(22, 129)
    std_original_spectrum=np.std(power_original_array, axis=0)
    average_reconstructed_spectrum=np.mean(power_reconstructed_array, axis=0) #(22, 129)
    std_reconstructed_spectrum=np.std(power_reconstructed_array, axis=0)

    #-------------------------------------------------------------------------------------------------------------
    #plot the original SPECTRUM for that channel with the area shaded for the std
    for index_ch, channel in enumerate(ch_names):
        fig_spectrum1=plt.figure(figsize=(20, 16))
        plt.plot(f, average_original_spectrum[index_ch,:], color="black")
        plt.fill_between(f, 
                    average_original_spectrum[index_ch, :] - std_original_spectrum[index_ch, :], 
                    average_original_spectrum[index_ch, :] + std_original_spectrum[index_ch, :], 
                    color='black', alpha=0.2, label="±1 STD (Original)")
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.9)
        plt.xlabel('Frequenza [Hz]',fontweight='bold', fontsize=20)
        plt.ylabel(r'PSD [$\mu$V$^2$/Hz]', fontweight='bold', fontsize=20)
        plt.title('Mean spectrum TRIAL SPECIFIC of the original signal- channel {}'.format(channel),fontweight='bold', fontsize=30 )
       
        plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=10)  # Maggiori dimensioni per tick principali
        plt.tick_params(axis='both', which='minor', labelsize=12, width=1, length=5)   # Tick minori, se presenti
    
        # Imposta i tick label in grassetto
        for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            tick.set_fontsize(16)        # Imposta dimensione del font per i tick
            tick.set_fontweight('bold')  # Imposta il font in grassett

        os.makedirs(optuput_path_folder+'/PSD_no_notch_with_reconstruction', exist_ok=True)
        output_path_html=optuput_path_folder + '/PSD_no_notch_with_reconstruction' + '/original_eeg_TRIAL_SPECIFIC_linear{}.html'.format(channel)
        output_path_png=optuput_path_folder + '/PSD_no_notch_with_reconstruction' + '/original_eeg_TRIAL_SPECIFIC_linear{}.png'.format(channel)
        mpld3.save_html(fig_spectrum1, output_path_html) 
        fig_spectrum1.savefig(output_path_png, format='png')
        plt.close()
    #-------------------------------------------------------------------------------------------------------------
    #plot the reconstructed SPECTRUM for that channel with the area shaded for the std
    for index_ch, channel in enumerate(ch_names):
        fig_spectrum2=plt.figure(figsize=(20, 16))
        plt.plot(f, average_reconstructed_spectrum[index_ch,:], color="red")
        plt.fill_between(f, 
                    average_reconstructed_spectrum[index_ch, :] - std_reconstructed_spectrum[index_ch, :], 
                    average_reconstructed_spectrum[index_ch, :] + std_reconstructed_spectrum[index_ch, :], 
                    color='red', alpha=0.2, label="±1 STD (Original)")
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.9)
        plt.xlabel('Frequency [Hz]', fontweight='bold', fontsize=20)
        plt.ylabel(r'PSD [$\mu$V$^2$/Hz]', fontweight='bold', fontsize=20)
        plt.title('Mean spectrum TRIAL SPECIFIC of the reconstructed signal- channel {}'.format(channel), fontweight='bold', fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=10)  # Maggiori dimensioni per tick principali
        plt.tick_params(axis='both', which='minor', labelsize=12, width=1, length=5)   # Tick minori, se presenti
    
        # Imposta i tick label in grassetto
        for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            tick.set_fontsize(16)        # Imposta dimensione del font per i tick
            tick.set_fontweight('bold')  # Imposta il font in grassett
    
        output_path_html=optuput_path_folder + '/PSD_no_notch_with_reconstruction'+'/reconstructed_eeg_TRIAL_SPECIFIC_linear{}.html'.format(channel)
        output_path_png=optuput_path_folder + '/PSD_no_notch_with_reconstruction'+'/reconstructed_eeg_TRIAL_SPECIFIC_linear{}.png'.format(channel)
        fig_spectrum2.savefig(output_path_png, format='png')
        mpld3.save_html(fig_spectrum2, output_path_html)
        plt.close(fig_spectrum2)
    #-------------------------------------------------------------------------------------------------------------
    #plot the original GLOBAL SPECTRUM for that channel with the area shaded for the std
    global_average_original_spectrum=np.mean(power_original_array, axis=(0, 1))
    global_std_original_spectrum=np.std(power_original_array, axis=(0, 1))

    fig_spectrum3=plt.figure(figsize=(20, 16))
    plt.plot(f, global_average_original_spectrum, color="black")
    plt.fill_between(f, 
                    global_average_original_spectrum - global_std_original_spectrum, 
                    global_average_original_spectrum + global_std_original_spectrum, 
                    color='black', alpha=0.2, label="±1 STD (Original)")
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.9)
    plt.xlabel('Frequency [Hz]', fontweight='bold', fontsize=20)
    plt.ylabel(r'PSD [$\mu$V$^2$/Hz]', fontweight='bold', fontsize=20)
    plt.title('Mean spectrum GLOBAL of the reconstructed signal', fontweight='bold', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=10)  # Maggiori dimensioni per tick principali
    plt.tick_params(axis='both', which='minor', labelsize=12, width=1, length=5)   # Tick minori, se presenti
    
    # Imposta i tick label in grassetto
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontsize(16)        # Imposta dimensione del font per i tick
        tick.set_fontweight('bold')  # Imposta il font in grassett
    output_path_html=optuput_path_folder + '/PSD_no_notch_with_reconstruction'+'/original_GLOBAL_eeg_linear.html'
    output_path_png=optuput_path_folder + '/PSD_no_notch_with_reconstruction'+'/original_GLOBAL_eeg_linear.png'
    fig_spectrum3.savefig(output_path_png, format='png', dpi=300)
    mpld3.save_html(fig_spectrum3, output_path_html)
    plt.close(fig_spectrum3)
    global_average_reconstructed_spectrum=np.mean(power_reconstructed_array, axis=(0, 1))
    global_std_reconstructed_spectrum=np.std(power_reconstructed_array, axis=(0, 1))

    #-------------------------------------------------------------------------------------------------------------
    #plot the reconstructed GLOBAL SPECTRUM for that channel with the area shaded for the std
    fig_spectrum4=plt.figure(figsize=(20, 16))
    plt.plot(f, global_average_reconstructed_spectrum, color="red")
    plt.fill_between(f, 
                    global_average_reconstructed_spectrum - global_std_reconstructed_spectrum, 
                    global_average_reconstructed_spectrum + global_std_reconstructed_spectrum, 
                    color='red', alpha=0.2, label="±1 STD (Original)")
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.9)
    plt.xlabel('Frequency [Hz]', fontweight='bold', fontsize=20)
    plt.ylabel(r'PSD [$\mu$V$^2$/Hz]', fontweight='bold', fontsize=20)
    plt.title('Mean spectrum GLOBAL of the reconstructed signal', fontweight='bold', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=10)  # Maggiori dimensioni per tick principali
    plt.tick_params(axis='both', which='minor', labelsize=12, width=1, length=5)   # Tick minori, se presenti
    
    # Imposta i tick label in grassetto
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontsize(16)        # Imposta dimensione del font per i tick
        tick.set_fontweight('bold')  # Imposta il font in grassett
    output_path_html=optuput_path_folder + '/PSD_no_notch_with_reconstruction'+'/reconstructed_GLOBAL_eeg_linear.html'
    output_path_png=optuput_path_folder + '/PSD_no_notch_with_reconstruction'+'/reconstructed_GLOBAL_eeg_linear.png'
    fig_spectrum4.savefig(output_path_png, format='png', dpi=300)
    mpld3.save_html(fig_spectrum4, output_path_html)
    plt.close(fig_spectrum4)
    #-------------------------------------------------------------------------------------------------------------
    #plot the original trials concatenated in time domain 
    n_rows = 22  # in each line I plot the EEG for a single channel
    n_cols = 1  # only one column
    n_plots = 22

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, 10), sharey=True, sharex=True)
    fig.suptitle('Original EEG', fontsize=20, fontweight='bold')
    for i in range(0, 22):
        print(i)
        # Plot the first channel
        ax[i].plot(test_data_2d[i, :], label="{}".format(ch_names[i]), linewidth=0.5, color='black')
        ax[i].legend()
        ax[i].grid(True)

    fig.text(0.04, 0.5, r"Amplitude [$\mu$V]", va='center', rotation='vertical', fontweight='bold', fontsize=20)
    fig.text(0.5, 0.04, 'Time Samples [s]', ha='center', va='center', fontweight='bold', fontsize=20)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.1) 

    os.makedirs(optuput_path_folder+'/time_domain_rec_WITHOUT_notch', exist_ok=True)
    png_path=optuput_path_folder+'/time_domain_rec_WITHOUT_notch'+'/original_test_eeg_TIME.png'
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()
    #-------------------------------------------------------------------------------------------------------------
    #plot a ZOOMED part of the original trials concatenated in time domain 
    for j in range(len(time_slots)):
        n_rows = 3 # in each line I plot the EEG for a single channel
        n_cols = 1  # only one column
        n_plots = 3
        ch_to_plot=['Fp1', 'Cz', 'P4']
        indx_1=ch_names.index(ch_to_plot[0])
        indx_2=ch_names.index(ch_to_plot[1])
        indx_3=ch_names.index(ch_to_plot[2])
        indx_channels=[indx_1, indx_2, indx_3]
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharey=True, sharex=True)
        fig.suptitle('Original EEG zoomed window', fontweight='bold', fontsize=20)
        for i in range(n_plots):
            # Plot the first channel
            ax[i].plot(test_data_2d[indx_channels[i], :], label="{}".format(ch_to_plot[i]), linewidth=0.5, color='black')
            ax[i].legend()
            ax[i].grid(True)
            ax[i].set_xlim(time_slots[j])
            ax[i].set_ylim((-2.5, 2.5))
    
        fig.text(0.04, 0.5, r"Amplitude [$\mu$V]", va='center', rotation='vertical',fontweight='bold', fontsize=20 )
        fig.text(0.5, 0.04, 'Time Samples [s]', ha='center', va='center', fontweight='bold', fontsize=20)
        plt.tight_layout(rect=[0.05, 0, 1, 0.95])
        plt.subplots_adjust(bottom=0.1) 
        os.makedirs(optuput_path_folder+'/time_domain_rec_WITHOUT_notch_zoom', exist_ok=True)
        png_path=optuput_path_folder+'/time_domain_rec_WITHOUT_notch_zoom'+'/original_test_eeg_TIME_{}.png'.format(j)
        plt.savefig(png_path, format='png', dpi=300)
        plt.close()
    #-------------------------------------------------------------------------------------------------------------
    #plot the reconstructed trials concatenated in time domain 
    dataset_reconstructed=np.concatenate((dataset_reconstructed.squeeze()), axis=1)
    n_rows = test_data_2d.shape[0]  # in each line I plot the EEG for a single channel
    n_cols = 1  # only one column
    n_plots = test_data_2d.shape[0]

    #reconstructed_signal=np.concatenate(test_data_2d_reconstructed_arr, axis=1) #C x T
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharey=True, sharex=True)
    fig.suptitle('Reconstructed EEG', fontweight='bold', fontsize=20)
    for i in range(n_plots):
        # Plot the first channel
        ax[i].plot(dataset_reconstructed[i, :], label="{}".format(ch_names[i]), linewidth=0.5, color='red')
        ax[i].legend()
        ax[i].grid(True)

    fig.text(0.04, 0.5, r"Amplitude [$\mu$V]", va='center', rotation='vertical', fontweight='bold', fontsize=20)
    fig.text(0.5, 0.04, 'Time Samples [s]', ha='center', va='center', fontweight='bold', fontsize=20)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.1) 
    os.makedirs(optuput_path_folder+'/time_domain_rec_WITHOUT_notch', exist_ok=True)
    png_path=optuput_path_folder+'/time_domain_rec_WITHOUT_notch'+'/reconstructed_test_eeg_TIME.png'
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()

    #-------------------------------------------------------------------------------------------------------------
    #plot a ZOOMED part of the reconstructed trials concatenated in time domain 
    for j in range(len(time_slots)):
        n_rows = 3  # in each line I plot the EEG for a single channel
        n_cols = 1  # only one column
        n_plots = 3
        ch_to_plot=['Fp1', 'Cz', 'P4']
        indx_1=ch_names.index(ch_to_plot[0])
        indx_2=ch_names.index(ch_to_plot[1])
        indx_3=ch_names.index(ch_to_plot[2])

        indx_channels=[indx_1, indx_2, indx_3]

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharey=True, sharex=True)
        fig.suptitle('Reconstructed EEG zoomed window', fontweight='bold', fontsize=20)
        for i in range(n_plots):
            # Plot the first channel
            ax[i].plot(dataset_reconstructed[indx_channels[i], :], label="{}".format(ch_to_plot[i]), linewidth=0.5, color='red')
            ax[i].legend()
            ax[i].grid(True)
            ax[i].set_xlim(time_slots[j])
            ax[i].set_ylim((-2.5, 2.5))

        fig.text(0.04, 0.5, r"Amplitude [$\mu$V]", va='center', rotation='vertical', fontweight='bold', fontsize=20)
        fig.text(0.5, 0.04, 'Time Samples [s]', ha='center', va='center', fontweight='bold', fontsize=20)
        plt.tight_layout(rect=[0.05, 0, 1, 0.95])
        plt.subplots_adjust(bottom=0.1) 
        os.makedirs(optuput_path_folder+'/time_domain_rec_WITHOUT_notch_zoom', exist_ok=True)
        png_path=optuput_path_folder+'/time_domain_rec_WITHOUT_notch_zoom'+'/reconstructed_test_eeg_TIME_{}.png'.format(j)
        plt.savefig(png_path, format='png', dpi=300)
        plt.close()

#train_shuffle_session='_jrj2'
#train_session="Shuffle{}".format(train_shuffle_session)

train_session='_jrj2'
dataset=np.load("/home/azorzetto/trainShuffle{}/dataset.npz".format(train_session))
train_data = dataset['train_data']
test_data= dataset['test_data']
channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']

new_channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']

model_epoch=85
path_to_model="/home/azorzetto/trainShuffle{}/model_weights_backup_shuffle{}/model_epoch{}.pth".format(train_session, train_session, model_epoch)

#path_to_model='/home/azorzetto/train{}/model_weights_backup{}/model_epoch{}.pth'.format(train_session,train_session, model_epoch)

#TO INSERT THE SHUFFLE NUMBER IN THE SECOND PARAMETER if using shuffle
#path_to_model='/home/azorzetto/train{}/model_weights_backup_shuffle{}/model_epoch{}.pth'.format(train_session, 4, model_epoch)
eeg_srate=250
output_path_folder="/home/azorzetto/trainShuffle{}/PSD_TRAIN_epoch_{}".format(train_session, model_epoch)

plot_average_test_PSD(ch_names=new_channel_names, test_data=train_data, path_to_model=path_to_model, eeg_srate=eeg_srate, optuput_path_folder= output_path_folder, notch_filter=False, time_slots=[(100980, 102020), (500, 1550), (13000, 14000)])
