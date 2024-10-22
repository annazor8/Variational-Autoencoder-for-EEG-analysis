import numpy as np
import matplotlib.pyplot as plt
from library.config import config_model as cm
from library.model import hvEEGNet
import torch
import os

def hist_computation(train_session, bins):
    """
        compute the histogram of the original values contained in the whole dataset and saves it in train{train_session} folder

        Args:
            train_session: the number of the session taken into consideration
            bins: the number of bins of the histogram 
    """
    
    all_data=np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))

    train_data=all_data['train_data']
    validation_data=all_data['validation_data']
    test_data=all_data['test_data']

    all_data=np.concatenate([train_data, test_data, validation_data])
    all_data=all_data.flatten()

    plt.figure(figsize=(12, 6))
    plt.hist(all_data, bins=bins,range=(-200, 200), color='blue',density=True, alpha=0.7)
    plt.title(f'Histogram of EEG Values Across All Files in log scale of train session {train_session}',fontweight='bold')
    plt.xlabel('EEG Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig("/home/azorzetto/train{}/hist_log_scale.png".format(train_session))

    plt.figure(figsize=(12, 6))
    plt.hist(all_data, bins=bins,range=(-200, 200), color='blue',density=True, alpha=0.7)
    plt.title(f'Histogram of EEG Values Across All Files in linear scale of train session {train_session}', fontweight='bold')
    plt.xlabel('EEG Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("/home/azorzetto/train{}/hist_linear_scale.png".format(train_session))

def hist_computation_reconstructed(train_session, bins, path_to_model):
    """
        compute the histogram of the reconstructed values contained in the train and test dataset separately and saves in the train{train_session}/RECONSTRUCTED_HIST folder 

        Args:
            train_session: the number of the session taken into consideration
            bins: the number of bins of the histogram
            path_to_model: path to the model used to calculate the reconstruction
    """

    all_data=np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))
    #all_data=np.load('/home/azorzetto/trainShuffle_jrj/dataset.npz')
    train_data=all_data['train_data']
    test_data=all_data['test_data']

    model_config = cm.get_config_hierarchical_vEEGNet(22, 1000)
    model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
    model.load_state_dict(torch.load(path_to_model, map_location= torch.device('cpu')))

    test_dataset_reconstructed=[]
    train_dataset_reconstructed=[]
    for j in range(test_data.shape[0]):
        x_eeg=test_data[j,:,:,:]
        x_eeg = x_eeg.astype(np.float32)
        x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
        model.eval()
        test_dataset_reconstructed.append(model.reconstruct(x_eeg))

    for j in range(train_data.shape[0]):
        x_eeg=train_data[j,:,:,:]
        x_eeg = x_eeg.astype(np.float32)
        x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
        model.eval()
        train_dataset_reconstructed.append(model.reconstruct(x_eeg))


    n_rows = 1  # in each line I plot the EEG for a single channel
    n_cols = 2  # only one column

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, 10))
    # Plot the first channel
    ax[0].hist(np.concatenate(train_dataset_reconstructed).flatten(), range=(-100, 100), bins=bins,density=True, color='blue', alpha=0.7 )
    ax[0].set_title('Train data', fontsize =20, weight='bold')
    ax[0].grid(True)
    ax[1].hist(np.concatenate(test_dataset_reconstructed).flatten(), range=(-100, 100), bins=bins,density=True, color='blue', alpha=0.7)
    ax[1].set_title('Test data', fontsize =20, weight='bold')
    ax[1].grid(True)

    for axs in ax:
        axs.set_xlabel('EEG reconstructed values', fontsize =16)
        axs.set_ylabel('frequency', fontsize = 16)
    fig.suptitle('Reconstructed EEG session {} in linear scale'.format(train_session), fontsize = 20, weight='bold')
    fig.tight_layout()

    optuput_path_folder='/home/azorzetto/train{}'.format(train_session)
    os.makedirs(optuput_path_folder+'/RECONSTRUCTED_HIST', exist_ok=True)
    png_path=optuput_path_folder+'/RECONSTRUCTED_HIST'+'/hist_linear_scale.png'
    plt.savefig(png_path, format='png')
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(18, 10))
    # Plot the first channel
    ax[0].hist(np.concatenate(train_dataset_reconstructed).flatten(), range=(-100, 100), bins=bins,density=True, color='blue', alpha=0.7 )
    ax[0].set_title('Train data', fontsize =20, weight='bold')
    ax[0].grid(True)
    ax[0].set_yscale('log')
    ax[1].hist(np.concatenate(test_dataset_reconstructed).flatten(), range=(-100, 100), bins=bins,density=True, color='blue', alpha=0.7)
    ax[1].set_title('Test data', fontsize =20, weight='bold')
    ax[1].set_yscale('log')
    ax[1].grid(True)

    for axs in ax:
        axs.set_xlabel('EEG reconstructed values', fontsize =16)
        axs.set_ylabel('frequency', fontsize = 16)
    fig.suptitle('Reconstructed EEG session {} in log scale'.format(train_session), fontsize = 20,weight='bold')
    fig.tight_layout()

    optuput_path_folder='/home/azorzetto/train{}'.format(train_session)
    os.makedirs(optuput_path_folder+'/RECONSTRUCTED_HIST', exist_ok=True)
    png_path=optuput_path_folder+'/RECONSTRUCTED_HIST'+'/hist_log_scale.png'
    plt.savefig(png_path, format='png')
    plt.close()

train_session=14
hist_computation(train_session= train_session, bins=150)