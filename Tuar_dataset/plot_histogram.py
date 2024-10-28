import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library.config import config_model as cm
from library.model import hvEEGNet

def hist_computation(path_to_dataset, bins, path_to_save_hist):
    """
        compute the histogram of the original values contained in the whole dataset and saves it in train{train_session} folder

        Args:
            path_to_dataset: the path to the datset
            bins: the number of bins of the histogram 
            path_to_save_hist: the path to save the figure
    """
    
    all_data=np.load(path_to_dataset)

    train_data=all_data['train_data']
    validation_data=all_data['validation_data']
    test_data=all_data['test_data']

    all_data=np.concatenate([train_data, test_data, validation_data])
    all_data=all_data.flatten()

    plt.figure(figsize=(12, 6))
    plt.hist(all_data, bins=bins,range=(-50, 50), color='blue',density=True, alpha=0.7)
    plt.title('Histogram of EEG Values Across All Files in log scale',fontweight='bold')
    plt.xlabel('EEG Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True)
    path_fig=path_to_save_hist+'hist_log_scale.png'
    plt.savefig(path_fig)

    plt.figure(figsize=(12, 6))
    plt.hist(all_data, bins=bins,range=(-50, 50), color='blue',density=True, alpha=0.7)
    plt.title('Histogram of EEG Values Across All Files in linear scale}', fontweight='bold')
    plt.xlabel('EEG Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    path_fig=path_to_save_hist+'hist_linear_scale.png'
    plt.savefig(path_fig)

def hist_computation_reconstructed(path_to_dataset, bins, path_to_model, output_path_folder):
    """
        compute the histogram of the reconstructed values contained in the train and test dataset separately and saves in the train{train_session}/RECONSTRUCTED_HIST folder 

        Args:
            path_to_dataset: the path to load the dataset
            bins: the number of bins of the histogram
            path_to_model: path to the model used to calculate the reconstruction
            output_path_folder: the folder to the output path 
    """

    all_data=np.load(path_to_dataset)
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
    fig.suptitle('Reconstructed EEG session in linear scale', fontsize = 20, weight='bold')
    fig.tight_layout()

    os.makedirs(output_path_folder+'/RECONSTRUCTED_HIST', exist_ok=True)
    png_path=output_path_folder+'/RECONSTRUCTED_HIST'+'/hist_linear_scale.png'
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
    fig.suptitle('Reconstructed EEG session in log scale', fontsize = 20,weight='bold')
    fig.tight_layout()

    os.makedirs(output_path_folder+'/RECONSTRUCTED_HIST', exist_ok=True)
    png_path=output_path_folder+'/RECONSTRUCTED_HIST'+'/hist_log_scale.png'
    plt.savefig(png_path, format='png')
    plt.close()

path_to_save_hist='/home/azorzetto/train16/'
path_to_dataset='/home/azorzetto/train16/dataset.npz'
hist_computation(path_to_dataset= path_to_dataset, bins=150, path_to_save_hist=path_to_save_hist)