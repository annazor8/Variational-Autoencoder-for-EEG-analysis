import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from library.analysis.dtw_analysis import compute_recon_error_between_two_tensor, compute_divergence_SDTW_recon_error_between_two_tensor
import torch 
from pathlib import Path
import mpld3
from library.config import config_model as cm
from library.model import hvEEGNet
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter


def reconstruction_ability_over_epochs(train_session : int, writer, epochs : int):

    """
    Computes and plots the ability of the model to reconstruct the signal over the epochs for the three splits (test, train and validation) and plots them in the img_rec_ability_over_epochs folder
    It performs this by computing the mse, sdtw, and divergence dtw between the original and reconstructed signal

    Arg:
        train_session: the number of the session taken into consideration
        writer: tensorboard where the parameters are loaded as soon as they are computed 
        epochs: the number of epochs
    """
    #load the test data
    data = np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))

    train_data=data['train_data']
    validation_data=data['validation_data']
    test_data=data['test_data']

    final_train_mse_array=[]
    final_val_mse_array=[]
    final_test_mse_array=[]
    final_train_sdtw_array=[]
    final_val_sdtw_array=[]
    final_test_sdtw_array=[]
    final_train_divergence_array=[]
    final_validation_divergence_array=[]
    final_test_divergence_array=[]

    #iterate over epochs
    for i in range(epochs):
        model_config = cm.get_config_hierarchical_vEEGNet(22, 1000)
        model = hvEEGNet.hvEEGNet_shallow(model_config)  # new model is instantiated for each iteration of the loop.
        model.load_state_dict(torch.load('/home/azorzetto/train{}/model_weights_backup{}/model_epoch{}.pth'.format(train_session, train_session, i+1), map_location= torch.device('cpu')))
        model.eval()
        train_mse_array=[]
        val_mse_array=[]
        test_mse_array=[]
        train_sdtw_array=[]
        val_sdtw_array=[]
        test_sdtw_array=[]
        train_divergence_array=[]
        validation_divergence_array=[]
        test_divergence_array=[]

        for k in range(64, train_data.shape[0]):
            x_eeg=train_data[k,:,:,:]
            x_eeg = x_eeg.astype(np.float32)
            x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
            #train_data=torch.unsqueeze(train_data, 1)
            train_rec=model.reconstruct(x_eeg)
            if torch.isnan(train_rec).any():
                print("reconstructed trial {} of the train set  is nan".format(k))
                continue
            elif torch.isinf(train_rec).any():
                print("reconstructed trial {} of the train set is inf".format(k))
            else:
                train_mse_array.append(F.mse_loss(x_eeg, train_rec))
                train_sdtw_array.append(compute_recon_error_between_two_tensor(x_eeg, train_rec, average_channels=True, average_time_samples=True))
                train_divergence_array.append(compute_divergence_SDTW_recon_error_between_two_tensor(x_eeg, train_rec, average_channels=True, average_time_samples=True))
        
        final_train_mse=np.mean(train_mse_array)
        final_train_sdtw=np.mean(train_sdtw_array)
        final_train_divergence=np.mean(train_divergence_array)
        final_train_mse_array.append(final_train_mse)
        final_train_sdtw_array.append(final_train_sdtw)
        final_train_divergence_array.append(final_train_divergence)

        for j in range(validation_data.shape[0]):
            x_eeg=validation_data[j,:,:,:]
            x_eeg = x_eeg.astype(np.float32)
            x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
            val_rec=model.reconstruct(x_eeg)
            if torch.isnan(val_rec).any():
                print("reconstructed trial {} of the validation set  is nan".format(j))
                continue
            elif torch.isinf(val_rec).any():
                print("reconstructed trial {} of the validation set is inf".format(j))
                continue
            else:
                val_mse_array.append(F.mse_loss(x_eeg, val_rec))
                val_sdtw_array.append(compute_recon_error_between_two_tensor(x_eeg, val_rec, average_channels=True, average_time_samples=True))
                validation_divergence_array.append(compute_divergence_SDTW_recon_error_between_two_tensor(x_eeg, val_rec, average_channels=True, average_time_samples=True))

        final_val_mse=np.mean(val_mse_array)
        final_val_sdtw=np.mean(val_sdtw_array)
        final_validation_divergence=np.mean(validation_divergence_array)
        final_val_mse_array.append(final_val_mse)
        final_val_sdtw_array.append(final_val_sdtw)
        final_validation_divergence_array.append(final_validation_divergence)

        for w in range(test_data.shape[0]):
            x_eeg=test_data[w,:,:,:]
            x_eeg = x_eeg.astype(np.float32)
            x_eeg = torch.from_numpy(x_eeg).unsqueeze(0)
            test_rec=model.reconstruct(x_eeg)
            if torch.isnan(test_rec).any():
                print("reconstructed trial {} of the test set  is nan".format(w))
                continue
            elif torch.isinf(test_rec).any():
                print("reconstructed trial {} of the test set is inf".format(w))
                continue
            else:
                test_mse_array.append(F.mse_loss(x_eeg, test_rec))
                test_sdtw_array.append(compute_recon_error_between_two_tensor(x_eeg, test_rec, average_channels=True, average_time_samples=True))
                test_divergence_array.append(compute_divergence_SDTW_recon_error_between_two_tensor(x_eeg, test_rec, average_channels=True, average_time_samples=True))

        final_test_mse=np.mean(test_mse_array)
        final_test_sdtw=np.mean(test_sdtw_array)
        final_test_divergence=np.mean(test_divergence_array)
        final_test_mse_array.append(final_test_mse)
        final_test_sdtw_array.append(final_test_sdtw)
        final_test_divergence_array.append(final_test_divergence)
        writer.add_scalars('MSE', {
            'Train MSE': final_train_mse,
            'Validation MSE': final_val_mse,
            'Test MSE': final_test_mse
        }, i)

        writer.add_scalars('SDTW', {
            'Train SDTW': final_train_sdtw,
            'Validation SDTW': final_val_sdtw,
            'Test SDTW': final_test_sdtw
        }, i)
        writer.add_scalars('divergence DTW', {
            'Train divergence DTW': final_train_divergence,
            'Validation SDTW': final_validation_divergence,
            'Test SDTW': final_test_divergence
        }, i)
        print('end iteration {}'.format(i))

    writer.close()

    t_train = torch.linspace(0, len(final_train_mse_array) - 1, steps=len(final_train_mse_array)).numpy()  # Convert to NumPy array if needed
    t_val = torch.linspace(0, len(final_val_mse_array) - 1, steps=len(final_val_mse_array)).numpy()  # Convert to NumPy array if needed
    t_test = torch.linspace(0, len(final_test_mse_array) - 1, steps=len(final_test_mse_array)).numpy()  # Convert to NumPy array if needed

    plt.rcParams.update()
    fig, ax = plt.subplots()  # Adjust figsize for better visibility
    ax.plot(t_train, final_train_mse_array, label='MSE on the train set', color='green', linewidth=1)
    ax.plot(t_val, final_val_mse_array, label='MSE on the validation set', color='red', linewidth=1)
    ax.plot(t_test, final_test_mse_array, label='MSE on the test set', color='blue', linewidth=1)
    ax.legend(loc='upper right')
    ax.set_ylabel('MSE')
    ax.set_xlabel('trial number')
    ax.grid(True)
    # Adjust layout
    plt.tight_layout()

    os.makedirs(f'/home/azorzetto/train{train_session}/img_rec_ability_over_epochs', exist_ok=True)

    output_path = Path('/home/azorzetto/train{}/img_rec_ability_over_epochs/{}.png'.format(train_session, 'MSE'))
    plt.savefig(output_path, dpi=300)
    mpld3.save_html(plt.gcf(), "/home/azorzetto/train{}/img_rec_ability_over_epochs/{}.html".format(train_session, 'MSE'))
    plt.close()

    plt.rcParams.update()
    fig, ax = plt.subplots()  # Adjust figsize for better visibility
    ax.plot(t_train, final_train_sdtw_array, label='Soft DTW on the train set', color='green', linewidth=1)
    ax.plot(t_val, final_val_sdtw_array, label='Soft DTW on the validation set', color='red', linewidth=1)
    ax.plot(t_test, final_test_sdtw_array, label='Soft DTW on the test set', color='blue', linewidth=1)
    ax.legend(loc='upper right')
    ax.set_ylabel('Soft DTW')
    ax.set_xlabel('trial number')
    ax.grid(True)
    # Adjust layout
    plt.tight_layout()

    output_path = Path('/home/azorzetto/train{}/img_rec_ability_over_epochs/{}.png'.format(train_session, 'Soft_DTW'))
    plt.savefig(output_path, dpi=300)
    mpld3.save_html(plt.gcf(), "/home/azorzetto/train{}/img_rec_ability_over_epochs/{}.html".format(train_session, 'Soft_DTW'))
    plt.close()

    plt.rcParams.update()
    fig, ax = plt.subplots()  # Adjust figsize for better visibility
    ax.plot(t_train, final_train_divergence_array, label='divergence DTW on the train set', color='green', linewidth=1)
    ax.plot(t_val, final_validation_divergence_array, label='divergence DTW on the validation set', color='red', linewidth=1)
    ax.plot(t_test, final_test_divergence_array, label='divergence DTW on the test set', color='blue', linewidth=1)
    ax.legend(loc='upper right')
    ax.set_ylabel('Soft DTW')
    ax.set_xlabel('trial number')
    ax.grid(True)
    # Adjust layout
    plt.tight_layout()

    output_path = Path('/home/azorzetto/train{}/img_rec_ability_over_epochs/{}.png'.format(train_session, 'Soft_DTW_divergence'))
    plt.savefig(output_path, dpi=300)
    mpld3.save_html(plt.gcf(), "/home/azorzetto/train{}/img_rec_ability_over_epochs/{}.html".format(train_session, 'Soft_DTW_divergence'))
    plt.close()


writer = SummaryWriter('runs/training_metrics')  # You can name the log directory as you wish

train_session=12
epochs=80 #number of epochs

reconstruction_ability_over_epochs(train_session=train_session, writer=writer, epochs=epochs)