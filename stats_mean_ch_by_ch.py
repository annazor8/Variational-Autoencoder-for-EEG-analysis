"""
For the subject in subject_list compute the std of each channel of each trials and plot them
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Import

import numpy as np
import matplotlib.pyplot as plt
import os

from library.analysis import support
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plot_config = dict(
    use_TkAgg_backend = True,
    figsize = (20, 12),
    bins = 400,
    use_log_scale_hist = False,
    fontsize = 20,
    save_fig = True,
)

path_dataset_config = 'training_scripts/config/TUAR/dataset.toml'
path_model_config = 'training_scripts/config/TUAR/model.toml'
path_traing_config = 'training_scripts/config/TUAR/training.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plot_config['use_TkAgg_backend']:
    plt.switch_backend('TkAgg')
for train_session in range(11):

    # Get data
    train_data = np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))['train_data'].squeeze()
    test_data = np.load('/home/azorzetto/train{}/dataset.npz'.format(train_session))['test_data'].squeeze()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Compute the mean for each channel

    mean_ch_train = train_data.mean(2)
    mean_ch_test = test_data.mean(2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Plot the data (histogram)

    fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

    axs[0].hist(mean_ch_train.flatten(), bins = plot_config['bins'], color = 'black')
    axs[0].set_title('Train data', fontsize = plot_config['fontsize'], fontweight='bold')

    axs[1].hist(mean_ch_test.flatten(), bins = plot_config['bins'], color = 'black')
    axs[1].set_title('Test data', fontsize = plot_config['fontsize'], fontweight='bold')

    for ax in axs:
        ax.set_xlabel('Mean', fontsize = plot_config['fontsize'], fontweight='bold')
        ax.set_ylabel('Number of occurrences', fontsize = plot_config['fontsize'], fontweight='bold')
        if plot_config['use_log_scale_hist'] : ax.set_yscale('log')

    fig.suptitle('Train session {}'.format(train_session), fontsize = plot_config['fontsize'], fontweight='bold')
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "/home/azorzetto/stats_ch/prep_dataset_train_session_{}/mean/".format(train_session)
        os.makedirs(path_save, exist_ok = True)
        path_save += "hist_mean_ch_by_ch_train_session_{}".format(train_session)
        if plot_config['use_log_scale_hist'] : path_save += '_log'
        fig.savefig(path_save + '.png', format = 'png')
        # fig.savefig(path_save + 'hist_mean_ch_by_ch_S{}.pdf'.format(subj), format = 'pdf')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Plot the data (average per trial) 

    fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

    axs[0].plot(mean_ch_train.mean(axis = 1), color = 'black')
    axs[0].fill_between(np.arange(mean_ch_train.shape[0]), mean_ch_train.mean(axis = 1) - mean_ch_train.std(axis = 1), mean_ch_train.mean(axis = 1) + mean_ch_train.std(axis = 1), color = 'black', alpha = 0.2)
    axs[0].set_title('Train data', fontsize = plot_config['fontsize'], fontweight='bold')

    axs[1].plot(mean_ch_test.mean(axis = 1), color = 'black')
    axs[1].fill_between(np.arange(mean_ch_test.shape[0]), mean_ch_test.mean(axis = 1) - mean_ch_test.std(axis = 1), mean_ch_test.mean(axis = 1) + mean_ch_test.std(axis = 1), color = 'black', alpha = 0.2)
    axs[1].set_title('Test data', fontsize = plot_config['fontsize'],fontweight='bold')

    for ax in axs:
        ax.set_xlabel('Trial number', fontsize = plot_config['fontsize'], fontweight='bold')
        ax.set_ylabel('Average mean per trial', fontsize = plot_config['fontsize'], fontweight='bold')

    fig.suptitle('Train session {}'.format(train_session), fontsize = plot_config['fontsize'], fontweight='bold')
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "/home/azorzetto/stats_ch/prep_dataset_train_session_{}/mean/".format(train_session)
        os.makedirs(path_save, exist_ok = True)
        fig.savefig(path_save + 'avg_per_trial_train_session_{}.png'.format(train_session), format = 'png')
        # fig.savefig(path_save + 'avg_per_trial_S{}.pdf'.format(subj), format = 'pdf')