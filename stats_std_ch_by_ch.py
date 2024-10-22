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

#train_session = 0

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
#for train_session in range(11):
train_session="_jrj2"
# Get data
train_data = np.load('/home/azorzetto/trainShuffle{}/dataset.npz'.format(train_session))['train_data'].squeeze()
test_data = np.load('/home/azorzetto/trainShuffle{}/dataset.npz'.format(train_session))['test_data'].squeeze()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute the std for each channel

std_ch_train = train_data.std(2)
std_ch_test = test_data.std(2)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot the data (histogram)

fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

n_train, bins_train, patches_train=axs[0].hist(std_ch_train.flatten(), bins = plot_config['bins'], color = 'black')
axs[0].set_title('Train data', fontsize = plot_config['fontsize'], fontweight='bold')

n_test, bins_test, patches_test =axs[1].hist(std_ch_test.flatten(), bins = plot_config['bins'], color = 'black')
axs[1].set_title('Test data', fontsize = plot_config['fontsize'], fontweight='bold')
max_occurrences = max(n_train.max(), n_test.max())
for ax in axs:
    ax.set_xlim([0, 10])
    ax.set_ylim([0, max_occurrences])
    ax.set_xlabel('Standard deviation', fontsize = plot_config['fontsize'],fontweight='bold' )
    ax.set_ylabel('Number of occurrences', fontsize = plot_config['fontsize'],fontweight='bold' )
    ax.grid(True)
    if plot_config['use_log_scale_hist'] : ax.set_yscale('log')

fig.suptitle('Train session Shuffle {}'.format(train_session),fontsize = plot_config['fontsize'],fontweight='bold')
fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = "/home/azorzetto/stats_ch/prep_dataset_train_session_Shuffle{}/std/".format(train_session)
    os.makedirs(path_save, exist_ok = True)
    path_save += "hist_std_ch_by_ch_train_session_Shuffle{}".format(train_session)
    if plot_config['use_log_scale_hist'] : path_save += '_log'
    fig.savefig(path_save + '.png', format = 'png')
    # fig.savefig(path_save + 'hist_std_ch_by_ch_S{}.pdf'.format(subj), format = 'pdf')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot the data (average per trial) 

fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

axs[0].plot(std_ch_train.mean(axis = 1), color = 'black')
axs[0].fill_between(np.arange(std_ch_train.shape[0]), std_ch_train.mean(axis = 1) - std_ch_train.std(axis = 1), std_ch_train.mean(axis = 1) + std_ch_train.std(axis = 1), color = 'black', alpha = 0.2)
axs[0].set_title('Train data', fontsize = plot_config['fontsize'], fontweight='bold')

axs[1].plot(std_ch_test.mean(axis = 1), color = 'black')
axs[1].fill_between(np.arange(std_ch_test.shape[0]), std_ch_test.mean(axis = 1) - std_ch_test.std(axis = 1), std_ch_test.mean(axis = 1) + std_ch_test.std(axis = 1), color = 'black', alpha = 0.2)
axs[1].set_title('Test data', fontsize = plot_config['fontsize'], fontweight='bold')

min_val=min((std_ch_train.mean(axis = 1) - std_ch_train.std(axis = 1)).min(), (std_ch_test.mean(axis = 1) - std_ch_test.std(axis = 1)).min())
max_val=max((std_ch_train.mean(axis = 1) + std_ch_train.std(axis = 1)).max(), (std_ch_test.mean(axis = 1) + std_ch_test.std(axis = 1)).max())

for ax in axs:
    ax.set_ylim([min_val, max_val])
    ax.set_xlabel('Trial number', fontsize = plot_config['fontsize'],fontweight='bold' )
    ax.set_ylabel('Average standard deviation per trial', fontsize = plot_config['fontsize'], fontweight='bold')
    ax.grid(True)

fig.suptitle('Train session Shuffle {}'.format(train_session), fontsize = plot_config['fontsize'], fontweight='bold')
fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = "/home/azorzetto/stats_ch/prep_dataset_train_session_Shuffle{}/std/".format(train_session)
    os.makedirs(path_save, exist_ok = True)
    fig.savefig(path_save + 'avg_per_trial_train_session_Shuffle{}.png'.format(train_session), format = 'png')
    # fig.savefig(path_save + 'avg_per_trial_S{}.pdf'.format(subj), format = 'pdf')