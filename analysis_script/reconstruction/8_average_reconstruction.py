"""
Compute the average reconstruction for a list of subjcet
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import sys
import os

from torch.utils.data import DataLoader

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.dataset import preprocess as pp
from library.training import train_generic
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [1, 2, 3]
# subj_list = [1]
use_test_set = False

epoch = 80
tot_epoch_training = 80

t_min = 2
t_max = 4
channel = 'C3'
compute_psd = False

plot_config = dict(
    figsize = (24, 8),
    fontsize = 24,
    add_std = True,
    alpha = 0.33,
    save_fig = True,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_to_color = ['red', 'blue', 'black', 'green', 'orange', 'violet', 'pink', 'brown', 'cyan']

def plot_signal(ax, horizontal_axis_value, x, horizontal_axis_value_r, x_r, compute_psd, plot_config):
    ax.plot(horizontal_axis_value, x, label = 'Original signal',
            color = plot_config['color_original'], linewidth = plot_config['linewidth_original'])
    ax.plot(horizontal_axis_value_r, x_r, label = 'Reconstructed signal',
            color = plot_config['color_reconstructed'], linewidth = plot_config['linewidth_reconstructed'])
    if compute_psd: ax.set_xlabel("Frequency [Hz]")
    else: ax.set_xlabel("Time [s]")
    ax.set_xlim([horizontal_axis_value_r[0], horizontal_axis_value_r[-1]])
    ax.legend()
    ax.grid(True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Compute the average reconstruction

plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }

x_avg_list = []
x_std_list = []

for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Reconstruction subj {}".format(subj))
    
    # Load/create data and model
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')
    
    # Select train/test data and create dataloader
    if use_test_set: 
        dataset = test_dataset
        string_dataset = 'test'
    else: 
        dataset = train_dataset
        string_dataset = 'train'
    dataloader = DataLoader(dataset, batch_size = 72)

    # Load model weights and move to device
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, 2, epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    model_hv.to(device)

    x_r = None

    for batch_data, batch_label in dataloader : 
        x = batch_data.to(device)

        latent_space_to_ignore = [True, True, False]
        x_r_deep_only = model_hv.h_vae.reconstruct_ignoring_latent_spaces(x, latent_space_to_ignore).squeeze()

        if x_r is None :
            x_r = x_r_deep_only
        else :
            x_r = torch.cat((x_r, x_r_deep_only), 0)

    x_avg_list.append(x_r.mean(0).cpu())
    x_std_list.append(x_r.std(0).cpu())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot data

idx_ch = dataset.ch_list == channel
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

for i in range(len(subj_list)):
    subj = subj_list[i]
    color = subj_to_color[i]

    x_avg = x_avg_list[i]
    x_std = x_std_list[i]

    if compute_psd:
        nperseg = 500
        horizontal_axis_value, x_avg_plot = signal.welch(x_avg[idx_ch].squeeze(), fs = 250, nperseg = nperseg)
        string_domain = 'freq'
    else :
        x_avg_plot, horizontal_axis_value = support.crop_signal(x_avg, idx_ch, 2, 6, t_min, t_max)
        x_std_plot, horizontal_axis_value = support.crop_signal(x_std, idx_ch, 2, 6, t_min, t_max)
        string_domain = 'time'

    ax.plot(horizontal_axis_value, x_avg_plot, 
            label = 'S{}'.format(subj), color = color
            )
    
    if plot_config['add_std']:
        ax.fill_between(horizontal_axis_value, x_avg_plot + x_std_plot, x_avg_plot - x_std_plot, 
                        color = color, alpha = plot_config['alpha']
                        )

if string_domain == 'freq': 
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"PSD [$\mu V^2/Hz$]")
elif string_domain == 'time': 
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Amplitude [$\mu$V]")

ax.set_xlim([horizontal_axis_value[0], horizontal_axis_value[-1]])
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.show()
