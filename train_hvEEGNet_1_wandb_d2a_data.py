""" 
Example of training script for hvEEGNet with dataset 2a of BCI Competition IV and use of wandb to log results
The dataset is automatically downloaded through the functions inside the library.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from library.training import wandb_training as wt
from library.config import config_training as ct
from library.config import config_model as cm
from library.config import config_dataset as cd
from library.dataset import preprocess as pp
from library.model import hvEEGNet
import torch
import matplotlib.pyplot as plt
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Specific parameters to change inside the dictionary

# Subject of dataset 2a to use during training
subj_list = [3]

# Training parameters to change (for more info check the function get_config_hierarchical_vEEGNet_training)
epochs = 80
path_to_save_model = 'model_weights_backup_subj3'
epoch_to_save_model = 1
project_name = "Example_project"                # Name of wandb project
name_training_run = "first_test_wandb"          # Name of the training run
model_artifact_name = "temporary_artifacts"     # Name of the artifact used to save the model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get dataset config
dataset_config = cd.get_moabb_dataset_config(subj_list)

# Get model config
C = 22  # Number of EEG channels of dataset 2a
if dataset_config['resample_data']: sf = dataset_config['resample_freq']
else: sf = 250
T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf) # Compute number of time samples
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# Get training config
train_config = ct.get_config_hierarchical_vEEGNet_training()

# Update training config
train_config['epochs'] = epochs
train_config['path_to_save_model'] = path_to_save_model
train_config['epoch_to_save_model'] = epoch_to_save_model

# Update training config (wandb)
train_config['project_name'] = project_name
train_config['name_training_run'] = name_training_run
train_config['model_artifact_name'] = model_artifact_name

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']

# Train the model. 
model = wt.train_wandb_V1('hvEEGNet_shallow', dataset_config, train_config, model_config)

train_dataset, _, test_dataset = pp.get_dataset_d2a(dataset_config)
# Get a random sample from the test dataset
idx_trial = int(torch.randint(0, len(test_dataset), (1, 1)))
x_eeg, label = test_dataset[idx_trial]

# Add batch dimension
x_eeg = x_eeg.unsqueeze(0)

# Note on input size.
# When you get a single signal from the dataset (as with the instruction x_eeg, label = test_dataset[idx]) the data you obtain has shape 1 x C x T
# But hvEEGNet want an input of shape B x 1 x C x T. So for a single sample I have to add the batch dimension

# EEG Reconstruction. To reconstruct an input signal you could use the function reconstruct, implemented inside hvEEGNet
x_r_eeg = model.reconstruct(x_eeg)

# Create time array (for the plot)
t = torch.linspace(dataset_config['trial_start'], dataset_config['trial_end'], T)

# Get the idx of channel to plot
idx_ch = test_dataset.ch_list == ch_to_plot

# Plot the original and reconstructed signal
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(1, 1, figsize = (12, 8))

ax.plot(t, x_eeg.squeeze()[idx_ch].squeeze(), label = 'Original EEG', color = 'grey', linewidth = 2)
ax.plot(t, x_r_eeg.squeeze()[idx_ch].squeeze(), label = 'Reconstructed EEG', color = 'black', linewidth = 1)

ax.legend()
ax.set_xlim([2, 4]) # Note that the original signal is 4s long. Here I plot only 2 second to have a better visualization
ax.set_xlabel('Time [s]')
ax.set_ylabel(r"Amplitude [$\mu$V]")
ax.set_title("Subj {} (test) - Trial {} - Ch. {}".format(subj, idx_trial, ch_to_plot))
ax.grid(True)

fig.tight_layout()
fig.show()