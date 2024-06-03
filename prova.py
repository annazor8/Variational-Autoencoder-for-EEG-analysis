#----------------------------------------- Dataset--------------------------------------------------


from library.dataset import preprocess as pp
from library.config import config_dataset as cd
from library.config import config_training as ct

# Dataset 2a contains motor imagery for 9 subjects (indentified with number from 1 to 9). You can decide to download the data for one or for multiple subjects
subj_list = [8]       # If you want the data of a single subject create a list with a single element
#subj_list = [2, 4, 9]   # If you want the data for multiple subjects create a list with the number of all the subjects you want

# Get the dataset config. Check inside the functions for all the details about the various parameters
dataset_config = cd.get_moabb_dataset_config(subj_list)

# If you want to modify the data you can change the default settings after the creation of the dataset (or inside the get_moabb_dataset_config function)
# dataset_config['resample_data'] = True
# dataset_config['resample_freq'] = 128

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset creation

# This function automatically download the data through moabb, preprocess the data and create the dataset
# Note that the dataset is an istance of the class EEG_Dataset (you can find the code for the class inside the dataset_time subpackage)
train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# If you specify multuple subjects in the subj_list the data of the various subjects are mixed together
# So for example if subj_list = [2, 4, 9] the train_dataset will contain all the train data of subjects 2, 4 and 9 while the test_dataset will contain all the test data of the three subjects

# The valdiation_dataset contain a percentage of the training data that will not used for training but only for validation.
# The split between train/validation is controlled by the variable percentage_split_train_validation inside the dataset_config dictionary.
# E.g. percentage_split_train_validation = 0.9 means that 90% of the training data will be used for training and 10% for validation.

# Take a single EEG trial
single_trial_eeg, single_trial_label = train_dataset[2] #questo è un trial, quindi dimensione 1x1x22x1000 di un torch.tensor creato dalla classe ds.time_EEG_Dataset()

# Take multiple EEG trial
multiple_trial_eeg, multiple_trial_labels = train_dataset[0:] #it's calling the get item method of the class EEG_Dataset that needs an index
print(type(train_dataset))

train_config = ct.get_config_hierarchical_vEEGNet_training()
""" # Take multiple EEG trial
multiple_trial_eeg, multiple_trial_labels = train_dataset[4:20]"""




"""##Training"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from library.training import wandb_training as wt
from library.plot_function import plot_original_vs_reconstructed_EEG_SINGLE
from library.config import config_training as ct
from library.config import config_model as cm
from library.config import config_dataset as cd
import torch
from fastdtw import fastdtw
import numpy as np
from library.model import hvEEGNet

from pathlib import Path

import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Specific parameters to change inside the dictionary

# Training config to change (for more info check the function get_config_hierarchical_vEEGNet_training)
epochs = 1
path_to_save_model = 'model_weights_backup'
epoch_to_save_model = 5
project_name = "Example_project"                # Name of wandb project
name_training_run = "first_test_wandb"          # Name of the training run
model_artifact_name = "temporarybase_path = Path(__file__).resolve(strict=True).parent

print(base_path)
plt.savefig(base_path / "signal.png")_artifacts"     # Name of the artifact used to save the model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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

# Update training config (wandb)base_path = Path(__file__).resolve(strict=True).parent

print(base_path)
plt.savefig(base_path / "signal.png")
train_config['project_name'] = project_name
train_config['name_training_run'] = name_training_run
train_config['model_artifact_name'] = model_artifact_name

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']

#########prova
print((train_dataset[0][0]).size())
# Train the model
"""
model = wt.train_wandb_V1('hvEEGNet_shallow', dataset_config, train_config, model_config)
#model = hvEEGNet.hvEEGNet_shallow(model_config)

#-------------------------------base_path = Path(__file__).resolve(strict=True).parent

print(base_path)
plt.savefig(base_path / "signal.png")---RECONSTRUCT 1 SAMPLE--------------------------------------
"""
# This line of code will automatically set the device as GPU whether the system has access to one
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Get dataset and model

# Get dataset
dataset_config = cd.get_moabb_dataset_config([2])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# Get number of channels and number of time samples
C = train_dataset[0][0].shape[1]
T = train_dataset[0][0].shape[2]

# Get model config
model_config = cm.get_config_hie
base_path = Path(__file__).resolve(strict=True).parent

print(base_path)
plt.savefig(base_path / "signal.png")

# Create the model
model = hvEEGNet.hvEEGNet_shallow(model_config)

# Load the weights
model.load_state_dict(torch.load('./model_weights_backup/model_BEST.pth', map_location = torch.device('cpu')))
# P.s. this line here can throw you an error, depending on how you run the code.
# You could see the error that python doesn't find the weights file.
# This because the path to this specific weights file is defined relative to the root folder of the repository. So it is valid only if you run the script from that folder

# Note on device and weights
# By default when the weights are saved during the training they keep track of the device used by the model (i.e. CPU or GPU)
# So if you don't specify the map_location argument the torch load function expects that model and weights are in the same location.
# When a model is created in PyTorch its fastdtwlocation is the CPU. Instead the weights are saved from the GPU (because on 99% of the the time you will train the model with GPU)
# So the torch.load() function will throw an error if it find the model on CPU and the weights that want a model on GPU.
# To avoid this, when you load the model remember to specify map_location as cpu.
# In this way everything will be loaded in the CPU.
# Reconstruction of a single EEGbase_path = Path(__file__).resolve(strict=True).parent

print(base_path)
plt.savefig(base_path / "signal.png")

# Get a random sample from the test dataset
#idx_trial = int(torch.randint(0, len(test_dataset), (1, 1)))
#x_eeg, label = test_dataset[idx_trial]

idx_trial=82
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

ax.plot(t, x_eeg.squeeze()[idx_ch].squeeze(), label = 'Original EEG', color = 'green', linewidth = 2)
ax.plot(t, x_r_eeg.squeeze()[idx_ch].squeeze(), label = 'Reconstructed EEG', color = 'red', linewidth = 1)

ax.legend()
ax.set_xlim([2, 4]) # Note that base_path = Path(__file__).resolve(strict=True).parent

print(base_path)
plt.savefig(base_path / "signal.png")
ax.set_xlabel('Time [s]')
ax.set_ylabel(r"Amplitude [$\mu$V]")
ax.set_title("Subj {} (test) - Trial {} - Ch. {}".format(subj, idx_trial, ch_to_plot))
ax.grid(True)

fig.tight_layout()
fig.show()

#----------------------------RECONSTRUCT MULTIPLE SAMPLES-----------------------
# Reconstruction multple EEG signal

subj = 8

ch_to_plot = 'CP4' #define the channel we want to plot 

# This line of code will automatically set the device as GPU whether the system has access to one
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Load the weights
model.load_state_dict(torch.load('./model_weights_backup/model_BEST.pth', map_location = torch.device('cpu')))
# Instead of reconstruct a signal EEG signal maybe you want to reconstruct tens or hundreds of signal simultaneously.
# In this case you just need to put all signal in a single tensor and use the reconstruct method.

x_eeg_multi, label_multi = test_dataset [0:]
# Note that if you use the notation above to get more than 1 eeg signal the data you get from the dataset have shape N x 1 x C x T, with N = the number of EEG signals, 150 in this case.
print(x_eeg_multi.shape)
# Move the model and data to device
x_eeg_multi = x_eeg_multi.to(device)
model.to(device)

# Other note on device
# If you want to reconstruct a single signal the time difference between CPU and GPU is negligible.
# But if you want the reconstruction of multiple signals together the GPU is much more efficient.
# You will find some benchmarks in the last section of hvEEGNet paper

# Reconstruct multiple EEG signal
x_r_eeg_multi = model.reconstruct(x_eeg_multi)
x_eeg_multi=x_eeg_multi.to('cpu')
x_r_eeg_multi=x_r_eeg_multi.to('cpu')

'''from library.analysis import dtw_analysis as dtwa
index_trial=np.zeros((x_eeg_multi.shape[0]))
for i in range(x_eeg_multi.shape[0]):
  index_trial[i]=dtwa.compute_dtw_softDTWCuda(x_eeg_multi[i,:,:,:].unsqueeze(0), x_r_eeg_multi[i,:,:,:].unsqueeze(0))'''

 
 
#dist= dtwa.compute_dtw_softDTWCuda(x_eeg_multi, x_r_eeg_multi)
#print(index_trial)
#print(index_trial.shape)
#print(type(index_trial))

'''import numpy as np
max_value = np.max(index_trial)
max_indices = np.argmax(index_trial)

min_value=np.min(index_trial)
min_indices = np.argmin(index_trial)'''
print(x_r_eeg_multi.shape)
idx_trial=82
x_eeg, label = test_dataset[idx_trial]
print(x_eeg.shape)

from library.analysis.support import compute_loss_dataset
#prende in imput eeg_multi
#recon_loss_matrix=compute_loss_dataset( test_dataset, model,device)
print('the reconstruction error matrix is ')
#print(recon_loss_matrix)
print('the single trial for multi eeg shape is ')
print(x_eeg_multi[82,:,:,:].shape)
fig =plot_original_vs_reconstructed_EEG_SINGLE(x_eeg_multi[82,:,:,:], x_r_eeg_multi[82,:,:,:], 82, dataset_config, test_dataset, ch_to_plot, subj, T)


base_path = Path(__file__).resolve(strict=True).parent

print(base_path)
plt.savefig(base_path / "signal.png")
plt.show()"""