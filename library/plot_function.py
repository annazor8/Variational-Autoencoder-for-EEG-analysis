from library.training import wandb_training as wt

from library.config import config_training as ct
from library.config import config_model as cm
from library.config import config_dataset as cd
import torch
import matplotlib.pyplot as plt

def plot_original_vs_reconstructed_EEG_MULTI(x_eeg_multi, x_r_eeg_multi, idx_ch, dataset_config, ch_to_plot, test_dataset, subj, T): 
  # Create time array (for the plot)
  t = torch.linspace(dataset_config['trial_start'], dataset_config['trial_end'], T)

  # Get the idx of channel to plot
  idx_ch = test_dataset.ch_list == ch_to_plot
  # Plot the original and reconstructed signal
  plt.rcParams.update({'font.size': 20})

  for i in range(x_eeg_multi.shape[0]):
    fig, ax = plt.subplots(1, 1, figsize = (12, 8))
    ax.plot(t, x_eeg_multi[i,:,:].squeeze()[idx_ch].squeeze(), label = 'Original EEG', color = 'green', linewidth = 2)
    ax.plot(t, x_r_eeg_multi[i,:,:].squeeze()[idx_ch].squeeze(), label = 'Reconstructed EEG', color = 'red', linewidth = 1)

    ax.legend()
    #ax.set_xlim([2, 4]) # Note that the original signal is 4s long. Here I plot only 2 second to have a better visualization
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r"Amplitude [$\mu$V]")
    ax.set_title("Subj {} (test) - Trial {} - Ch. {}".format(subj, i, ch_to_plot))
    ax.grid(True)

    fig.tight_layout()
  return fig
 
def plot_original_vs_reconstructed_EEG_SINGLE(x_eeg, x_r_eeg, idx_trial, dataset_config, test_dataset, ch_to_plot, subj, T):
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
  ax.set_xlim([2, 4]) # Note that the original signal is 4s long. Here I plot only 2 second to have a better visualization
  ax.set_xlabel('Time [s]')
  ax.set_ylabel(r"Amplitude [$\mu$V]")
  ax.set_title("Subj {} (test) - Trial {} - Ch. {}".format(subj, idx_trial, ch_to_plot))
  ax.grid(True)

  fig.tight_layout()
  return fig