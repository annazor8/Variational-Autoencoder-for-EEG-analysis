"""
Create a binary image which indicates the presence or absence of artifacts in the dataset
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plot_config = dict(
    figsize = (12, 8),
    fontsize = 24,
    colormap = 'Reds',
    # colormap = 'Greys',
    save_fig = True,
)

channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']

train_session=16
dataset_reconstructed=np.load("/home/azorzetto/train{}/reconstructed_dataset_divergence.npz".format(train_session))
dataset=np.load("/home/azorzetto/train{}/dataset.npz".format(train_session))
E_train=dataset_reconstructed["E_train"]
E_test=dataset_reconstructed["E_test"]

E_train=E_train*1000
E_test=E_test*1000

channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']


def plot_image(image_to_plot, plot_config : dict, title = None):
    image_to_plot=image_to_plot.squeeze()
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    ax.imshow(image_to_plot, aspect = 'auto', 
              cmap = plot_config['colormap'], interpolation='nearest'
              )

    xticks = np.arange(0, image_to_plot.shape[1], 150)
    yticks = np.arange(22)
    ax.set_xticks(xticks, labels = xticks)
    ax.set_yticks(yticks, labels = channel_names)
    ax.grid(True, color = 'black')
    
    ax.set_xlabel('Trials')
    ax.set_ylabel('Channels')
    
    if title is not None : ax.set_title(title, fontsize=plot_config['fontsize'])

    fig.tight_layout()
    plt.close()
    
    return fig, ax
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot image Train trials
fig_session_1_hv, ax_session_1_hv = plot_image(E_train, plot_config, title="Train trials")

# Create path
path_save = f'/home/azorzetto/train{train_session}/artifacts_map/'
os.makedirs(path_save, exist_ok = True)

# Save figs
path_save = f'/home/azorzetto/train{train_session}/artifacts_map/'
path_save += 'train_session_1000'
fig_session_1_hv.savefig(path_save + ".png", format = 'png')
fig_session_1_hv.savefig(path_save + ".eps", format = 'eps')


# Plot image test trials
fig_session_2_hv, ax_session_2_hv = plot_image(E_test, plot_config, title="Test trials")

# Create path
path_save = f'/home/azorzetto/train{train_session}/artifacts_map/'
os.makedirs(path_save, exist_ok = True)

# Save figs
path_save = f'/home/azorzetto/train{train_session}/artifacts_map/'
path_save += 'test_session_1000'
fig_session_2_hv.savefig(path_save + ".png", format = 'png')
fig_session_2_hv.savefig(path_save + ".eps", format = 'eps')

test_data_artifact=dataset['test_data_artifact']
train_data_artifact=dataset['train_data_artifact']
fig_session_1_artifacts, ax_session_1_artifacts = plot_image(train_data_artifact.squeeze().sum(2).transpose(), plot_config, title="Train trials artifact label")
fig_session_2_artifacts, ax_session_2_artifacts = plot_image(test_data_artifact.squeeze().sum(2).transpose(), plot_config, title="Test trials artifact label")
path_save = f'/home/azorzetto/train{train_session}/artifacts_map/'
path_save += 'train_session_labels'
fig_session_1_artifacts.savefig(path_save + ".png", format = 'png')
fig_session_1_artifacts.savefig(path_save + ".eps", format = 'eps')
path_save = f'/home/azorzetto/train{train_session}/artifacts_map/'
path_save += 'test_session_labels'
fig_session_2_artifacts.savefig(path_save + ".png", format = 'png')
fig_session_2_artifacts.savefig(path_save + ".eps", format = 'eps')

