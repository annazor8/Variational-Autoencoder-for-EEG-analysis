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
import plotly.graph_objects as go
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

train_session=12
dataset_reconstructed=np.load("/home/azorzetto/train{}/reconstructed_dataset_divergence.npz".format(train_session))
dataset=np.load("/home/azorzetto/train{}/dataset.npz".format(train_session))
E_train=dataset_reconstructed["E_train"]
E_test=dataset_reconstructed["E_test"]

E_train=E_train*1000
E_test=E_test*1000

channel_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']

def plot_image_plotly(image_to_plot, plot_config: dict, title=None, channel_names=None):
    """
    Plots an image with Plotly, includes grid lines, and returns a Plotly Figure object.
    """
    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=image_to_plot, 
            colorscale=plot_config['colormap'], 
            colorbar=dict(title="Value")
        )
    )

    # Define x-axis and y-axis ticks
    xticks = np.arange(0, image_to_plot.shape[1], 150)
    yticks = np.arange(len(channel_names)) if channel_names else np.arange(image_to_plot.shape[0])

    # Update layout for axes and title
    fig.update_layout(
        xaxis=dict(
            title="Trials",
            tickvals=xticks,
            ticktext=[str(x) for x in xticks],
            title_font=dict(size=20, family="Arial", color="black"),
            tickfont=dict(size=16, family="Arial", color="black"),
            showgrid=False  # Disable default grid
        ),
        yaxis=dict(
            title="Channels",
            tickvals=yticks,
            ticktext=channel_names if channel_names else [str(y) for y in yticks],
            title_font=dict(size=20, family="Arial", color="black"),
            tickfont=dict(size=16, family="Arial", color="black"),
            showgrid=False  # Disable default grid
        ),
        shapes=[  # Add custom grid lines
            dict(type='line', x0=x, x1=x, y0=0, y1=image_to_plot.shape[0], line=dict(color="black", width=0.5))
            for x in xticks
        ] + [
            dict(type='line', x0=0, x1=image_to_plot.shape[1], y0=y, y1=y, line=dict(color="black", width=0.5))
            for y in yticks
        ],
    )

    # Add title if provided
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=plot_config['fontsize'], family="Arial", color="black"),
                x=0.5,  # Center title
            )
        )

    return fig
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
    
    ax.set_xlabel('Trials', fontsize=20, fontweight='bold')
    ax.set_ylabel('Channels', fontsize=20, fontweight='bold')
    
    if title is not None : ax.set_title(title, fontsize=plot_config['fontsize'], fontweight='bold')

    fig.tight_layout()
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=10)  # Maggiori dimensioni per tick principali
    plt.tick_params(axis='both', which='minor', labelsize=12, width=1, length=5)   # Tick minori, se presenti

    # Imposta i tick label in grassetto
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontsize(16)        # Imposta dimensione del font per i tick
        tick.set_fontweight('bold')  # Imposta il font in grassetto
    plt.close()
    
    return fig, ax
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot image Train & test trials png
fig_session_1_hv, ax_session_1_hv = plot_image(E_train, plot_config, title="Train trials")
fig_session_2_hv, ax_session_2_hv = plot_image(E_test, plot_config, title="Test trials")
# Plot image Train & test trials html --------------------------------------
fig_session_1_hv_html = plot_image_plotly(E_train, plot_config, title="Train trials", channel_names=channel_names)
fig_session_2_hv_html = plot_image_plotly(E_test, plot_config, title="Test trials", channel_names=channel_names)

# Create path
base_path = f'/home/azorzetto/train{train_session}/artifacts_map/'
os.makedirs(base_path, exist_ok = True)

train_dtw_path=os.path.join(base_path, 'train_session_1000.html')
test_dtw_path=os.path.join(base_path, 'test_session_1000.html')
fig_session_1_hv_html.write_html(train_dtw_path)
fig_session_2_hv_html.write_html(test_dtw_path)

test_png_path_dtw = os.path.join(base_path, 'test_session_1000.png')
train_png_path_dtw = os.path.join(base_path, 'train_session_1000.png')
fig_session_1_hv.savefig(train_png_path_dtw, format = 'png', dpi=300)
fig_session_2_hv.savefig(test_png_path_dtw, format = 'png', dpi=300)  

#-------------------labelling TUAR-----------------------------------------------------
test_data_artifact=dataset['test_data_artifact']
train_data_artifact=dataset['train_data_artifact']
fig_session_1_artifacts, ax_session_1_artifacts = plot_image(train_data_artifact.squeeze().sum(2).transpose(), plot_config, title="Train trials artifact label")
fig_session_2_artifacts, ax_session_2_artifacts = plot_image(test_data_artifact.squeeze().sum(2).transpose(), plot_config, title="Test trials artifact label")
# Plot image Train & test trials html --------------------------------------
fig_session_1_artifacts_html = plot_image_plotly(train_data_artifact.squeeze().sum(2).transpose(), plot_config, title="Train trials", channel_names=channel_names)
fig_session_2_artifacts_html = plot_image_plotly(test_data_artifact.squeeze().sum(2).transpose(), plot_config, title="Test trials", channel_names=channel_names)

train_png_path = os.path.join(base_path, 'train_session_labels.png')
train_html_path = os.path.join(base_path, 'train_session_labels.html')
fig_session_1_artifacts.savefig(train_png_path, format = 'png', dpi=300)
fig_session_1_artifacts_html.write_html(train_html_path)

test_png_path = os.path.join(base_path, 'test_session_labels.png')
test_html_path = os.path.join(base_path, 'test_session_labels.html')
fig_session_2_artifacts.savefig(test_png_path, format = 'png', dpi=300)
fig_session_2_artifacts_html.write_html(test_html_path)

