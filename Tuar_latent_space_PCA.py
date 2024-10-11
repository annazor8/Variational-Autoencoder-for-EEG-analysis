import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch 
from library.config import config_model as cm
from library.training import train_generic
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly as ploty
from pathlib import Path

def plot_latent_space_interactive(latent_pca, artifact_perc, file_name:str, title="PCA Latent Space", save_path=None):
    # Creiamo la traccia per il plot
    trace = go.Scatter(
        x=latent_pca[:, 0], 
        y=latent_pca[:, 1], 
        mode='markers',
        text=[str(i) for i in range(latent_pca.shape[0])],
        textposition='top center',
        marker=dict(
            size=10,
            color=artifact_perc,  # Coloriamo i punti in base alla media del PSD
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Artifact Sum") 
        ),
        hoverinfo='text',  # Mostra l'indice al passaggio del mouse
    )

    # Configuriamo il layout
    layout = go.Layout(
        title={
            'text': title,
            'font': {
                'size': 24,  # Imposta la dimensione del titolo
                'family': 'Arial, sans-serif',  # Imposta il font
                'color': 'black',  # Imposta il colore del titolo
                'weight': 'bold'  # Imposta il titolo in grassetto
            },
            'x': 0.5,  # Centra il titolo rispetto all'asse x
            'xanchor': 'center'  # Ancora il titolo al centro dell'asse x
        },
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        width=1000,  # Aumenta la larghezza
        height=800,  # Aumenta l'altezza
        autosize=False,  # Disabilita il ridimensionamento automatico
        margin=dict(l=100, r=100, t=100, b=100),  # Margini per centrare il grafico
        xaxis=dict(
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False
        ),
    )

    # Creiamo la figura
    fig = go.Figure(data=[trace], layout=layout)

    # Salviamo il plot se è specificato un percorso
    if save_path:
        save_path=Path(save_path)
        file_path=f"fig_{file_name}_HTML.html"
        save_path= save_path / file_path
        ploty.io.write_html(fig, save_path)
    else:
        # Mostriamo il plot
        fig.show()

#-----specify parameters 
train_session=14
model_epoch=64

#-----load the dataset
dataset=np.load("/home/azorzetto/train{}/datasetREAL.npz".format(train_session))
dataset2=np.load("/home/azorzetto/train{}/dataset.npz".format(train_session))
train= dataset['train_data']#(685, 1, 22, 1000)
test = dataset['test_data']#(671, 1, 22, 1000)
artifact_train=dataset2['train_data_artifact']
artifact_test=dataset2['test_data_artifact']

#-------load the model
model_name='hvEEGNet_shallow'
model_config = cm.get_config_hierarchical_vEEGNet(22, 1000, type_decoder = 0, parameters_map_type = 0)
model_hv = train_generic.get_untrained_model(model_name, model_config)
path_weight="/home/azorzetto/train{}/model_weights_backup{}/model_epoch{}.pth".format(train_session, train_session, model_epoch)
model_hv.load_state_dict(torch.load(path_weight, map_location=torch.device('cpu')))
model_hv.eval()

# Estrai i vettori latenti usando il modello

#--------compute latent variables for train
array_train_PSD=[]
array_train_embedded=[]
for i in range(train.shape[0]):
    trial=train[i:i+1, :,:,:]
    x_embedding_train=model_hv.encode(torch.from_numpy(trial).float())[0].flatten(1) 
    freqs_session_1, psd_session_1=welch(x_embedding_train.detach().numpy(), fs=250, nperseg=256)
    array_train_PSD.append(psd_session_1)
    array_train_embedded.append(x_embedding_train.detach().numpy())

train_mean_latet_power=np.mean(array_train_PSD, axis=0)
train_std_latet_power=np.std(array_train_PSD, axis=0)


#--------compute latent variables for test
array_test_PSD=[]
array_test_embedded=[]
for i in range(test.shape[0]):
    trial=test[i:i+1, :,:,:]
    x_embedding_test=model_hv.encode(torch.from_numpy(trial).float())[0].flatten(1) 
    freqs_session_1, psd_session_1=welch(x_embedding_test.detach().numpy(), fs=250, nperseg=256)
    array_test_PSD.append(psd_session_1)
    array_test_embedded.append(x_embedding_test.detach().numpy())

test_mean_latet_power=np.mean(array_test_PSD, axis=0)
test_std_latet_power=np.std(array_test_PSD, axis=0)

train_latent=np.concatenate(array_train_embedded) #len(array_train_embedded)=685
test_latent=np.concatenate(array_test_embedded)#len(array_test_embedded)=671

pca = PCA(n_components=2)
train_latent_pca = pca.fit_transform(train_latent)# Addestra PCA sui dati e train_latent_pca ha dimensione 2, 685
test_latent_pca = pca.transform(test_latent) #applica la trasformazione ai dati di test ed è (671, 2)

array_perc_train=np.sum(artifact_train, axis=(1, 2))/22000
# Visualizza il latent space per il training
plot_latent_space_interactive(train_latent_pca, array_perc_train, file_name = 'TRAIN', title="PCA Latent Space - Training Data", save_path="/home/azorzetto/train{}".format(train_session))
array_perc_test=np.sum(artifact_test, axis=(1, 2))/22000
# Visualizza il latent space per il test
plot_latent_space_interactive(test_latent_pca, array_perc_test, file_name = 'TEST', title="PCA Latent Space - Test Data", save_path="/home/azorzetto/train{}".format(train_session))