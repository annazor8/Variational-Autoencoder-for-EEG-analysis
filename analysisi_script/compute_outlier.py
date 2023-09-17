#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors as knn

from library.analysis import support
from library.config import config_dataset as cd 

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

tot_epoch_training = 80
subj_list = [2, 9]
epoch = 80

neighborhood_order = 5
knn_algorithm = 'auto'
s_knee = 1

plot_config = dict(
    figsize = (12, 8),
    fontsize = 12,
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

for subj in subj_list:
    path_recon_error = './Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch)
    # Load data

    # dataset_config = cd.get_moabb_dataset_config([subj])
    # dataset_config['percentage_split_train_validation'] = -1
    # train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

    # Load the reconstruction error
    recon_error = np.load(path_recon_error)

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Outlier identifications

    # Compute the KNN
    neighborhood_set   = knn(n_neighbors = neighborhood_order, algorithm = knn_algorithm).fit(recon_error)
    distances, indices = neighborhood_set.kneighbors(recon_error)

    # compute distances from nth nearest neighbors (given by neighborhood_order) and sort them
    dk_sorted     = np.sort(distances[:,-1])
    dk_sorted_ind = np.argsort(distances[:,-1])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #%% Knee plot

    try:
        from kneed import KneeLocator
        i = np.arange(len(distances))
        knee = KneeLocator(i, dk_sorted, S = s_knee, curve = 'convex', direction = 'increasing', interp_method = 'interp1d', online = True)
        knee_x = knee.knee
        knee_y = knee.knee_y    # OR: distances[knee.knee]

        print("Number of outliers for subj {} (S = {}): {}".format(subj, s_knee, recon_error.shape[0] - knee_x))

        ax.plot(dk_sorted, 'o-', label = 'Subject {}'.format(subj))
        ax.set_xlabel('EEG Trials', fontsize = plot_config['fontsize'])
        ax.set_ylabel('Distances (sorted)', fontsize = plot_config['fontsize'])
        ax.axvline(x = knee_x, color = 'k', linestyle = '--')
        ax.axhline(y = knee_y, color = 'k', linestyle = '--')
        ax.plot((knee_x), (knee_y), 'o', color = 'r')
        ax.set_title("Knee plot at epoch {}".format(epoch))
        ax.grid()
        ax.legend()

        fig.tight_layout()
        fig.show()

    except ImportError as e:
        print("Error -> ", e)
        print("If you want the knee plot install the kneed package (pip install kneed)")
