"""
Computation of the average reconstruction error for each subject for each epoch across repetition
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.signal as signal
import pickle

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.dataset import preprocess as pp
from library.training import train_generic
from library.training.soft_dtw_cuda import SoftDTW
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

if len(sys.argv) > 1:
    tot_epoch_training = sys.argv[1]
    subj_list = sys.argv[2]
else:
    tot_epoch_training = 80
    subj_list = [2, 9]
    repetition_list = np.arange(19) + 1
    epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    use_test_set = False

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

recon_loss_results = dict()

for subj in subj_list:
    print("Subj: ", subj)
    recon_loss_results[subj] = dict()
    
    # Get the datasets 
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

    for repetition in repetition_list:
        print("\tRep: ", repetition)
        for epoch in epoch_list:
    
            if use_test_set: dataset = test_dataset
            else: dataset = train_dataset
            
            # Load model weight
            path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, epoch)
            model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
            
            if epoch not in recon_loss_results[subj]:    
                recon_loss_results[subj][epoch] = support.compute_loss_dataset(dataset, model_hv, device, batch_size) / 1000
            else:
                recon_loss_results[subj][epoch] += support.compute_loss_dataset(dataset, model_hv, device, batch_size) / 1000
              
            # Save the results for each repetition
            path_save = 'Saved Results/repetition_hvEEGNet_{}/subj {}/'.format(tot_epoch_training, subj)
            os.makedirs(path_save, exist_ok = True)
            
            path_save = 'Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_rep_{}.pickle'.format(tot_epoch_training, subj, epoch, repetition)
            pickle_out = open(path_save, "wb") 
            pickle.dump(recon_loss_results[subj][epoch] , pickle_out) 
            pickle_out.close() 
            
            path_save = 'Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_rep_{}.npy'.format(tot_epoch_training, subj, epoch, repetition)
            np.save(path_save, recon_loss_results[subj][epoch])

#%% Average accross repetition and save the results
for subj in subj_list:
    for epoch in epoch_list:
        path_save = 'Saved Results/repetition_hvEEGNet_{}/subj {}/'.format(tot_epoch_training, subj)
        os.makedirs(path_save, exist_ok = True)
        
        path_save = 'Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_average.pickle'.format(tot_epoch_training, subj, epoch)
        pickle_out = open(path_save, "wb") 
        pickle.dump(recon_loss_results[subj][epoch] / len(repetition_list) , pickle_out) 
        pickle_out.close() 
        
        path_save = 'Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch)
        np.save(path_save, recon_loss_results[subj][epoch] / len(repetition_list))
