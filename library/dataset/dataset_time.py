"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function to the creation the PyTorch dataset with EEG data in format channels x time samples
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch.utils.data import Dataset, IterableDataset

"""
%load_ext autoreload
%autoreload 2

import dataset as ds
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% PyTorch Dataset
class EEG_Dataset_list(Dataset):
    def __init__(self, data_list, labels, ch_list, normalize=-1):
        """
        data_list: list of data arrays, each having shape [Trials x 1 x channels x time samples]
        labels: array of label arrays corresponding to the data
        ch_list: list of channels
        normalize: normalization method (1 for min-max normalization of the entire dataset, 2 for channel-wise normalization, -1 for nothing)
        """
        self.data_list = data_list
        self.labels = torch.from_numpy(labels).long()
        self.ch_list = ch_list
        self.normalize = normalize
        
        if self.normalize not in [-1, 1, 2]:
            raise ValueError("Normalize parameter must be -1, 1, or 2")
        
        # Transform data and labels into torch tensors
        self.data_list = [torch.from_numpy(data).float() for data in self.data_list]

    def __getitem__(self, idx: int):
            """
            Retrieve a sample and its label at the specified index.
            """ 
            if idx >= len(self.data_list) or idx < 0:
                raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self.data_list)}")
            sample = self.data_list[idx]
            return sample, self.labels
    def __len__(self):
            """
            Return the total number of samples in the dataset.
            """
            return len(self.data_list)
    

class EEG_Dataset(Dataset):
    """
    """

    def __init__(self, data, labels, ch_list, normalize = -1):
        
        #data = data used for the dataset. Must have shape [Trials x 1 x channels x time samples]
        #Note that if you use normale EEG data depth dimension (the second axis) has value 1.
        

        if len(data.shape) != 4 or data.shape[1] != 1 :
            raise ValueError("The input shape of data must be [Trials x 1 x channels x time samples]. Current shape {}".format(data.shape))

        # Transform data in torch array
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
        self.ch_list = ch_list
        
        # (OPTIONAL) Normalize
        if normalize == 1:
            self.minmax_normalize_all_dataset(-1, 1)
        elif normalize == 2:
            self.normalize_channel_by_channel(-1, 1)
            
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.labels)

    def minmax_normalize_all_dataset(self, a, b):
        """
        Normalize the entire dataset between a and b.
        """
        self.data = ((self.data - self.data.min()) / (self.data.max() - self.data.min())) * (b - a) + a

    def normalize_channel_by_channel(self, a, b):
        """
        Normalize each channel so the value are between a and b
        """
        
        # N.b. self.data.shape = [trials, 1 , channels, eeg samples]
        # The dimension with the 1 is needed because the conv2d require a 3d tensor input
        
        for i in range(self.data.shape[0]): # Cycle over samples
            for j in range(self.data.shape[2]): # Cycle over channels
                tmp_ch = self.data[i, 0, j]
                
                normalize_ch = ((tmp_ch - tmp_ch.min()) / (tmp_ch.max() - tmp_ch.min())) * (b - a) + a
                
                self.data[i, 0, j] = normalize_ch

#%% End file
