"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of vEEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#%% Imports
import torch
from torch import nn

import EEGNet, MBEEGNet, Decoder_EEGNet 
import config_model

"""
%load_ext autoreload
%autoreload 2
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class vEEGNet(nn.Module):

    def __init__(self, config : dict):
        super().__init__()

        self.check_model_config(config)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Create Encoder

        # Convolutional section
        if config["type_encoder"] == 0:
            self.cnn_encoder = EEGNet.EEGNet(config['encoder_config']) 
        elif config["type_encoder"] == 1:
            self.cnn_encoder = MBEEGNet.MBEEGNet(config['encoder_config']) 
        
        # Get the size and the output shape after an input has been fed into the encoder
        # This info will also be used during the encoder creation
        n_input_neurons, decoder_ouput_shape = self.decoder_shape_info(config['encoder_config']['C'], config['encoder_config']['T'])

        # Get hidden space dimension
        self.hidden_space = config['hidden_space']

        # Feed-Forward section
        self.ff_encoder_mean = nn.Linear(n_input_neurons, self.hidden_space)
        self.ff_encoder_std = nn.Linear(n_input_neurons, self.hidden_space)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Create Decoder
        # Note that the config used for the encoder  are also used for the decoder
        
        # Information specific for the creation of the decoder
        config['encoder_config']['dimension_reshape'] = decoder_ouput_shape
        config['encoder_config']['hidden_space'] = self.hidden_space
        
        # For the decoder we use the same type of the encoder
        # E.g. if the encoder is EEGNet also the decoder will be EEGNet
        if config["type_encoder"] == 0:
            if config['type_decoder'] == 0:
                self.decoder = Decoder_EEGNet.EEGNet_Decoder_Upsample(config['encoder_config']) 
            elif config['type_decoder'] == 1:
                self.decoder = Decoder_EEGNet.EEGNet_Decoder_Transpose(config['encoder_config']) 
        elif config["type_encoder"] == 1:
            # TODO Implement MBEEGNet decoder 
            self.decoder = MBEEGNet(config['encoder_config']) 

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        # Other


    def forward(self, x):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -                 
        # Encoder section
        x = self.cnn_encoder(x)
        z_mean = self.ff_encoder_mean(x)
        z_log_var = self.ff_encoder_std(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Reparametrization

        z = self.reparametrize(z_mean, z_log_var)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Decoder

        x_r = self.decoder(z)

        return x_r, z_mean, z_log_var

    def reparametrize(self, mu, log_var):
        """
        Reparametrization Trick to allow gradients to backpropagate from the stochastic part of the model
        mu = Mean of the laten gaussian
        log_var = logartim of the variance of the latent guassian
        """
        
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        eps = eps.type_as(mu) # Setting z to be cuda when using GPU training 
        
        return mu + (sigma * eps)

    def generate(self):
        # Sample laten space (normal distribution)
        z = torch.randn(1, self.hidden_space)
        
        # Generate EEG sample
        x_g = self.decoder(z)

        return x_g

    def decoder_shape_info(self, C, T):
        """
        Compute the total number of neurons for the feedforward layer
        Compute the shape of the input after pass through the convolutional encoder

        Note that the computation are done for an input with batch size = 1
        """
        # Create fake input
        x = torch.rand(1, 1, C, T)

        # Deactivate the flatten of the output to obtain the output shape of the convolutional encoder
        self.cnn_encoder.flatten_output = False

        # Pass the fake input inside the encoder
        x = self.cnn_encoder(x)

        # Reactivate the flatten of the output
        self.cnn_encoder.flatten_output = True

        # Compute the number of neurons needed for the feedforward layer
        input_neurons = len(x.flatten())
        
        # Get the shape at the output of the convolutional encoder
        # The dimension in position 0 is the batch dimension and it is set to -1 to ignore it during the reshape
        decoder_ouput_shape = list(x.shape)
        decoder_ouput_shape[0] = -1

        return input_neurons, decoder_ouput_shape

    def check_model_config(self, config : dict):
        # Check type encoder
        if config["type_encoder"] == 0: print("EEGNet encoder selected")
        elif config["type_encoder"] == 1: print("MBEEGNet encoder selected")
        else: raise ValueError("type_encoder must be 0 (EEGNET) or 1 (MBEEGNet)")

        # Check type decoder 
        if config["type_decoder"] == 0: print("Upsample decoder selected")
        elif config["type_decoder"] == 1: print("Transpose decoder selected")
        else: raise ValueError("type_decoder must be 0 (Upsample) or 1 (Transpose)")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def check_vEEGNet():
    """
    Function to check the absence of breaking bug during the creation and the forward pass of vEEGNet
    """
    C = 22
    T = 512
    hidden_space = 16
    type_encoder = 0
    type_decoder = 0

    config = config_model.get_config_vEEGNet(C, T, hidden_space, type_encoder, type_decoder)
    model = vEEGNet(config)

    x = torch.rand(5, 1, C, T)
    x_r, z_mean, z_log_var = model(x)

    print("Input shape : ", x.shape)
    print("Output shape: ", x_r.shape)
    print(z_mean.shape)
    print(z_log_var.shape)

