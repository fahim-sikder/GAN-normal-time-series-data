import torch
import torch.nn as nn
import math


class CommonGRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, mode = 'GRU', activation_fn = torch.sigmoid):
        
        super(CommonGRU, self).__init__()
        
        if mode == 'GRU':
            
        
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            
        else:
            
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)
        #self.linear = nn.DataParallel(self.linear) # parallel GPU
        
        self.activation_fn = activation_fn
        
    def forward(self, x):
        
        output, _ = self.rnn(x)
        
        output = self.linear(output)
        
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        
        
        return output
    
####################################################################################


class SimpleConvGen(nn.Module):
    
    def __init__(self, latent_dim):
        
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.gen = nn.Sequential(
            
            nn.BatchNorm1d(self.latent_dim),
            nn.Conv1d(self.latent_dim, 64, kernel_size = 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size = 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size = 1),
            nn.LeakyReLU(0.2, inplace =True),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, kernel_size = 1),
            nn.Tanh()
        
        )
        
    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        
        gen_out = self.gen(x)
        
        return gen_out
    
####################################################################################

class SimpleConvDisc(nn.Module):
    
    def __init__(self, input_features):
        
        super().__init__()
        
        self.input_features = input_features
        
        self.disc = nn.Sequential(
        
            nn.BatchNorm1d(self.input_features),
            nn.Conv1d(self.input_features, 64, kernel_size = 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size = 1),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, kernel_size = 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 1, kernel_size = 1),
            nn.Sigmoid()        
        
        )
        
    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        
        x = self.disc(x)
        
        x = torch.transpose(x, 1, 2)
        
        return x
    
################################################################


class LinearDisc(nn.Module):
    
    def __init__(self, input_features, hidden_dim):
        
        super().__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        
        self.disc = nn.Sequential(
        
            nn.BatchNorm1d(self.input_features),
            nn.Linear(self.input_features, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        
        )
        
    def forward(self, x):
        
        x = self.disc(x)
        
        return x
    
class LinearGen(nn.Module):
    
    def __init__(self, z_dim, input_features, hidden_dim):
        
        super().__init__()
        
        self.z_dim = z_dim
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        
        self.gen = nn.Sequential(
        
            nn.BatchNorm1d(self.z_dim),
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.input_features),
            nn.Tanh()
        
        )
        
    def forward(self, x):
        
        x = self.gen(x)
        
        return x