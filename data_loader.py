import numpy as np
import torch

from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

    
def normalize(data):

  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  
  return norm_data
  


class TimeSeriesData(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_len):
        
        data = np.loadtxt(data_dir, delimiter=",", skiprows=1)
        data = data[::-1]

        norm_data = normalize(data)

        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)

        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
#####################################################################

class SyntheticDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        
        data = np.loadtxt(data_dir, delimiter=",", skiprows=1)
        data = data[::-1]

        norm_data = normalize(data)

        self.samples = []
        idx = torch.randperm(len(norm_data))
        for i in range(len(norm_data)):
            self.samples.append(norm_data[idx[i]])
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
