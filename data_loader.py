import numpy as np
import torch

from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def normalize(data):
    
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    data = data / (max_val + 1e-7)
    
    data = data.astype(np.float32)
    #data = torch.from_numpy(data)
    
    return data


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