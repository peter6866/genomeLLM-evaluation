import torch
from torch.utils.data import Dataset
import pandas as pd
import selene_sdk
import numpy as np


def one_hot_encode(seqs):
    """Given a list of sequences, one-hot encode them.

    Parameters
    ----------
    seqs : list-like
        Each entry is a DNA sequence to be one-hot encoded

    Returns
    -------
    seqs_hot : ndarray, shape (number of sequences, 4, length of sequence)
    """
    seqs_hot = list()
    for seq in seqs:
        seqs_hot.append(
            selene_sdk.sequences.Genome.sequence_to_encoding(seq).T
        )
    seqs_hot = np.stack(seqs_hot)
    return seqs_hot

class DeepSTARRDataset(Dataset):
    def __init__(self, data_type):
        if data_type == 'train':
            data_df = pd.read_csv('data/train_data.csv')
        
        if data_type == 'val':
            data_df = pd.read_csv('data/val_data.csv')
        
        if data_type == 'test':
            data_df = pd.read_csv('data/test_data.csv')
            
        sequence = data_df['Sequence']
        
        self.seqs_hot = one_hot_encode(sequence)
        self.target = torch.tensor(data_df["Dev_log2_enrichment"].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        x = self.seqs_hot[idx]
        y = self.target[idx]
        
        return x, y