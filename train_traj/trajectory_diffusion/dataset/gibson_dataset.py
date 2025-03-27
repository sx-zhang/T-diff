import bz2
import torch
import _pickle as cPickle
from torch.utils.data import Dataset
import os
from typing import Dict

def count_file_in_folder(path):
    count = 0
    for _, _, files in os.walk(path):
        count += len(files)
    return count

class GibsonMapDataset(Dataset):
    def __init__(self, path, train_idx):
        self.train_idx = train_idx
        self.path = path
                    
    def __len__(self):
        return len(self.train_idx)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tmp_idx = self.train_idx[idx]
        with bz2.BZ2File("{}/{}.pbz2".format(self.path, str(tmp_idx)), 'rb') as fp:
            data = cPickle.load(fp)
        
        tmp = torch.zeros(32,2)
        tmp[:, 0] = 1-data['action'][:, 1]
        tmp[:, 1] = data['action'][:, 0]
        o_data = {
            'obs':{
                'sem_map': data['obs']['sem_map'],
                'target': data['obs']['target'],
                'loc': torch.Tensor([1-data['obs']['loc'][1],data['obs']['loc'][0]]),
            },
            'action': tmp[:28, :],
        }
        
        return o_data
    