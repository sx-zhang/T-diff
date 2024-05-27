import gc
import cv2
import bz2
import math
import json
import tqdm
import h5py
import glob
import torch
import random
import numpy as np
import os.path as osp
import _pickle as cPickle
import skimage.morphology as skmp
from torch.utils.data import Dataset
import os
import clip
from diffusion_policy.common.pytorch_util import dict_apply
from torch.utils.data import DataLoader
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from typing import Dict

def count_file_in_folder(path):
    count = 0
    for _, _, files in os.walk(path):
        count += len(files)
    return count

class TrajectoryDataset(Dataset):
    def __init__(self, train_idx):
        self.train_idx = train_idx
    
    def  get_normalizer(self, mode='limits', **kwargs):
        with bz2.BZ2File("..diffusion/data/sample_h16/{}.pbz2".format(str(0)), 'rb') as fp:
            tmp_data = cPickle.load(fp)
            
        data = {
            'clip_feature': tmp_data['obs']['clip_feature'].numpy(),
            'action': tmp_data['action'].numpy()
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
                    
    def __len__(self):
        return len(self.train_idx)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tmp_idx = self.train_idx[idx]
        with bz2.BZ2File("..diffusion/data/sample_h16/{}.pbz2".format(str(tmp_idx)), 'rb') as fp:
            data = cPickle.load(fp)
        return data
    
