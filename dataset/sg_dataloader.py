'''
Author: zhanghao
LastEditTime: 2023-06-28 16:45:24
FilePath: /my_vectornet_github/dataset/sg_dataloader.py
LastEditors: zhanghao
Description: 
'''
import os
import copy
import glob
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset.data_augment import *


# def collate_padding(samples):
#     traj_nums = np.array([d["traj_num"] for d in samples])
#     lane_nums = np.array([d["lane_num"] for d in samples])
#     valid_nums = traj_nums + lane_nums
#     num_valid_len_max = np.max(valid_nums)
#     candidate_nums = np.array([d["candidate"].shape[0] for d in samples])
#     num_candi_len_max = np.max(candidate_nums)


def collate_list(samples):
    return samples


def collate_list_cuda(samples, device=torch.device('cuda:0')):
    for i, b_data in enumerate(samples):
        for k, v in samples[i].items():
            if torch.is_tensor(v):
                samples[i][k] = samples[i][k].to(device)
    return samples


class SGTrajDataset(Dataset):
    def __init__(self,
                data_roots,
                augmentation = None,
                num_features = 10,
                in_mem = True,
                ):
        self.data_roots = data_roots
        self.in_mem = in_mem
        self.data_paths = []
        for data_root in data_roots: 
            self.data_paths = self.data_paths + sorted(glob.glob(data_root + "/*.pkl"))
        self.num_features = num_features
        
        assert len(self.data_paths) > 0, "Error, No file found under : %s"%(data_roots)
        if self.in_mem:
            self.data = [self.extract_data(idx) for idx in tqdm(range(len(self)), desc="Loading data in memory")]
        
        self.augmentation = augmentation
        self._set_group_flag()


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        if self.in_mem:
            data = self.data[idx]
        else:
            data = self.extract_data(idx)
        
        if self.augmentation:
            data = self.augmentation(data)

        return data


    def extract_data(self, idx):
        with open(self.data_paths[idx], "rb") as ppp:
            raw_data = pickle.load(ppp)
            
            # for compare
            if raw_data['x'].shape[1] > self.num_features:
                raw_data['x'] = raw_data['x'][:, [0,1,2,3,8,9]]
            
            return raw_data

            
    def _set_group_flag(self):
        self.flag = np.ones(len(self), dtype=np.uint8)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dataset.util.vis_utils_v2 import Visualizer
    vis = Visualizer(convert_coordinate=False, candidate=True)

    dataset = SGTrajDataset(data_roots = ['/mnt/data/SGTrain/TRAJ_DATASET/EXP8_Heading_Diamond_DIM10_BALANCE_MINI/train/'],
                            augmentation = TrainAugmentation(),
                            num_features = 10, 
                            in_mem = True
                        )
    
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_list)
    
    print(len(loader))
    for batch_data in tqdm(loader):
        if 1:
            vis.draw_once(batch_data[0], gts=batch_data[0]["y"].view(-1, 2).cumsum(axis=0).cpu().numpy())
            plt.show()
            plt.close()
