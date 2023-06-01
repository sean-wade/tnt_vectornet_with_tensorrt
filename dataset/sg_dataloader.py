'''
Author: zhanghao
LastEditTime: 2023-04-13 10:53:16
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
                in_mem = True):
        self.data_roots = data_roots
        self.in_mem = in_mem
        self.data_paths = []
        for data_root in data_roots: 
            self.data_paths = self.data_paths + sorted(glob.glob(data_root + "/*.pkl"))
        self.num_features = 6
        
        if self.in_mem:
            self.data = [self.extract_data(idx) for idx in tqdm(range(len(self)), desc="Loading data in memory")]


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        if self.in_mem:
            return self.data[idx]
        else:
            return self.extract_data(idx)


    def extract_data(self, idx):
        with open(self.data_paths[idx], "rb") as ppp:
            raw_data = pickle.load(ppp)
            return raw_data


if __name__ == '__main__':
    dataset = SGTrajDataset(data_roots = ['/home/jovyan/workspace/DATA/TRAJ_DATASET/TRAJ_ALL_AGENTS_0516/train/',
                                          '/home/jovyan/workspace/DATA/TRAJ_DATASET/TRAJ_ALL_AGENTS_0427/train/'], in_mem=True)
    
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, collate_fn=collate_list)
    
    print(len(loader))
    # for data in loader:
    #     print(data[0]["seq_id"], data[1]["seq_id"])
