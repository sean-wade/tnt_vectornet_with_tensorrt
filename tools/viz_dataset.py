'''
Author: zhanghao
LastEditTime: 2023-06-02 17:56:16
FilePath: /my_vectornet_github/tools/viz_dataset.py
LastEditors: zhanghao
Description: 
'''
import os
import glob
import argparse
from tqdm import tqdm
from dataset.sg_dataloader import SGTrajDataset, collate_list
from torch.utils.data import Dataset, DataLoader
from dataset.util.vis_utils_v2 import Visualizer
import matplotlib.pyplot as plt


def viz(pkl_dirs, fig_save_path, inteval):
    os.makedirs(fig_save_path, exist_ok=True)
    dataset = SGTrajDataset(pkl_dirs, in_mem=False)
    vis = Visualizer(convert_coordinate=False, candidate=True)
    i = 0
    for data in tqdm(dataset, desc="Plotting dataset...."):
        if i % inteval == 0:
            vis.draw_once(data, gts=data["y"].view(-1, 2).cumsum(axis=0).cpu().numpy())
            plt.savefig(fig_save_path + "/" + data["seq_id"] + ".png")
            plt.close()
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", type=str, 
                            default="/mnt/data/SGTrain/TRAJ_DATASET/TRAJ_ALL_AGENTS_0516/train", 
                            help="root dir for datasets")

    parser.add_argument("-s", "--save_root", type=str, 
                            default="/mnt/data/SGTrain/TRAJ_DATASET/TRAJ_ALL_AGENTS_0516/viz/train", 
                            help="root dir for datasets")

    parser.add_argument("-i", "--inteval", type=int, default=1, help="every inteval save once.")

    args = parser.parse_args()

    pkl_list = [args.data_root]
    viz(pkl_list, args.save_root, args.inteval)
