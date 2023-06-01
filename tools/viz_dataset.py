'''
Author: zhanghao
LastEditTime: 2023-06-01 17:44:58
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


def viz(args):
    os.makedirs(args.data_root + "viz/train/", exist_ok=True)
    os.makedirs(args.data_root + "viz/val/", exist_ok=True)

    train_path_list = glob.glob(args.data_root + "/train")
    val_path_list = glob.glob(args.data_root + "/val")
    print(train_path_list)
    print(val_path_list)

    train_set = SGTrajDataset(train_path_list, in_mem=False)
    val_set = SGTrajDataset(val_path_list, in_mem=False)
    vis = Visualizer(convert_coordinate=False, candidate=True)

    # i = 0
    # for data in tqdm(train_set, desc="Plotting trainset...."):
    #     if i % args.inteval == 0:
    #         vis.draw_once(data, gts=data["y"].view(-1, 2).cumsum(axis=0).cpu().numpy())
    #         plt.savefig(args.data_root + "viz/train/" + data["seq_id"] + ".png")
    #         plt.close()
    #     i += 1

    i = 0
    for data in tqdm(val_set, desc="Plotting valset...."):
        if i % args.inteval == 0:
            vis.draw_once(data, gts=data["y"].view(-1, 2).cumsum(axis=0).cpu().numpy())
            plt.savefig(args.data_root + "viz/val/" + data["seq_id"] + ".png")
            plt.close()
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", type=str, default="/mnt/data/SGTrain/TRAJ_DATASET/TRAJ_ALL_AGENTS_0516/", help="root dir for datasets")
    parser.add_argument("-i", "--inteval", type=int, default=10, help="every inteval save once.")
    args = parser.parse_args()

    viz(args)
