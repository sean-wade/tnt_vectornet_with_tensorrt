'''
Author: zhanghao
LastEditTime: 2023-06-05 17:31:42
FilePath: /my_vectornet_github/tools/analysis_traj_type.py
LastEditors: zhanghao
Description: 
'''
import os
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from dataset.sg_dataloader import SGTrajDataset
from viz_dataset import viz

np.set_printoptions(suppress=True)

"""
分析轨迹类型，分为如下几个类别：
    1. 直行
    2. 转弯
    3. 静止
    4. 。。。。。。
"""

CURVE_THRESH = 2.0
STATIC_THRESH = 3.0

def analysis(args):
    # curve_dir    = args.ana_dir + "/" + args.split + "/curve"
    # static_dir   = args.ana_dir + "/" + args.split + "/static"
    # straight_dir = args.ana_dir + "/" + args.split + "/straight"
    # os.makedirs(straight_dir, exist_ok=True)
    # os.makedirs(static_dir, exist_ok=True)
    # os.makedirs(curve_dir, exist_ok=True)
    dst_dir = args.ana_dir + "/" + args.split
    os.makedirs(dst_dir, exist_ok=True)

    path_list = glob.glob(args.data_root + "/" + args.split)

    data_set = SGTrajDataset(path_list, in_mem=False)

    for data in tqdm(data_set, desc="Analysing trainset...."):
        # ['seq_id', 'candidate', 'candidate_gt', 'offset_gt', 'target_gt', 'orig', 'rot', 'traj_num', 'lane_num', 'y', 'x', 'cluster', 'identifier']
        traj_obs_rot = data["x"][data["x"][:, -1]==0][:,:2].numpy()
        traj_gt_rot = data["y"].view(-1, 2).cumsum(axis=0).numpy()

        x_shift_with_orig = abs(traj_gt_rot[-1, 0] - traj_obs_rot[0, 0])
        y_shift_with_orig = abs(traj_gt_rot[-1, 1] - traj_obs_rot[0, 1])
        xy_dis = (x_shift_with_orig**2+y_shift_with_orig**2) ** 0.5
        # print("x,y,dis = ", x_shift_with_orig, y_shift_with_orig, xy_dis)

        pkl_path = glob.glob(args.data_root + "/" + args.split + "/*" + data["seq_id"] + ".pkl")[0]
        if xy_dis < STATIC_THRESH:
            dst_path = dst_dir + "/" + data["seq_id"] + "_static.pkl"
        elif x_shift_with_orig > CURVE_THRESH:
            dst_path = dst_dir + "/" + data["seq_id"] + "_curve.pkl"
        else:
            dst_path = dst_dir + "/" + data["seq_id"] + "_straight.pkl"
        
        shutil.copy(pkl_path, dst_path)
    
    if args.viz:
        viz([dst_dir], dst_dir + "_fig", args.inteval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", type=str, default="/home/jovyan/workspace/DATA/TRAJ_DATASET/EXP6_Heading_Diamond_ALL/", help="root dir for datasets")
    parser.add_argument("-s", "--split", type=str, default="train", help="split of dataset")
    parser.add_argument("-a", "--ana_dir", type=str, default="/home/jovyan/workspace/DATA/TRAJ_DATASET/EXP6_Heading_Diamond_ANA/", help="save dir for datasets")
    parser.add_argument("-v", "--viz", type=bool, default=True, help="save dir for datasets")
    parser.add_argument("-i", "--inteval", type=int, default=100, help="every inteval save once.")
    args = parser.parse_args()
    
    analysis(args)
