'''
Author: zhanghao
LastEditTime: 2023-03-02 18:03:14
FilePath: /sg_generate_vectorize_data_rosbag/visualize_traj.py
LastEditors: zhanghao
Description: 
    使用 matplotlib 绘制某一帧下的 bev 视角 obj 轨迹和车道线信息,
    磁盘上存储的数据均为 global 坐标系
'''
import os
import math 
import glob
import json
import shutil
import pickle
import argparse
import numpy as np
import pymap3d as pm
from tqdm import tqdm
from time import time
from numba import jit
from multiprocessing import Process
from matplotlib import pyplot as plt


tl_colors = ["black", "red", "green", "yellow", "gray"]


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as pp:
        return pickle.load(pp)


def plot_traj_data(tj_data, obs=20, pred=30, save_path=None):
    plt.figure(figsize=(12, 9))
    plt.grid(linestyle='dashed')

    trajs = tj_data['trajs']
    ego_mask = trajs[:, 1] == 0
    ego_traj = trajs[ego_mask]
    
    expand = 80
    orig = ego_traj[obs-1, 3:5]
    # plt.scatter(orig[0], orig[1], marker="o", s=60, c="orange", label="ego-t_obs")
    plt.xlim(orig[0] - expand, orig[0] + expand)
    plt.ylim(orig[1] - expand, orig[1] + expand)

    # # ego(target-agent)'s traj
    plt.plot(ego_traj[:obs, 3], ego_traj[:obs, 4], linestyle='-.', marker = '.', markersize=5, label="ego-obs", c="r")
    plt.plot(ego_traj[obs, 3], ego_traj[obs, 4], alpha=0.5, color='r', marker='o', zorder=20, markersize=10)
    # plt.plot(ego_traj[obs:, 3], ego_traj[obs:, 4], linestyle=':', marker = '^', markersize=3, label="ego-gt", c="coral")
    plt.plot(
        ego_traj[obs:, 3], ego_traj[obs:, 4], 
        linestyle=':', 
        linewidth=2,
        marker = '^', 
        markersize=5, 
        label="ego-gt", 
        c="coral", 
        alpha=0.9
    )

    # agents' traj
    other_traj = trajs[~ego_mask]
    track_ids = np.unique(other_traj[:, 1])
    for track_id in track_ids:
        cur_traj = other_traj[other_traj[:,1] == track_id][:obs]
        # plt.plot(cur_traj[:obs, 3], cur_traj[:obs, 4], linestyle='-.')
        # plt.scatter(cur_traj[-1, 3], cur_traj[-1, 4], s=15, c="pink", marker = '^')
        clr = "gold"
        zorder = 10
        plt.plot(cur_traj[:obs, 3], cur_traj[:obs, 4], marker='.', alpha=0.5, color=clr, zorder=zorder)
        plt.plot(cur_traj[-1, 3], cur_traj[-1, 4], alpha=0.5, color=clr, marker='o', zorder=zorder, markersize=10)

    # lane with single frame
    for cur_lane in tj_data["lane"]:
        points = np.array(cur_lane["points"])
        # xs = points[:, 0]
        # ys = points[:, 1]
        # plt.plot(xs, ys, label="lanes",linewidth=3.0)
        plt.plot(points[:, 0], points[:, 1], marker='.', alpha=0.5, color="grey")
    # # lane with stack frames
    # plt.scatter(tj_data["lane"][:,0], tj_data["lane"][:,1], label="lanes", s=2)

    # tls
    if tj_data["tl"][:,0].sum() > 0:
        tl_data = tj_data["tl"][:4, :]
        tl_dis_mean = tl_data[tl_data[:,0] > 0][:,1].mean()
        heading = ego_traj[obs-2, -1]
        tl_x = orig[0] + tl_dis_mean * np.cos(heading) - 4
        tl_y = orig[1] + tl_dis_mean * np.sin(heading)
        tl_color = tl_data[:, 0]
        ccs = [tl_colors[int(cc)] for cc in tl_color]
        tl_locs = np.array([[tl_x+i*2, tl_y] for i in range(len(tl_color))])
        plt.scatter(tl_locs[:,0], tl_locs[:,1], marker="o", s=50, c=ccs, label="traffic-lights(s-l-r-u)", alpha=0.7, zorder=30)

    plt.plot(-999, -999, marker='.', alpha=0.5, color="gold", label="obs_agent")
    plt.plot(-999, -999, marker='.', alpha=0.5, color="grey", label="lane")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_save_tjs(root):
    fig_path = root + "/fig_traj/"
    os.makedirs(fig_path, exist_ok=True)
    data_ps = sorted(glob.glob(root + "/traj_data/*.pkl"))

    for ii in tqdm(range(0, len(data_ps)), desc="Plotting & saving..."):
        tj_data = load_pkl(data_ps[ii])
        # plot_traj_data(tj_data)
        plot_traj_data(tj_data, save_path=fig_path + data_ps[ii].split("/")[-1][:-4] + ".png")


if __name__ == "__main__":
    root = "/mnt/data/SGTrain/rosbag/bag2/MKZ-A3QV50_2023-02-27_17-08-28_4_tj"
    plot_save_tjs(root)


