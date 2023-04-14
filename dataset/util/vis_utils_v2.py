'''
Author: zhanghao
LastEditTime: 2023-04-14 10:26:32
FilePath: /my_vectornet_github/dataset/util/vis_utils_v2.py
LastEditors: zhanghao
Description: 
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Visualizer():
    def __init__(self, xlim=80, ylim=80, candidate=False, convert_coordinate=False):
        self.xlim = xlim
        self.ylim = ylim
        self.candidate = candidate
        self.convert_coordinate = convert_coordinate


    def draw_once(self, data, preds, gts=None, probs=None):
        _, ax = plt.subplots(figsize=(12, 12))
        ax.axis('equal')
        # plt.axis('off')
        ax.set_title('seq_id: {}'.format(data["seq_id"]))

        if self.convert_coordinate:
            orig = data["orig"]
            rot = data["rot"]
        else:
            orig = np.array([0,0])
            rot = np.array([[1,0], [0,1]])

        # obs trajs
        traj_num = int(data["traj_num"])
        for i in range(traj_num):
            cur_traj = data["x"][data["cluster"] == i][:, :2]
            cur_traj = np.matmul(np.linalg.inv(rot), cur_traj.T).T + orig.reshape(-1, 2)
            clr = "r" if i==0 else "gold"
            zorder = 20  if i==0 else 10
            ax.plot(cur_traj[:, 0], cur_traj[:, 1], marker='.', alpha=0.5, color=clr, zorder=zorder)
            ax.plot(cur_traj[-1, 0], cur_traj[-1, 1], alpha=0.5, color=clr, marker='o', zorder=5, markersize=15)

        # lane
        lane_len = int(data["lane_num"])
        for i in range(lane_len):
            cur_lane = data["x"][data["cluster"] == i + traj_num][:, :2]
            cur_lane = np.matmul(np.linalg.inv(rot), cur_lane.T).T + orig.reshape(-1, 2)
            ax.plot(cur_lane[:, 0], cur_lane[:, 1], marker='.', alpha=0.5, color="grey")

        # gts 
        ax.plot(
            gts[:, 0], 
            gts[:, 1], 
            linestyle=':', 
            linewidth=3,
            marker = '^', 
            markersize=5, 
            label="ego-gt", 
            c="coral", 
            alpha=0.9
        )
        ax.plot(gts[-1, 0], gts[-1, 1], marker='o', color='coral', markersize=15, alpha=0.4, zorder=5, label="gt-final")

        # pred
        for pp, pred in enumerate(preds):
            # ax.plot(pred[:, 0], pred[:, 1], linestyle='--', alpha=0.7, label="pred")
            # alpha = (6 - pp)/6
            alpha = 0.65 if pp==0 else 0.2
            ax.plot(pred[:, 0], pred[:, 1], alpha=alpha, color='g', linewidth=2, marker='.', zorder=15, label="pred")
            ax.plot(pred[-1, 0], pred[-1, 1], marker='*', color='g', markersize=12, alpha=alpha, zorder=30)
            if probs is not None:
                ax.text(pred[-1, 0], pred[-1, 1], '{:.3f}'.format(probs[pp]), zorder=15)

        if self.candidate:
            candidate_pts = data["candidate"]
            candidate_pts = np.matmul(np.linalg.inv(rot), candidate_pts.T).T + orig.reshape(-1, 2)
            ax.scatter(candidate_pts[:, 0], candidate_pts[:, 1], marker='.', color='deepskyblue', alpha=0.1)

        # for x-y lim
        ax.set_xlim(orig[0] - self.xlim, orig[0] + self.xlim)
        ax.set_ylim(orig[1] - self.ylim, orig[1] + self.ylim)

        # for label legend
        ax.plot(-999, -999, marker='.', alpha=0.5, color="gold", zorder=zorder, label="obs_agent")
        ax.plot(-999, -999, marker='.', alpha=0.5, color="r", zorder=zorder, label="obs_target")
        ax.plot(-999, -999, marker='.', alpha=0.5, color="grey", label="lane")

        plt.legend()
        # plt.show()