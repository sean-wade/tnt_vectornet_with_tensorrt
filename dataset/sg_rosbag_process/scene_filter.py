'''
Author: wys
LastEditTime: 2023-03-08 11:04:07
FilePath: /scene_filter.py
LastEditors: wys
Description: 

    根据需要分类的数据标签对从bag中提取的数据帧进行分类保存 如车道保持，变道，绕障等等: 
    
'''
import os
import sys
import numpy as np
import math
import time
import glob
import pickle
from matplotlib import pyplot as plt
from lc_filter import LaneChangeFilter

scene_dict = {'lc':'LaneChangeFilter'}
tl_colors = ["black", "red", "green", "yellow", "gray"]

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

class SceneFilter:
    def __init__(self,data,filter_scene_list) -> None:
        self.data = self.read_sg_data(data)
        #plot_traj_data(data)
        self.filter_scene_list = filter_scene_list
        self.scene = []

    def process(self):
        for scene in self.filter_scene_list:
            if scene == 'lc':
                self.lc_filter = LaneChangeFilter(self.data)
                if self.lc_filter.process():
                    self.scene.append(scene)

        return self.scene

    def read_sg_data(self,pkl_data):
        #pkl_data = pickle.load(open(pkl_path, "rb"))
        traj_info = pkl_data["trajs"]
        # [id , lane_type , confidence ,points] 目前来看每条车道points点数不固定
        lane_info = pkl_data["lane"]
        tl_info = pkl_data["tl"]  # traffic light

        agt_ts = sorted(np.unique(traj_info[:, 0]))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        steps = [mapping[x] for x in traj_info[:, 0]]
        steps = np.asarray(steps, np.int64)

        # track_id = 0: agent.
        agt_mask = traj_info[:, 1] == 0
        agt_traj = traj_info[agt_mask]
        agt_step = steps[agt_mask]

        ctx_ids = np.unique(traj_info[:, 1])
        ctx_trajs, ctx_steps = [], []
        for ctx_id in ctx_ids:
            if ctx_id != 0:
                ctx_mask = traj_info[:, 1] == ctx_id
                ctx_trajs.append(traj_info[ctx_mask])
                # steps 表示障碍物在哪一帧出现了，如果采集的数据是5s的话，最完整的是【0，1，2，..，50】
                ctx_steps.append(steps[ctx_mask])

        data = dict()
        data["trajs"] = [agt_traj] + ctx_trajs
        data["steps"] = [agt_step] + ctx_steps
        data["lane_raw"] = lane_info
        data["tl_raw"] = tl_info
        return data

if __name__ == "__main__":
    root = '/home/wangyisong/wys/TNT-SG/lc/MKZ-A3QV50_2023-03-01_10-45-47_6/traj_data/'
    save_dir ='/home/wangyisong/wys/TNT-SG/filter/'
    
    filter_list = ['lc']
    file_paths = glob.glob(root+"/*.pkl")
    for pkl_path in file_paths:
        #print(pkl_path)
        pkl_data = pickle.load(open(pkl_path, "rb"))
        sf = SceneFilter(pkl_data,filter_list)
        sl = sf.process()
        for scene in sl:
            tmp_dir = save_dir + '/' + scene
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            pkl_name=os.path.basename(pkl_path)
            #print(pkl_name)
            save_path = os.path.join(tmp_dir,pkl_name)
            with open(save_path, "wb") as f:
                pickle.dump(pkl_data, f)
