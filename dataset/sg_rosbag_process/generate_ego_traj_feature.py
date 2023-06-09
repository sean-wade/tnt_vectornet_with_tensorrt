'''
Author: zhanghao
LastEditTime: 2023-03-02 17:19:07
FilePath: /sg_generate_vectorize_data_rosbag/generate_ego_traj_feature.py
LastEditors: zhanghao
Description: 

    每隔 5 个时间单位, 取 50s 的数据生成 dict(保存为 pickle), 内容及格式如下: 
        """
            以 ego 为 target-agent, 前后 50 帧(20帧为observe, 30帧为prediction)

            trajectory: 
                分为 obs 和 pred, 分别是前 20 帧和后 30 帧的所有 agent 观测轨迹
                其中每个 timestamp 的第一个目标是 ego 自车信息, 非感知输出, 需要额外计算
            
            lane:
                timestamp_19 时刻下观测到的车道线信息, 考虑以下两个方向
                    a. 转换为 SPline 存储
                    b. 将过去 20 帧的观测车道线按车道线ID合并
            
            # tl:
            #     timestamp_19 下的红绿灯感知结果, 格式为 4 * 1 
        """
        data = {
            "obs" : [
                [timestamp_0, track_id, 'ego', gx, gy, gz, -, -, -, -, -, heading],
                [timestamp_0, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],
                [timestamp_0, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],

                [timestamp_1, track_id, 'ego', gx, gy, gz, -, -, -, -, -, heading],
                [timestamp_1, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],
                [timestamp_1, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],
                ......
                [timestamp_19, track_id, 'ego', gx, gy, gz, -, -, -, -, -, heading],
                [timestamp_19, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],
            ],
            "pred" : [
                [timestamp_20, track_id, 'ego', gx, gy, gz, -, -, -, -, -, heading],
                [timestamp_20, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],
                [timestamp_20, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],

                [timestamp_21, track_id, 'ego', gx, gy, gz, -, -, -, -, -, heading],
                [timestamp_21, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],
                [timestamp_21, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],
                ......
                [timestamp_49, track_id, 'ego', gx, gy, gz, -, -, -, -, -, heading],
                [timestamp_49, track_id, type, gx, gy, gz, dx, dy, dz, vx, vy, heading],
            ],
            "lane" : [
                SPline2D,
                SPline2D,
                ......
            ]
            # "tl" : [gx, gy, color_1, color_2, color_3, color_4]
        }
    
'''
import os
import math
import glob
import json
import shutil
import rosbag
import pickle
import argparse
import numpy as np
import pymap3d as pm
from tqdm import tqdm
from time import time
from numba import jit
from multiprocessing import Process
from scene_filter import SceneFilter

classes = [
    "CAR",
    "TRUCK",
    "BUS",
    "CONSTRUCTION",
    "TRAILER",

    "TRICYCLE",
    "CYCLIST",

    "PEDESTRIAN",

    "CONE",
    "BLUR",
    "OTHER",
    "UNKNOWN",
]


def load_trk_obj(txt_path, time_stamp=0.0, uuid_to_int={}):
    with open(txt_path, "r") as ff:
        lines = ff.readlines()
        obj_list = []
        for line in lines:
            o = line.split(" ")
            uuid = str(o[11].strip())
            if uuid not in uuid_to_int:
                uuid_to_int[uuid] = len(uuid_to_int.keys()) + 1

            obj_list.append([
                time_stamp,
                uuid_to_int[uuid],
                classes.index(o[10].strip()),
                float(o[0]), float(o[1]), float(o[2]),
                float(o[3]), float(o[4]), float(o[5]),
                float(o[6]), float(o[7]), float(o[8]),
            ])
        return obj_list

def read_sg_data(pkl_data):
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

class TrajDataGenerater:
    def __init__(self, root, seq_inteval=10, obs_horizon=20, pred_horizon=30,filter_scene=[]):
        self.root = root
        self.seq_inteval = seq_inteval
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.len_horizon = obs_horizon + pred_horizon

        self.is_filter = len(filter_scene)
        self.fiter_scene_list = filter_scene

        self.trk_path = os.path.join(root, "lidar_obj_tracked")
        self.pose_path = os.path.join(root, "ins_to_global")
        self.lane_path = os.path.join(root, "lane_detected")
        # self.tl_path = os.path.join(root, "tl_detected")

        assert os.path.exists(self.trk_path), "track path not exist !!!"
        assert os.path.exists(self.pose_path), "pose path not exist !!!"
        assert os.path.exists(self.lane_path), "lane path not exist !!!"
        # assert os.path.exists(self.tl_path), "tl path not exist !!!"

        self.trk_ps = sorted(glob.glob(self.trk_path + "/*.txt"))
        self.pose_ps = sorted(glob.glob(self.pose_path + "/*.txt"))
        self.lane_ps = sorted(glob.glob(self.lane_path + "/*.json"))
        # self.tl_ps   = sorted(glob.glob(self.tl_path + "/*.txt"))

        self.tl_ps = [ss.replace("lane_detected", "tl_detected").replace(
            ".json", ".txt") for ss in self.lane_ps]

        # assert len(self.trk_ps) == len(self.tl_ps) == len(self.pose_ps) == len(self.lane_ps), "file nums does not equal !"
        assert len(self.trk_ps) == len(
            self.pose_ps), "file nums does not equal !"
        self.file_num = len(self.trk_ps)
        self.get_seq_nums()
        print("Init done, total file nums = %d, seq nums = %d" %
              (self.file_num, self.seq_nums))

        self.traj_path = os.path.join(self.root, "traj_data")
        os.makedirs(self.traj_path, exist_ok=True)

        self.filter_path = {}
        if self.is_filter:
            for scene in self.fiter_scene_list:
                self.filter_path[scene] = os.path.join(self.root,scene)
                os.makedirs(self.filter_path[scene], exist_ok=True)

    def get_seq_nums(self):
        if self.file_num < self.len_horizon:
            self.seq_nums = 0
        else:
            self.seq_nums = (
                self.file_num - self.len_horizon) // self.seq_inteval + 1

    def process(self, seq_add=0, dir_name=''):
        for seq_idx in tqdm(range(self.seq_nums), desc="Processing and saving traj data."):
            idx_s = seq_idx * self.seq_inteval

            traj_data = self.get_trajs_by_idxs(idx_s)
            lane_data = self.get_lane_by_idx(idx_s + self.obs_horizon - 1)
            # lane_data = self.get_lane_by_idxs_stack(idx_s)
            tl_data = self.get_tl_by_idx(idx_s + self.obs_horizon - 1)
            data = {
                "trajs": traj_data,
                "lane": lane_data,
                "tl": tl_data,
            }

            if self.is_static_station(data):
                continue

            if self.is_filter:
                sf = SceneFilter(data,self.fiter_scene_list)
                sl = sf.process()
                for scene in sl:
                    save_path = os.path.join(self.filter_path[scene], "data_seq_%s_%d_%s.pkl" % (dir_name, seq_idx + seq_add,scene))
                    with open(save_path, "wb") as f:
                        pickle.dump(data, f)
            else:
                save_path = os.path.join(
                    self.traj_path, "data_seq_%s_%d.pkl" % (dir_name, seq_idx + seq_add))
                with open(save_path, "wb") as f:
                    pickle.dump(data, f)

    def get_tl_by_idx(self, idx):
        if not os.path.exists(self.tl_ps[idx]):
            return np.array([[0.,  0.],
                             [0.,  0.],
                             [0.,  0.],
                             [0.,  0.],
                             [0., -1.]])
        else:
            return np.loadtxt(self.tl_ps[idx])

    def get_trajs_by_idxs(self, idx_start):
        trajs = []
        uuid_to_int = {}
        for idx in range(idx_start, idx_start + self.len_horizon):
            ins2global = np.loadtxt(self.pose_ps[idx])
            yaw_rad_diff = np.arctan2(ins2global[1, 0], ins2global[0, 0])
            ego_feature = [idx, 0, 0, 0, 0, 0,
                           4.5, 2.0, 1.8, 0.0, 0.0, np.pi/2.0]

            traj_agents = load_trk_obj(self.trk_ps[idx], idx, uuid_to_int)
            traj_agents = np.array([ego_feature] + traj_agents)

            points_homo = np.hstack((traj_agents[:, 3:6], np.ones(
                (traj_agents.shape[0], 1)))).reshape(-1, 4, 1)
            traj_agents[:, 3:6] = (
                ins2global @ points_homo).reshape(-1, 4)[:, :3]
            traj_agents[:, -1] += yaw_rad_diff
            trajs.append(traj_agents)

        return np.concatenate(trajs)

    def get_lane_by_idxs_stack(self, idx_start):
        ''' 
            应使用 lane_id 进行拼接, 但是目前车道线没有跟踪 id
            这里使用简单把所有点记录下的方法
        '''
        all_lane_points = np.empty((0, 2))
        for idx in range(idx_start, idx_start + self.len_horizon):
            lane_data = json.load(open(self.lane_ps[idx], "r"))
            ins2global = np.loadtxt(self.pose_ps[idx])
            for cur_lane in lane_data["lanes"]:
                points = np.array(cur_lane["points"]).reshape(-1, 3)
                points_homo = np.hstack(
                    (points, np.ones((points.shape[0], 1)))).reshape(-1, 4, 1)
                points_global = ins2global @ points_homo
                all_lane_points = np.vstack(
                    (all_lane_points, points_global.reshape(-1, 4)[:, :2]))
                # if cur_lane["lane_id"] not in all_lanes:
                #     all_lanes[cur_lane["lane_id"]] = {
                #         "lane_type" : cur_lane["lane_type"],
                #         "points_color" : cur_lane["points_color"],
                #         "points_type" : cur_lane["points_type"],
                #         "points" : cur_lane["points"],
                #     }
                # else:
                #     all_lanes[cur_lane["lane_id"]]["points"].append(
                #         cur_lane["points"]
                #     )

        return all_lane_points

    def get_lane_by_idx(self, idx):
        if not os.path.exists(self.lane_ps[idx]):
            return np.empty((0, 2))

        lane_data = json.load(open(self.lane_ps[idx], "r"))["lanes"]
        ins2global = np.loadtxt(self.pose_ps[idx])

        for cur_lane in lane_data:
            points = np.array(cur_lane["points"])[:, :3]
            points_homo = np.hstack(
                (points, np.ones((points.shape[0], 1)))).reshape(-1, 4, 1)
            points_global = ins2global @ points_homo
            cur_lane["points"] = points_global.reshape(-1, 4)[:, :2]

        return lane_data

    def is_static_station(self,pkl_data):
        data = read_sg_data(pkl_data)

        orig = data['trajs'][0][self.obs_horizon -1][3:5].copy().astype(np.float32)
        end = data['trajs'][0][-1][3:5].copy().astype(np.float32)
        start = data['trajs'][0][0][3:5].copy().astype(np.float32)

        pre = (orig - data['trajs'][0][self.obs_horizon-4][3:5]) / 2.0
        theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2

        rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)

        dis_o2s = math.sqrt(pow((start[0]-orig[0]),2)+pow((start[1]-orig[1]),2))
        dis_o2e = math.sqrt(pow((end[0]-orig[0]),2)+pow((end[1]-orig[1]),2))

        obj_front_list = self.has_front_obj(data,orig,rot)
        obj_front_list.sort(key=lambda x:(x[1],x[0]))
        if dis_o2e < 4 and dis_o2s < 4 :
            if len(obj_front_list)==0 or obj_front_list[0][1] > 10:
                return True
        
        return False

    def has_front_obj(self,data,orig,rot):
        obj_front_list = []
        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_horizon-1 not in step:
                continue
            index = np.where(step == self.obs_horizon-1)
            # print(index[0][0])
            obj_pos = traj[index[0][0]][3:5].astype(np.float32)
            #print(obj_pos)
            obj2ego = np.matmul(rot, (obj_pos - orig).T).T
            if abs(obj2ego[0]) < 2 and obj2ego[1] > 0:
                obj_front_list.append(obj2ego)
        return obj_front_list

if __name__ == "__main__":
    tdg = TrajDataGenerater("/mnt/data/SGTrain/rosbag/bag1/traj_data_tl/")
    tdg.process()
