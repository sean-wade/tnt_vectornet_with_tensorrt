'''
Author: zhanghao
LastEditTime: 2023-03-10 20:33:38
FilePath: /vectornet/dataset/sg_preprocess.py
LastEditors: zhanghao2 zhanghao2@sg.cambricon.com
Description: 
    根据SG数据保存的 data_seq_{id}.pkl 数据进行转换，生成训练使用的 data_seq_{id}_features.pkl
    转换前的 pickle 格式如下:
        {
            "trajs"     : [ N * 12], 
            "lane"      : [ M * dict{'lane_id'      : 0, 
                                    'lane_type'     : 'LANE_LINE'
                                    'confidence',   : 0.0
                                    'points_color', : 1
                                    'points_type',  : 1
                                    'points'        : L * 2
                                    }],
            # "tl"      : [ 5 * 2]
        }
    转换后的 pickle keys 详见 assets/SGPreprocessor.png


'''
import os
import sys
PROJ_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
sys.path.append(PROJ_DIR)

import glob
import torch
import pickle
import argparse
import numpy as np
# import pandas as pd
from tqdm import tqdm
from copy import deepcopy, copy
from os.path import join as pjoin
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset.util.cubic_spline import Spline2D

import warnings
warnings.filterwarnings("ignore")


class SGPreprocessor:
    def __init__(self,
            root_dir,
            save_dir=None,
            algo="tnt",
            split="train",
            obs_range=100,
            obs_horizon=20,
            pred_horizon=30,
            normalized=True,
            sample_range=60,
            sample_resolution=4,
            viz=False
        ):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.algo = algo
        self.split = split
        self.obs_range = obs_range
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.normalized = normalized
        self.sample_range = sample_range
        self.sample_resolution = sample_resolution
        self.viz = viz

        self.file_paths = glob.glob(root_dir+"/*.pkl")
        print("SGPreprocessor file nums = ", len(self.file_paths))
        os.makedirs(self.save_dir, exist_ok=True)


    def __getitem__(self, idx):
        f_path = self.file_paths[idx]
        seq_name = f_path.split("/")[-1][:-4]
        # print(f_path)
        return self.process_and_save(f_path, seq_id=seq_name)


    def process_and_save(self, f_path, seq_id):
        data = self.read_sg_data(f_path)
        data = self.get_obj_feats(data)

        data['graph'] = self.get_lane_graph(data)
        data['seq_id'] = seq_id.split("_")[-1]
        # visualization for debug purpose
        if self.viz:
            self.visualize_data(data)

        f_path = self.save_dir + "/%s_features.pkl"%seq_id
        train_data = self.transform_for_training(data)
        self.save(train_data, f_path)
        return []


    def save(self, data, f_path):
        with open(f_path, 'wb') as fff:
            pickle.dump(data, fff)


    def __len__(self):
        return len(self.file_paths)


    @staticmethod
    def transform_for_training(interm_data):
        train_data = {
            "seq_id" : interm_data["seq_id"],
            "candidate" : torch.from_numpy(interm_data['tar_candts']).float(),
            "candidate_gt" : torch.from_numpy(interm_data['gt_candts']).bool(),
            "offset_gt" : torch.from_numpy(interm_data['gt_tar_offset']).float(),
            "target_gt" : torch.from_numpy(interm_data['gt_preds'][0][-1, :]).float(),
            "orig" : interm_data['orig'],
            "rot" : interm_data['rot'],
            "traj_num" : interm_data['feats'].shape[0],
        }
        if len(interm_data['graph']['lane_idcs']) > 0:
            lane_num = interm_data['graph']['lane_idcs'].max() + 1
        else:
            lane_num = 0
        train_data["lane_num"] = lane_num

        # get y
        traj_obs = interm_data['feats'][0]
        traj_fut = interm_data['gt_preds'][0]
        offset_fut = np.vstack([traj_fut[0, :] - traj_obs[-1, :2], traj_fut[1:, :] - traj_fut[:-1, :]])
        train_data["y"] = torch.from_numpy(offset_fut.reshape(-1).astype(np.float32)).float()

        # get x
        feats = np.empty((0, 6))
        identifier = np.empty((0, 2))

        traj_feats = interm_data['feats']
        traj_has_obss = interm_data['has_obss']
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        traj_cnt = 0
        for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
            xy_s = feat[has_obs][:-1, :2]
            vec = feat[has_obs][1:, :2] - feat[has_obs][:-1, :2]
            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[has_obs][:-1], polyline_id])])
            traj_cnt += 1
            
        # get lane features
        graph = interm_data['graph']
        if len(graph['lane_idcs']) > 0:
            ctrs = graph['ctrs']
            vec = graph['feats']
            lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt
            steps = np.zeros((len(lane_idcs), 1))
            feats = np.vstack([feats, np.hstack([ctrs, vec, steps, lane_idcs])])
        
        cluster = copy(feats[:, -1].astype(np.int64))
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])
        
        train_data["x"] = torch.from_numpy(feats).float()
        train_data["cluster"] = torch.from_numpy(cluster).short()
        train_data["identifier"] = torch.from_numpy(identifier).float()

        return train_data


    @staticmethod
    def read_sg_data(pkl_path):
        pkl_data = pickle.load(open(pkl_path, "rb"))
        traj_info = pkl_data["trajs"]
        lane_info = pkl_data["lane"]
        tl_info = pkl_data["tl"]

        agt_ts = sorted(np.unique(traj_info[:,0]))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        steps = [mapping[x] for x in traj_info[:,0]]
        steps = np.asarray(steps, np.int64)

        # track_id = 0: agent.
        agt_mask = traj_info[:,1] == 0
        agt_traj = traj_info[agt_mask]
        agt_step = steps[agt_mask]

        ctx_ids = np.unique(traj_info[:,1])
        ctx_trajs, ctx_steps = [], []
        for ctx_id in ctx_ids:
            if ctx_id != 0:
                ctx_mask = traj_info[:,1] == ctx_id
                ctx_trajs.append(traj_info[ctx_mask])
                ctx_steps.append(steps[ctx_mask])

        data = dict()
        data["trajs"] = [agt_traj] + ctx_trajs
        data["steps"] = [agt_step] + ctx_steps
        data["lane_raw"] = lane_info
        data["tl_raw"] = tl_info
        return data


    def get_obj_feats(self, data):
        # get the origin and compute the oritentation of the target agent
        orig = data['trajs'][0][self.obs_horizon-1][3:5].copy().astype(np.float32)

        # comput the rotation matrix
        if self.normalized:
            pre = (orig - data['trajs'][0][self.obs_horizon-4][3:5]) / 2.0
            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)
        else:
            # if not normalized, do not rotate.
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float32)

        # rotate lane points
        lane_points_spline = []
        for l in data["lane_raw"]:
            if l["points"].shape[0] > 5:
                lane_points_rot = np.matmul(rot, (l["points"] - orig.reshape(-1, 2)).T).T
                if len(lane_points_rot) > 300:
                    sample_inteval = len(lane_points_rot)//300+1
                    lane_points_rot = lane_points_rot[::sample_inteval]
                lane_points_spline.append(Spline2D(x=lane_points_rot[:, 0], y=lane_points_rot[:, 1], resolution=2.0))
        
        # get the target candidates and candidate gt
        agt_traj_obs = data['trajs'][0][0:self.obs_horizon, 3:5].copy().astype(np.float32)
        agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon, 3:5].copy().astype(np.float32)
        agt_traj_obs_rot = np.matmul(rot, (agt_traj_obs - orig.reshape(-1, 2)).T).T
        agt_traj_fut_rot = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T
        # use uniform sampling, range: [-60m-60m] / 30, resolution = 2m
        tar_candts = self.uniform_candidate_sampling(sampling_range=self.sample_range, resolution=self.sample_resolution)

        if self.split == "test":
            tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            # splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut)
            splines, ref_idx = None, None
            tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut_rot[-1])

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_horizon-1 not in step:
                continue

            # normalize and rotate
            traj_nd = np.matmul(rot, (traj[:,3:5] - orig.reshape(-1, 2)).T).T

            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), np.float32)
            has_pred = np.zeros(self.pred_horizon, np.bool)
            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
            post_step = step[future_mask] - self.obs_horizon
            post_traj = traj_nd[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = True

            # colect the observation
            obs_mask = step < self.obs_horizon
            step_obs = step[obs_mask]
            traj_obs = traj_nd[obs_mask]
            idcs = step_obs.argsort()
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs]
            # zhanghao add: why this? if step_obs = [0,1,2...17,-,19], should we delete it?
            for i in range(len(step_obs)):
                if step_obs[i] == self.obs_horizon - len(step_obs) + i:
                    break
            step_obs = step_obs[i:]
            traj_obs = traj_obs[i:]

            if len(step_obs) <= 1:
                continue

            feat = np.zeros((self.obs_horizon, 3), np.float32)
            has_obs = np.zeros(self.obs_horizon, np.bool)

            feat[step_obs, :2] = traj_obs
            feat[step_obs, 2] = 1.0
            has_obs[step_obs] = True

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            feats.append(feat)                  # displacement vectors
            has_obss.append(has_obs)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        # if len(feats) < 1:
        #     raise Exception()

        feats = np.asarray(feats, np.float32)
        has_obss = np.asarray(has_obss, np.bool)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offse_gt
        data['lane_points_spline'] = lane_points_spline

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
        return data


    def get_lane_graph(self, data):
        ctrs, feats, turn, control, intersect = [], [], [], [], []

        for lane_spline in data["lane_points_spline"]:
            lane_fine = np.hstack((lane_spline.x_fine.reshape(-1,1), lane_spline.y_fine.reshape(-1,1)))
            num_segs = len(lane_fine) - 1

            if num_segs > 0:
                ctrs.append(np.asarray((lane_fine[:-1] + lane_fine[1:]) / 2.0, np.float32))
                feats.append(np.asarray(lane_fine[1:] - lane_fine[:-1], np.float32))

                x = np.zeros((num_segs, 2), np.float32)
                turn.append(x)
                control.append(x.copy())
                intersect.append(x.copy())

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)
        num_nodes = count
        if num_nodes > 0:
            lane_idcs = np.concatenate(lane_idcs, 0)

            graph = dict()
            graph['ctrs'] = np.concatenate(ctrs, 0)
            graph['num_nodes'] = num_nodes
            graph['feats'] = np.concatenate(feats, 0)
            graph['turn'] = np.concatenate(turn, 0)
            graph['control'] = np.concatenate(control, 0)
            graph['intersect'] = np.concatenate(intersect, 0)
            graph['lane_idcs'] = lane_idcs
        else:
            graph = dict()
            graph['ctrs'] = None
            graph['num_nodes'] = num_nodes
            graph['feats'] = None
            graph['turn'] = None
            graph['control'] = None
            graph['intersect'] = None
            graph['lane_idcs'] = []

        return graph


    @staticmethod
    def get_centerline_from_edgelines(lane_points):
        pass


    @staticmethod
    def uniform_candidate_sampling(sampling_range, resolution=2):
        """
        uniformly sampling of the target candidate
        :param sampling_range: int, the maximum range of the sampling
        :param rate: the sampling rate (num. of samples)
        return rate^2 candidate samples
        """
        # x = np.linspace(-sampling_range, sampling_range, rate)
        x = np.arange(-sampling_range, sampling_range, resolution)
        return np.stack(np.meshgrid(x, x), -1).reshape(-1, 2)


    @staticmethod
    def get_candidate_gt(target_candidate, gt_target):
        """
        find the target candidate closest to the gt and output the one-hot ground truth
        :param target_candidate, (N, 2) candidates
        :param gt_target, (1, 2) the coordinate of final target
        """
        displacement = gt_target - target_candidate
        gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

        onehot = np.zeros((target_candidate.shape[0], 1))
        onehot[gt_index] = 1

        offset_xy = gt_target - target_candidate[gt_index]
        return onehot, offset_xy


    @staticmethod
    def get_ref_centerline(cline_list, pred_gt):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            line_idx = 0
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx


    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        plt.figure(0, figsize=(8, 7))
        plt.grid(linestyle='dashed')

        # visualize the centerlines
        lines_ctrs = data['graph']['ctrs']
        lines_feats = data['graph']['feats']
        lane_idcs = data['graph']['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            plt.plot(line[:,0], line[:,1], label="lanes", linewidth=3.0)

        # visualize the trajectory
        trajs = data['feats'][:, :, :2]
        has_obss = data['has_obss']
        preds = data['gt_preds']
        has_preds = data['has_preds']
        for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
            self.plot_traj(traj[has_obs], pred[has_pred], i)

        # plot target sample
        candidate_targets = data["tar_candts"]
        tar_candts_idx = data["gt_candts"].argmax()
        plt.scatter(candidate_targets[:, 0], candidate_targets[:, 1], marker="*", c="grey", alpha=0.2, s=6, zorder=15, label="sample points")
        plt.scatter(candidate_targets[tar_candts_idx, 0], candidate_targets[tar_candts_idx, 1], marker="*", c="orange", alpha=0.8, s=40, zorder=15, label="target sample")

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.legend()
        plt.show()


    def plot_traj(self, obs, pred, traj_id=None):
        assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        if traj_id == 0:
            plt.plot(obs[:, 0], obs[:, 1], linestyle='-.', marker = '.', markersize=5, label="ego-obs", c="limegreen")
            plt.plot(pred[:, 0], pred[:, 1], linestyle=':', marker = '^', markersize=3, label="ego-gt", c="coral")
        else:
            plt.plot(obs[:, 0], obs[:, 1], linestyle='--', c="#98FB98", linewidth=2)
            plt.scatter(obs[-1, 0], obs[-1, 1], s=15, c="pink", marker = '^')
            plt.plot(pred[:, 0], pred[:, 1], linestyle=':', c="#FFA07A", linewidth=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, default="../../TNT-SG/dataset/interm_data/val/raw_data_bag/")
    parser.add_argument("-d", "--dest", type=str, default="/mnt/data/SGTrain/rosbag/medium/val/")
    parser.add_argument("-s", "--small", action='store_true', default=False)
    parser.add_argument("-v", "--viz",   action='store_true', default=False)
    args = parser.parse_args()

    print("root_dir : ", args.root)
    print("save_dir : ", args.dest)

    argoverse_processor = SGPreprocessor(root_dir=args.root, split="train", save_dir=args.dest, viz=args.viz)
    loader = DataLoader(argoverse_processor,
                        batch_size=1,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)

    for i, data in enumerate(tqdm(loader, total=len(argoverse_processor), desc="Generate & saving features ")):
        if args.small and i >= 50:
            break
