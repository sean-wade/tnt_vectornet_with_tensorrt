'''
Author: zhanghao
LastEditTime: 2023-05-29 19:58:48
FilePath: /my_vectornet_github/tensorrt_deploy/tnt_trt/tnt_compare.py
LastEditors: zhanghao
Description: 
'''
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F

from model.layers.mlp import MLP
from model.layers.subgraph import SubGraph
from model.layers.global_graph import GlobalGraph

from model.layers.target_prediction import TargetPred
from model.layers.motion_etimation import MotionEstimation
from model.layers.scoring_and_selection import TrajScoreSelection


class TNTExport(nn.Module):
    def __init__(self,
                horizon=30,
                in_channels=6,
                k=6,
                m=50,
                num_subgraph_layers=3,
                num_global_graph_layer=1,
                subgraph_width=64,
                global_graph_width=64,
                target_pred_hid=64,
                motion_esti_hid=64,
                score_sel_hid=64,
                device=torch.device("cpu")
                ):
        super(TNTExport, self).__init__()
        # some params
        self.device = device
        self.k = k
        self.m = m
        self.horizon = horizon
        self.out_channels = 2
        self.num_subgraph_layers = num_subgraph_layers
        self.global_graph_width = global_graph_width
        self.target_pred_hid = target_pred_hid
        self.motion_esti_hid = motion_esti_hid
        self.score_sel_hid = score_sel_hid

        # subgraph feature extractor
        self.subgraph = SubGraph(
            in_channels, num_subgraph_layers, subgraph_width)

        # global graph
        self.global_graph = GlobalGraph(self.subgraph.out_channels + 2,
                                        self.global_graph_width,
                                        num_global_layers=num_global_graph_layer)

        self.target_pred_layer = TargetPred(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=m,
            device=device
        )
        self.motion_estimator = MotionEstimation(
            in_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=motion_esti_hid
        )
        self.traj_score_layer = TrajScoreSelection(
            feat_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=score_sel_hid,
            device=self.device
        )

    def forward(self, x, cluster, id_embedding, target_candidate):
        # print("x: \n", x)
        # print("cluster: \n", cluster)
        # print("id_embedding: \n", id_embedding)

        sub_graph_out = self.subgraph(x, cluster)
        # print("sub_graph_out: \n", sub_graph_out)

        x = torch.cat([sub_graph_out, id_embedding], dim=1).unsqueeze(0)
        global_feat = self.global_graph(x, valid_lens=None)
        target_feat = global_feat[:, 0]

        target_prob, offset = self.target_pred_layer(target_feat, target_candidate)

        _, indices = target_prob.topk(self.m, dim=0)
        target_pred_se, offset_pred_se = target_candidate[indices], offset[indices]
        target_loc_se = (target_pred_se + offset_pred_se).view(self.m, 2)

        # print("target_loc_se: \n", target_loc_se)

        trajs = self.motion_estimator(target_feat, target_loc_se)

        # print("trajs: \n", trajs.flatten()[-50:])

        scores = self.traj_score_layer(target_feat, trajs)

        # print("scores: \n", scores.flatten())

        return trajs, scores


    def load_ckpt(self, ckpt_path):
        """
        Convert trained model's state_dict and load in.
        """
        weights_dict = torch.load(ckpt_path, map_location=self.device)
        new_weights_dict = {}
        for k, v in weights_dict.items():
            if "aux_mlp" in k:
                continue
            elif "backbone." in k:
                new_k = k.replace("backbone.", "")
                new_weights_dict[new_k] = v
            else:
                new_weights_dict[k] = v

        self.load_state_dict(new_weights_dict)
        print("Success load state dict from: ", ckpt_path)


def save_weights(model, wts_file):
    import struct
    print(f'Writing into {wts_file}')
    with open(wts_file, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
            

def load_tnt(ckpt_path, num_features=6, horizon=30):
    model = TNTExport(in_channels=num_features, horizon=horizon)
    model.load_ckpt(ckpt_path)
    model.eval()
    return model


def load_txt(txt_path):
    import numpy as np
    with open(txt_path, "r") as fff:
        lines = fff.readlines()
    feats_num = int(lines[0].split("=")[-1])
    cluster_num = int(lines[1].split("=")[-1])
    candidate_num = int(lines[2].split("=")[-1])
    # print(feats_num, cluster_num, candidate_num)
    feature = []
    for n in range(feats_num):
        feature += lines[4 + n].strip().split(",")

    feature = [float(f.strip()) for f in feature if len(f)>0]
    feature = np.array(feature).reshape(-1, 6)
    # print(feature)

    id_embedding = []
    for n in range(cluster_num):
        id_embedding += lines[6 + feats_num + n].strip().split(",")
    id_embedding = np.array([float(f.strip()) for f in id_embedding if len(f)>0]).reshape(-1, 2)
    # print(id_embedding)

    cluster = lines[8 + feats_num + cluster_num].strip().split(",")
    cluster = np.array([float(f.strip()) for f in cluster if len(f)>0])
    # print(cluster)

    candidates = []
    for n in range(candidate_num):
        candidates += lines[12 + feats_num + cluster_num + n].strip().split(",")
    candidates = np.array([float(f.strip()) for f in candidates if len(f)>0]).reshape(-1, 2)
    # print(candidates)
    return feature, id_embedding, cluster, candidates


def plot_feature(feats):
    _, ax = plt.subplots(figsize=(12, 12))
    ax.axis('equal')
    pids = np.unique(feats[:,-1])
    for i, pid in enumerate(pids):
        cur_coords = feats[feats[:,-1]==pid][:,:2]
        clr = "r" if i==0 else "gold"
        ax.plot(cur_coords[:, 0], cur_coords[:, 1], marker='.', alpha=0.5, color=clr)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    ckpt = "weights/sg_best_TNT_0529.pth"

    model = load_tnt(ckpt)
    # print(model)

    test_txt = "/home/zhanghao/code/master/10_PREDICTION/TNT_VECTORNET/TrajectoryPredictionSDK/sample/debug2/ddddd.txt"
    feature, id_embedding, cluster, candidates = load_txt(test_txt)

    plot_feature(feature)

    x = torch.from_numpy(feature).float()
    cluster = torch.from_numpy(cluster).long()
    id_embedding = torch.from_numpy(id_embedding).float()
    target_candidate = torch.from_numpy(candidates).float()
    # print(target_candidate.shape)

    # import numpy as np
    # np.savetxt("candidate.txt", target_candidate.flatten().reshape(1,-1).numpy(), fmt="%.1f", delimiter=",")

    model.cuda()
    with torch.no_grad():
        trajs, scores = model(x.cuda(), 
                              cluster.cuda(), 
                              id_embedding.cuda(), 
                              target_candidate.cuda())
        # print(trajs[:2].reshape(-1,30,2))
        # # print(trajs.shape)
        # print(scores[:2])
        # # print(scores.shape)

        for ss, tj in zip(scores, trajs):
            print(float(ss), ": ", tj)

