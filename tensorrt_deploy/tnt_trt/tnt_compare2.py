'''
Author: zhanghao
LastEditTime: 2023-05-29 19:49:01
FilePath: /my_vectornet_github/tensorrt_deploy/tnt_trt/tnt_compare2.py
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
from model.tnt import TNT


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

    model = TNT(in_channels=6)

    test_txt = "/home/zhanghao/code/master/10_PREDICTION/TNT_VECTORNET/TrajectoryPredictionSDK/sample/debug/19_ab084e19-8172-4a3d-adbe-d8d3025378cc_input.txt"
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

