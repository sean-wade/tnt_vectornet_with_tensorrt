'''
Author: zhanghao
LastEditTime: 2023-06-07 19:59:19
FilePath: /my_vectornet_github/tools/analysis_anchor.py
LastEditors: zhanghao
Description: 
'''
import os
import glob
import pickle
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})


root = "/mnt/data/SGTrain/TRAJ_DATASET/EXP5_0427_0516_BALANCE/"
K_ANCHORS = 50

for split in ["train", "val"]:
# for split in ["val"]:
    fs = glob.glob(root + "/%s/*.pkl"%split)

    target_points = []
    for ff in tqdm(fs, desc="Counting split : %s"%split):
        with open(ff, "rb") as ooo:
            dd = pickle.load(ooo)
            target_points.append(dd["y"].view(-1, 2).cumsum(axis=0).cpu().numpy()[-1])

    print(len(target_points))

    x = pd.DataFrame(np.array(target_points), columns=['x', 'y'])
    sns.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.grid(alpha=0.2)
    plt.savefig(root + '/%s_xy_correlogram.jpg'%split, dpi=200)
    plt.close()
    
    print("Start k-means++")
    k_means = KMeans(init='k-means++', n_clusters=K_ANCHORS, n_init=10)
    k_means.fit(target_points)
    # k_means_labels = k_means.labels_
    k_means_cluster_centers = np.array(k_means.cluster_centers_)
    # k_means_labels_unique = np.unique(k_means_labels)
    print("Finishe k-means++")
    print("k_means_cluster_centers : \n", k_means_cluster_centers)

    for k in range(K_ANCHORS):
        cluster_center = k_means_cluster_centers[k]
        plt.plot(cluster_center[0], cluster_center[1], '.', markersize=6)

    plt.title('KMeans')    
    plt.grid(True)
    plt.savefig(root + '/%s_clusters.jpg'%split, dpi=200)
    # plt.show()

    np.savetxt(root + "/%s_candidate.txt"%split, k_means_cluster_centers, fmt="%.1f")