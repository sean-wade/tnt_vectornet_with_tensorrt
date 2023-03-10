'''
Author: zhanghao
LastEditTime: 2023-03-10 16:26:19
FilePath: /vectornet/model/vectornet.py
LastEditors: zhanghao
Description: 
'''
import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader, Batch, Data

# from core.model.layers.global_graph import GlobalGraph, SelfAttentionFCLayer
from model.layers.global_graph import GlobalGraph
from model.layers.subgraph import SubGraph
from model.layers.mlp import MLP
from model.backbone.vectornet_backbone import VectorNetBackbone


class VectorNet(nn.Module):
    def __init__(self,
                 in_channels=8,
                 horizon=30,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 traj_pred_mlp_width=64,
                 with_aux = False,
                 device=torch.device("cpu")):
        super(VectorNet, self).__init__()
        # some params
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.out_channels = 2
        self.horizon = horizon
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.k = 1

        self.device = device

        # subgraph feature extractor
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layers=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            device=device
        )

        # pred mlp
        self.traj_pred_mlp = nn.Sequential(
            MLP(global_graph_width, traj_pred_mlp_width, traj_pred_mlp_width),
            nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels)
        )

    def forward(self, data):
        """
        args:
            data (Data): list(batch_data: dict)
        """
        if self.training:
            batch_preds = []
            batch_global_feat, batch_aux_out, batch_aux_gt = self.backbone(data)
            for global_feat in batch_global_feat:
                target_feat = global_feat[:, 0]
                pred = self.traj_pred_mlp(target_feat)
                batch_preds.append(pred)

            return {"pred": batch_preds, "aux_out" : batch_aux_out, "aux_gt" : batch_aux_gt}
        else:
            batch_preds = []
            batch_global_feat = self.backbone(data)
            for global_feat in batch_global_feat:
                target_feat = global_feat[:, 0]
                pred = self.traj_pred_mlp(target_feat)
                batch_preds.append(pred)
            return batch_preds

    def inference(self, data):
        batch_preds = self.forward(data)
        batch_pred_traj = [pred.view(self.k, self.horizon, 2).cumsum(1) for pred in batch_preds]
        return batch_pred_traj


if __name__ == "__main__":
    from dataset.sg_dataloader import SGTrajDataset, collate_list, collate_list_cuda
    from torch.utils.data import Dataset, DataLoader

    dataset = SGTrajDataset(data_root='/mnt/data/SGTrain/rosbag/train_feature', in_mem=True)
    loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, collate_fn=collate_list_cuda)

    device = torch.device('cuda:0')
    model = VectorNet(in_channels=6, device=device, with_aux=True)
    model.to(device)
    # model.eval()

    # def data_to_device(data, device):
    #     for i, b_data in enumerate(data):
    #         for k, v in data[i].items():
    #             if torch.is_tensor(v):
    #                 data[i][k] = data[i][k].to(device)
    #     return data

    import time
    s = time.time()
    for i in range(10):
        for data in loader:
            # data = data_to_device(data, device)
            output = model(data)
            # output = model.inference(data)

            # training
            # for pred in output['pred']:
            #     print(pred.shape)
            # for aux_out in output['aux_out']:
            #     print(aux_out.shape)

            ## val
            # for out in output:
            #     print(out[0].shape)
            # print("--------")
        print("epoch %d done..."%i)
    e = time.time()
    print("using %.2f s"%(e - s))
    