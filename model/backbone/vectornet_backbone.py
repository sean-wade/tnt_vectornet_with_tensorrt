import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn

from model.layers.mlp import MLP
from model.layers.subgraph import SubGraph
from model.layers.global_graph import GlobalGraph


class VectorNetBackbone(nn.Module):
    def __init__(self,
                 in_channels=8,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 aux_mlp_width=64,
                 with_aux=False,
                 device=torch.device("cpu")
                 ):
        super(VectorNetBackbone, self).__init__()
        # some params
        self.num_subgraph_layers = num_subgraph_layers
        self.global_graph_width = global_graph_width
        self.device = device

        # subgraph feature extractor
        self.subgraph = SubGraph(
            in_channels, num_subgraph_layers, subgraph_width)

        # global graph
        self.global_graph = GlobalGraph(self.subgraph.out_channels + 2,
                                        self.global_graph_width,
                                        num_global_layers=num_global_graph_layer)

        # auxiliary recoverey mlp
        self.with_aux = with_aux
        if self.with_aux:
            self.aux_mlp = nn.Sequential(
                MLP(self.global_graph_width, aux_mlp_width, aux_mlp_width),
                nn.Linear(aux_mlp_width, self.subgraph.out_channels)
            )

    def forward(self, data):
        """
        args:
            data (Data): list(batch_data: dict)
        """
        batch_output, batch_aux_gt, batch_aux_out = [], [], []
        for batch_idx, batch_data in enumerate(data):
            id_embedding = batch_data["identifier"]
            valid_len = batch_data["traj_num"] + batch_data["lane_num"]

            sub_graph_out = self.subgraph(batch_data["x"], batch_data["cluster"].long())

            if self.training and self.with_aux:
                mask_polyline_indices = torch.randint(1, valid_len-2, (1,))
                aux_gt = sub_graph_out[mask_polyline_indices]
                sub_graph_out[mask_polyline_indices] = 0.0
                batch_aux_gt.append(aux_gt)

            x = torch.cat([sub_graph_out, id_embedding], dim=1).unsqueeze(0)
            global_graph_out = self.global_graph(x, valid_lens=None)

            if self.training and self.with_aux:
                aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices].to(self.device)
                aux_out = self.aux_mlp(aux_in)
                batch_aux_out.append(aux_out)

            batch_output.append(global_graph_out)

        if self.training:
            return batch_output, batch_aux_gt, batch_aux_out
        else:
            return batch_output


if __name__ == "__main__":
    from dataset.sg_dataloader import SGTrajDataset, collate_list
    from torch.utils.data import Dataset, DataLoader

    dataset = SGTrajDataset(data_root='/mnt/data/SGTrain/rosbag/train_feature', in_mem=True)
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, collate_fn=collate_list)

    device = torch.device('cuda:0')
    model = VectorNetBackbone(in_channels=6, device=device, with_aux=True)
    model.eval()

    for data in loader:
        print(data[0]["seq_id"], data[1]["seq_id"])
        output = model(data)

        ## training
        # print(output[0][0].shape)
        # print(output[1][0].shape)
        # print(output[2][0].shape)

        ## val
        print(output[0].shape)
        print(output[1].shape)
        print("--------")
