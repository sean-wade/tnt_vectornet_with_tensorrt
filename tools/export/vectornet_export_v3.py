'''
Author: zhanghao
LastEditTime: 2023-03-21 16:18:52
FilePath: /vectornet/tools/export/vectornet_export_v3.py
LastEditors: zhanghao
Description: 
    TRT 移植的工具代码
    作用：
        将 vectornet 的 state_dict 导出为 weights file, 使用 tensorrt 搭载网络并 load weights file.
'''
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from model.layers.mlp import MLP
from model.layers.subgraph import SubGraph
from model.layers.global_graph import GlobalGraph


class VectorNetExport(nn.Module):
    def __init__(self,
                horizon=30,
                in_channels=6,
                num_subgraph_layers=3,
                num_global_graph_layer=1,
                subgraph_width=64,
                global_graph_width=64,
                traj_pred_mlp_width=64,
                device=torch.device("cpu")
                ):
        super(VectorNetExport, self).__init__()
        # some params
        self.num_subgraph_layers = num_subgraph_layers
        self.global_graph_width = global_graph_width
        self.device = device
        self.k = 1
        self.out_channels = 2
        self.horizon = horizon

        # subgraph feature extractor
        self.subgraph = SubGraph(
            in_channels, num_subgraph_layers, subgraph_width)

        # global graph
        self.global_graph = GlobalGraph(self.subgraph.out_channels + 2,
                                        self.global_graph_width,
                                        num_global_layers=num_global_graph_layer)

        self.traj_pred_mlp = nn.Sequential(
            MLP(global_graph_width, traj_pred_mlp_width, traj_pred_mlp_width),
            nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels)
        )

    # def forward(self, x, cluster, id_embedding, traj_num, lane_num):
    def forward(self, x, cluster, id_embedding):
        """
        args:
            data (Data): list(batch_data: dict)
        """
        # valid_len = traj_num + lane_num
        sub_graph_out = self.subgraph(x, cluster)

        x = torch.cat([sub_graph_out, id_embedding], dim=1).unsqueeze(0)
        global_feat = self.global_graph(x, valid_lens=None)
        pred = self.traj_pred_mlp(global_feat[:, 0])
        pred = pred.view(self.horizon, 2).cumsum(0)
        return pred

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
            

def load_vectornet(ckpt_path, num_features=6, horizon=30):
    model = VectorNetExport(in_channels=num_features, horizon=horizon)
    model.load_ckpt(ckpt_path)
    model.eval()
    return model


if __name__ == "__main__":
    ckpt = "/home/zhanghao/code/master/10_PREDICTION/VectorNet/vectornet/work_dir/vectornet/03_10_20_43/best_VectorNet.pth"
    wts = "/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/vectornet/vectornet.wts"
    test_pkl = "tools/export/data_seq_40050_features.pkl"

    model = load_vectornet(ckpt)

    test_data = pickle.load(open(test_pkl, "rb"))
    x = test_data["x"]
    cluster = test_data["cluster"].long()
    id_embedding = test_data["identifier"]

    out = model(x, cluster, id_embedding)
    print(out.shape)
    # print(out)

    gt = test_data["y"].reshape(30, 2).cumsum(0)
    # print(gt)
    print(gt[-1] - out[-1])

    save_weights(model, wts)



