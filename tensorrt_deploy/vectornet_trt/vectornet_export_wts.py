'''
Author: zhanghao
LastEditTime: 2023-03-24 17:24:05
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
        # print("sub_graph_out: \n", sub_graph_out, "\n\n\n")

        x = torch.cat([sub_graph_out, id_embedding], dim=1).unsqueeze(0)
        global_feat = self.global_graph(x, valid_lens=None)

        # print("global_feat[:, 0]: \n", global_feat[:, 0], "\n")
        # print("global_feat[:, 0].shape: \n", global_feat.shape, "\n\n")

        pred = self.traj_pred_mlp(global_feat[:, 0])
        # pred = pred.view(self.horizon, 2).cumsum(0)
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
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    ckpt = "work_dir/vectornet/03_10_20_43/best_VectorNet.pth"
    wts = "tensorrt_deploy/src/data/vectornet/vectornet.wts"
    test_pkl = "tensorrt_deploy/src/data/data_seq_40050_features.pkl"

    model = load_vectornet(ckpt)
    # for k,v in model.state_dict().items():
    #     print(k)

    test_data = pickle.load(open(test_pkl, "rb"))
    x = test_data["x"]
    cluster = test_data["cluster"].long()
    id_embedding = test_data["identifier"]

    print(test_data["lane_num"])
    print(test_data["traj_num"])
    print(x.shape)

    cluster_count = torch.unique(cluster, return_counts=True)[1]
    
    # x = torch.tensor([-0.3801, -0.1300, 1.1666,  -1.1327, 0.6438, 0.6729,  -1.1299, -2.2857, 0.1849,  0.0493,
    #                 -0.4179, -0.5331, 0.7467,  -1.0006, 1.4848, 0.2771,  0.1393,  -0.9162, -1.7744, 0.8850,
    #                 -1.6748, 1.3581,  -0.4987, -0.7244, 0.7941, -0.4109, -0.3446, -0.5246, -0.8153, -0.5685,
    #                 1.9105,  -0.1069, 0.7214,  0.5255,  0.3654, -0.3434, 0.7163,  -0.6460, 1.9680,  0.8964,
    #                 0.3845,  3.4347,  -2.6291, -0.9330, 0.6411, 0.9983,  0.6731,  0.9110,  -2.0634, -0.5751,
    #                 1.4070,  0.5285,  -0.1171, -0.1863, 2.1200, 1.3745,  0.9763,  -0.1193, -0.3343, -1.5933]).reshape(-1,6)
    # cluster = torch.tensor([0, 1, 1, 2, 2, 3, 3, 3, 3, 4])
    # id_embedding = torch.randn((5,2))

    # print("id_embedding = \n", id_embedding)
    # print("x = \n", x.reshape(1,-1))

    out = model(x, cluster, id_embedding)
    # print(out.shape)
    print(out.reshape(-1,2))

    import numpy as np
    np.savetxt("/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/feature.txt", x.reshape(1,-1).detach().cpu().numpy(), delimiter=",")
    np.savetxt("/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/cluster.txt", cluster.reshape(1,-1).detach().cpu().numpy(), delimiter=",")
    np.savetxt("/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/id_embedding.txt", id_embedding.reshape(1,-1).detach().cpu().numpy(), delimiter=",")
    np.savetxt("/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/cluster_count.txt", cluster_count.reshape(1,-1).detach().cpu().numpy(), delimiter=",")
    np.savetxt("/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/out.txt", out.reshape(1,-1).detach().cpu().numpy(), delimiter=",")

    # gt = test_data["y"].reshape(30, 2).cumsum(0)
    # # print(gt)
    # print(gt[-1] - out[-1])

    save_weights(model, wts)



