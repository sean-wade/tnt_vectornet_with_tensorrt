'''
Author: zhanghao
LastEditTime: 2023-03-15 11:37:21
FilePath: /vectornet/tools/export/vectornet_export_v1.py
LastEditors: zhanghao
Description: 
    CustomScatterMax: Custom implement, just for export onnx. Cannot inference by onnxruntime, need implemention.
'''
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from model.layers.mlp import MLP
# from model.layers.subgraph import SubGraph
from model.layers.global_graph import GlobalGraph


class CustomScatterMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index):
        # 调 unique 仅为了输出对应的维度信息
        index_unique = torch.unique(index)
        out = torch.zeros((index_unique.shape[0], src.shape[1]), dtype=torch.float32, device=src.device)
        for idx in index_unique:
            out[idx] = torch.max(src[index==idx], dim=0)[0]
        return out

    @staticmethod
    def symbolic(g, src, index):
        return g.op("Custom::ScatterMaxPlugin", src, index)


# from torch.onnx.symbolic_registry import register_op 
# register_op('ScatterMaxPlugin', CustomScatterMax, '', 11)


class SubGraph(nn.Module):
    def __init__(self, in_channels, num_subgraph_layers=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.hidden_unit = hidden_unit
        self.out_channels = hidden_unit

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'glp_{i}', MLP(in_channels, hidden_unit, hidden_unit))
            in_channels = hidden_unit * 2

        self.linear = nn.Linear(hidden_unit * 2, hidden_unit)

    def forward(self, x, cluster): 
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                x_max = CustomScatterMax.apply(x, cluster)
                x = torch.cat([x, x_max[cluster]], dim=-1)

        x = self.linear(x)
        x = CustomScatterMax.apply(x, cluster)

        return F.normalize(x, p=2.0, dim=1)  # L2 normalization


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


if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = VectorNetExport(in_channels=6, device=device)
    ckpt = "/home/zhanghao/code/master/10_PREDICTION/VectorNet/vectornet/work_dir/vectornet/03_10_20_43/best_VectorNet.pth"
    model.load_ckpt(ckpt)
    model.eval()

    # x = torch.randn((200, 6))
    # cluster = torch.cat((torch.zeros(50), torch.ones(70), torch.ones(40)*2, torch.ones(40)*3)).long()
    # id_embedding = torch.randn((int(cluster.max())+1,2))
    # print(id_embedding.shape)

    test_pkl = "tools/export/data_seq_40050_features.pkl"
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
    # [ 0.6829, -0.2413]

    ONNX_EXPORT = 1
    if ONNX_EXPORT:
        import onnx
        from onnxsim import simplify

        model.eval()
        torch.onnx._export(
            model,
            (x, cluster, id_embedding),
            "tools/export/models/fake_vectornet.onnx",
            input_names=["feat_tensor", "cluster", "id_embedding"],
            output_names=["pred"],
            dynamic_axes=None,
            opset_version=11,
        )
        print("export done.")

        # import onnxruntime
        # sess = onnxruntime.InferenceSession("tools/export/models/fake_vectornet.onnx", providers='TensorrtExecutionProvider') 
        # ort_output = sess.run(None, {'0': x.numpy(), '1' : cluster.numpy(), '2' : id_embedding.numpy()})[0]