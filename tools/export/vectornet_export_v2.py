'''
Author: zhanghao
LastEditTime: 2023-03-21 16:23:22
FilePath: /vectornet/tools/export/vectornet_export_v2.py
LastEditors: zhanghao
Description: 
    sub_graph 中的 scatter_max 改为了 scatter_reduce
    经过测试，目前可以使用 pytorch > 1.12.0 导出 traced pytorch 模型，onnx 无法导出
'''
from numpy import poly
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

from model.layers.mlp import MLP
# from model.layers.subgraph import SubGraph
from model.layers.global_graph import GlobalGraph


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

    def forward(self, x, cluster, poly_num): 
        poly_max_feats = torch.zeros((poly_num, 64))
        cluster_index = cluster.unsqueeze(1).repeat((1, 64))
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                # x_max = scatter(x, cluster, dim=0, reduce='max')
                x_max = poly_max_feats.scatter_reduce(0, cluster_index, x, reduce="amax", include_self=False)
                
                x = torch.cat([x, x_max[cluster]], dim=-1)

        x = self.linear(x)
        # x = scatter(x, cluster, dim=0, reduce='max')
        x = poly_max_feats.scatter_reduce(0, cluster_index, x, reduce="amax", include_self=False)

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
    def forward(self, x, cluster, id_embedding, poly_num):
        """
        args:
            data (Data): list(batch_data: dict)
        """
        # valid_len = traj_num + lane_num
        sub_graph_out = self.subgraph(x, cluster, poly_num)

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
    poly_num = torch.tensor(test_data["traj_num"] + test_data["lane_num"]).int()
    cluster = test_data["cluster"].long()
    id_embedding = test_data["identifier"]

    out = model(x, cluster, id_embedding, poly_num)
    print(out.shape)
    # print(out)

    gt = test_data["y"].reshape(30, 2).cumsum(0)
    # print(gt)
    print(gt[-1] - out[-1])
    print("Inference done!!! \n\n\n")

    TRACE_JIT_EXPORT = 0
    if TRACE_JIT_EXPORT:
        print("Start jit tracing...")
        traced_script_module = torch.jit.trace(model, (x, cluster, id_embedding, poly_num))
        out2 = traced_script_module(x, cluster, id_embedding, poly_num)
        print(gt[-1] - out2[-1])
        traced_script_module.save("tools/export/models/traced_vectornet.pt")
        print("TRACE_JIT_EXPORT done!!! \n\n\n")

    # ONNX_EXPORT = 1
    # if ONNX_EXPORT:
    #     import onnx
    #     from onnxsim import simplify

    #     model.eval()
    #     torch.onnx._export(
    #         model,
    #         (x, cluster, id_embedding, poly_num),
    #         "t.onnx",
    #         input_names=["feat_tensor", "cluster", "id_embedding", "poly_num"],
    #         output_names=["pred"],
    #         dynamic_axes=None,
    #         opset_version=16,
    #     )
    #     print("export done.")

    #     # use onnxsimplify to reduce reduent model.
    #     onnx_model = onnx.load("t.onnx")
    #     model_simp, check = simplify(onnx_model)
    #     assert check, "Simplified ONNX model could not be validated"
    #     onnx.save(model_simp, "t.onnx")
    #     print("simplify done.")









"""
Jit traced:
RecursiveScriptModule(
  original_name=VectorNetExport
  (subgraph): RecursiveScriptModule(
    original_name=SubGraph
    (layer_seq): RecursiveScriptModule(
      original_name=Sequential
      (glp_0): RecursiveScriptModule(
        original_name=MLP
        (linear1): RecursiveScriptModule(original_name=Linear)
        (linear2): RecursiveScriptModule(original_name=Linear)
        (norm1): RecursiveScriptModule(original_name=LayerNorm)
        (norm2): RecursiveScriptModule(original_name=LayerNorm)
        (act1): RecursiveScriptModule(original_name=ReLU)
        (act2): RecursiveScriptModule(original_name=ReLU)
        (shortcut): RecursiveScriptModule(
          original_name=Sequential
          (0): RecursiveScriptModule(original_name=Linear)
          (1): RecursiveScriptModule(original_name=LayerNorm)
        )
      )
      (glp_1): RecursiveScriptModule(
        original_name=MLP
        (linear1): RecursiveScriptModule(original_name=Linear)
        (linear2): RecursiveScriptModule(original_name=Linear)
        (norm1): RecursiveScriptModule(original_name=LayerNorm)
        (norm2): RecursiveScriptModule(original_name=LayerNorm)
        (act1): RecursiveScriptModule(original_name=ReLU)
        (act2): RecursiveScriptModule(original_name=ReLU)
        (shortcut): RecursiveScriptModule(
          original_name=Sequential
          (0): RecursiveScriptModule(original_name=Linear)
          (1): RecursiveScriptModule(original_name=LayerNorm)
        )
      )
      (glp_2): RecursiveScriptModule(
        original_name=MLP
        (linear1): RecursiveScriptModule(original_name=Linear)
        (linear2): RecursiveScriptModule(original_name=Linear)
        (norm1): RecursiveScriptModule(original_name=LayerNorm)
        (norm2): RecursiveScriptModule(original_name=LayerNorm)
        (act1): RecursiveScriptModule(original_name=ReLU)
        (act2): RecursiveScriptModule(original_name=ReLU)
        (shortcut): RecursiveScriptModule(
          original_name=Sequential
          (0): RecursiveScriptModule(original_name=Linear)
          (1): RecursiveScriptModule(original_name=LayerNorm)
        )
      )
    )
    (linear): RecursiveScriptModule(original_name=Linear)
  )
  (global_graph): RecursiveScriptModule(
    original_name=GlobalGraph
    (layers): RecursiveScriptModule(
      original_name=Sequential
      (glp_0): RecursiveScriptModule(
        original_name=SelfAttentionFCLayer
        (q_lin): RecursiveScriptModule(original_name=Linear)
        (k_lin): RecursiveScriptModule(original_name=Linear)
        (v_lin): RecursiveScriptModule(original_name=Linear)
      )
    )
  )
  (traj_pred_mlp): RecursiveScriptModule(
    original_name=Sequential
    (0): RecursiveScriptModule(
      original_name=MLP
      (linear1): RecursiveScriptModule(original_name=Linear)
      (linear2): RecursiveScriptModule(original_name=Linear)
      (norm1): RecursiveScriptModule(original_name=LayerNorm)
      (norm2): RecursiveScriptModule(original_name=LayerNorm)
      (act1): RecursiveScriptModule(original_name=ReLU)
      (act2): RecursiveScriptModule(original_name=ReLU)
    )
    (1): RecursiveScriptModule(original_name=Linear)
  )
)
m.code
def forward(self,
    x: Tensor,
    cluster: Tensor,
    id_embedding: Tensor,
    poly_num: Tensor) -> Tensor:
  traj_pred_mlp = self.traj_pred_mlp
  global_graph = self.global_graph
  subgraph = self.subgraph
  _0 = (subgraph).forward(int(poly_num), cluster, x, )
  input = torch.unsqueeze(torch.cat([_0, id_embedding], 1), 0)
  _1 = torch.slice((global_graph).forward(input, ), 0, 0, 9223372036854775807)
  input0 = torch.select(_1, 1, 0)
  _2 = torch.view((traj_pred_mlp).forward(input0, ), [30, 2])
  return torch.cumsum(_2, 0)
"""