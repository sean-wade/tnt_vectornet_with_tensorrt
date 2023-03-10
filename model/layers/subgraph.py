'''
Author: zhanghao
LastEditTime: 2023-03-09 17:20:49
FilePath: /vectornet/model/layers/subgraph.py
LastEditors: zhanghao
Description: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.mlp import MLP
from torch_scatter import scatter


class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

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
                x_max = scatter(x, cluster, dim=0, reduce='max')
                x = torch.cat([x, x_max[cluster]], dim=-1)

        x = self.linear(x)
        x = scatter(x, cluster, dim=0, reduce='max')

        return F.normalize(x, p=2.0, dim=1)  # L2 normalization


if __name__ == "__main__":
    layer = SubGraph(in_channels=6, num_subgraph_layers=1, hidden_unit=64)
    data = torch.randn((11, 6))
    cluster = torch.cat((torch.zeros(6), torch.ones(5))).long()
    out = layer(data, cluster)
    print(out.shape)

    EXPORT = 0
    if EXPORT:
        import onnx
        from onnxsim import simplify

        layer.eval()
        torch.onnx._export(
            layer,
            (data, cluster),
            "t.onnx",
            input_names=["data", ],
            output_names=["output"],
            dynamic_axes=None,
            opset_version=11,
        )
        print("export done.")

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load("t.onnx")
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, "t.onnx")
        print("simplify done.")
