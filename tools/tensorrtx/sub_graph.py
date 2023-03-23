'''
Author: zhanghao
LastEditTime: 2023-03-22 20:25:19
FilePath: /vectornetx/sub_graph.py
LastEditors: zhanghao
Description: 

    MLP(
        (linear1): Linear(in_features=6, out_features=64, bias=True)
        (linear2): Linear(in_features=64, out_features=64, bias=True)
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (act1): ReLU(inplace=True)
        (act2): ReLU(inplace=True)
        (shortcut): Sequential(
            (0): Linear(in_features=6, out_features=64, bias=True)
            (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        )
    )
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class MLP(nn.Module):
    def __init__(self, in_channel=6, out_channel=64, hidden=64, bias=True):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channel, hidden, bias=bias)
        self.linear2 = nn.Linear(hidden, out_channel, bias=bias)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(out_channel)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)

        self.shortcut = None
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=bias),
                nn.LayerNorm(out_channel)
            )

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.norm2(out)
        # print("shortcut = ", self.shortcut)
        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
            
        return self.act2(out)


class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels=6, num_subgraph_layers=3, hidden_unit=64):
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
        # print(x)
        # print(cluster)
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                # print(x)
                x_max = scatter(x, cluster, dim=0, reduce='max')
                x = torch.cat([x, x_max[cluster]], dim=-1)

        x = self.linear(x)
        x = scatter(x, cluster, dim=0, reduce='max')
        print(x)
        print("\n\n\n")
        return F.normalize(x, p=2.0, dim=1)  # L2 normalization


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    sub_graph = SubGraph(num_subgraph_layers=3)

    # x = torch.tensor([[0.0, 1, 2, 3, 4, 5], 
    #                     [10.0, 11.0, 12.0, 13.0, 14.0, 15.0], 
    #                     [121, 132, 103, 114, 105, 135], 
    #                     [3.0, 11, 22, 33, 44, 55], 
    #                     [20.0,21, 32, 43, 14, 25]])

    x = torch.randn((10, 6))
    cluster = torch.tensor([0, 1, 1, 2, 2, 3, 3, 3, 3, 4])

    print(x.reshape(1,-1))

    import time
    s = time.time()
    
    for i in range(1):
        out2 = sub_graph(x, cluster)

    e = time.time()
    print("using %.5f ms"%((e-s) * 1000))
    print(out2.shape)
    for out in out2:
        print(out)

    # import numpy as np
    # np.savetxt("data/mlp2/inp1.txt", inp1.squeeze().cpu().detach().numpy())
    # np.savetxt("data/mlp2/out1.txt", out1.squeeze().cpu().detach().numpy())
    # np.savetxt("data/mlp2/inp2.txt", inp2.cpu().detach().numpy())
    # np.savetxt("data/mlp2/out2.txt", out2.cpu().detach().numpy())
    
    import struct
    wts_file = "/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/sub_graph/sub_graph.wts"
    print(f'Writing into {wts_file}')
    with open(wts_file, 'w') as f:
        f.write('{}\n'.format(len(sub_graph.state_dict().keys())))
        for k, v in sub_graph.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
