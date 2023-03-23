'''
Author: zhanghao
LastEditTime: 2023-03-23 16:02:18
FilePath: /vectornet/model/layers/mlp.py
LastEditors: zhanghao
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# MLP
class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden=64, bias=True, activation="relu", norm='layer'):
        super(MLP, self).__init__()

        # define the activation function
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "relu6":
            act_layer = nn.ReLU6
        elif activation == "leaky":
            act_layer = nn.LeakyReLU
        elif activation == "prelu":
            act_layer = nn.PReLU
        else:
            raise NotImplementedError

        # define the normalization function
        if norm == "layer":
            norm_layer = nn.LayerNorm
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d
        else:
            raise NotImplementedError

        # insert the layers
        self.linear1 = nn.Linear(in_channel, hidden, bias=bias)
        self.linear1.apply(self._init_weights)
        self.linear2 = nn.Linear(hidden, out_channel, bias=bias)
        self.linear2.apply(self._init_weights)

        self.norm1 = norm_layer(hidden)
        self.norm2 = norm_layer(out_channel)

        self.act1 = act_layer(inplace=True)
        self.act2 = act_layer(inplace=True)

        self.shortcut = None
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=bias),
                norm_layer(out_channel)
            )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
        return self.act2(out)


if __name__ == "__main__":
    batch_size = 256
    in_feat = 10
    out_feat = 64
    in_tensor = torch.randn((batch_size, 30, 10, in_feat), dtype=torch.float).cuda()

    mlp = MLP(in_feat, out_feat).cuda()
    print(mlp)

    out = mlp(in_tensor)
    print(in_tensor.shape)
    print(out.shape)
    print("--------")

    in_tensor2 = torch.randn((30, 10, in_feat), dtype=torch.float).cuda()
    mlp = MLP(in_feat, out_feat).cuda()
    out2 = mlp(in_tensor2)
    print(in_tensor2.shape)
    print(out2.shape)

