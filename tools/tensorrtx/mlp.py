'''
Author: zhanghao
LastEditTime: 2023-03-20 16:07:40
FilePath: /vectornetx/mlp.py
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


# MLP
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

        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
            
        return self.act2(out)


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    mlp = MLP()

    # inp1 = torch.randn((1, 4, 6))
    # out1 = mlp(inp1)
    # print(out1.shape)
    
    # inp2 = torch.randn((500, 6))
    inp2 = torch.tensor([[0.0, 1, 2, 3, 4, 5], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]])

    import time
    s = time.time()
    
    for i in range(1000):
        out2 = mlp(inp2)

    e = time.time()
    print("using %.5f ms"%((e-s) * 1000))
    print(out2)

    # import numpy as np
    # np.savetxt("data/mlp2/inp1.txt", inp1.squeeze().cpu().detach().numpy())
    # np.savetxt("data/mlp2/out1.txt", out1.squeeze().cpu().detach().numpy())
    # np.savetxt("data/mlp2/inp2.txt", inp2.cpu().detach().numpy())
    # np.savetxt("data/mlp2/out2.txt", out2.cpu().detach().numpy())
    
    import struct
    wts_file = "/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/mlp/my_mlp.wts"
    print(f'Writing into {wts_file}')
    with open(wts_file, 'w') as f:
        f.write('{}\n'.format(len(mlp.state_dict().keys())))
        for k, v in mlp.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
