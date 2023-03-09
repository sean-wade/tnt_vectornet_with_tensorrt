import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalGraphMultihead(nn.Module):
    def __init__(self, in_channels, global_graph_width=64): 
        super(GlobalGraphMultihead, self).__init__()
        self.key_weights = nn.Linear(in_channels, global_graph_width)
        self.value_weights = nn.Linear(in_channels, global_graph_width)
        self.query_weights = nn.Linear(in_channels, global_graph_width)
        self.multihead_attn = nn.MultiheadAttention(embed_dim = global_graph_width, num_heads = 1)

    def forward(self, input_var, input_mask): 
        batch_size, num_nodes, dim_in = input_var.shape
        key = input_var / (torch.sum(input_var ** 2, dim = -1, keepdim = True) ** 0.5 + 1e-6)
        query = input_var / (torch.sum(input_var ** 2, dim = -1, keepdim = True) ** 0.5 + 1e-6)
        output, attention_softmax = self.multihead_attn(query = query.transpose(0, 1), 
                                                        key = key.transpose(0, 1), 
                                                        value = input_var.transpose(0, 1), 
                                                        key_padding_mask = None)
        output = output.transpose(0, 1)
#         keys = self.key_weights(input_var)
#         values = self.value_weights(input_var)
#         query = self.query_weights(input_var)
#         attention = torch.matmul(query, keys.transpose(1, 2))
#         attention[~input_mask[:, None, :].expand(-1, num_nodes, -1)] = -float('inf')
#         attention_softmax = F.softmax(attention,dim = -1)
#         attention_softmax = attention_softmax * input_mask[:, None, :].float()
#         attention_softmax = attention_softmax / (torch.sum(attention_softmax, dim = 2, keepdim = True) + torch.finfo(torch.float).eps)
#         weighted_values = torch.matmul(attention_softmax,values)
#         weighted_values[~input_mask, :] = 0
#         output = weighted_values
        
#         print(keys.shape)
#         print(keys)
#         print(query.shape)
#         print(query)
        
#         print(attention_softmax[0, 0, :])
#         print(torch.sum(attention_softmax[0, 0, :]))
#         print(attention_softmax[0, 1, :])
#         print(torch.sum(attention_softmax[0, 1, :]))
        return output   


if __name__ == "__main__":
    batch_size = 4
    node_num = 120
    layer = GlobalGraphMultihead(in_channels=64, global_graph_width=64)

    mask = None
    feat_in = torch.randn((batch_size, node_num, 64))
    feat_out = layer(feat_in, mask)
    print("shape of feat_out: ", feat_out.size())

    # Error occured.
    EXPORT = 0
    if EXPORT:
        import onnx
        from onnxsim import simplify

        layer.eval()
        torch.onnx._export(
            layer,
            (feat_in, mask),
            "t.onnx",
            input_names=["feat_tensor", "traj_in"],
            output_names=["score"],
            dynamic_axes=None,
            opset_version=11,
        )
        print("export done.")

        # # use onnxsimplify to reduce reduent model.
        # onnx_model = onnx.load("t.onnx")
        # model_simp, check = simplify(onnx_model)
        # assert check, "Simplified ONNX model could not be validated"
        # onnx.save(model_simp, "t.onnx")
        # print("simplify done.")
