'''
Author: zhanghao
LastEditTime: 2023-04-13 13:46:37
FilePath: /my_vectornet_github/model/layers/scoring_and_selection.py
LastEditors: zhanghao
Description: 
'''
# score the predicted trajectories
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.mlp import MLP
# from model.loss import distance_metric


class TrajScoreSelection(nn.Module):
    def __init__(self,
                 feat_channels,
                 horizon=30,
                 hidden_dim=64,
                 temper=0.01,
                 device=torch.device("cpu")):
        """
        init trajectories scoring and selection module
        :param feat_channels: int, number of channels
        :param horizon: int, prediction horizon, prediction time x pred_freq
        :param hidden_dim: int, hidden dimension
        :param temper: float, the temperature
        """
        super(TrajScoreSelection, self).__init__()
        self.feat_channels = feat_channels
        self.horizon = horizon
        self.temper = temper

        self.device = device

        self.score_mlp = nn.Sequential(
            MLP(feat_channels + horizon * 2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat_in: torch.Tensor, traj_in: torch.Tensor):
        assert feat_in.dim() == 2, "[TrajScoreSelection]: Error in input feature dimension."
        assert traj_in.dim() == 2, "[TrajScoreSelection]: Error in candidate trajectories dimension"

        M, _ = traj_in.size()
        input_tenor = torch.cat([feat_in.repeat(M, 1), traj_in], dim=1)

        return F.softmax(self.score_mlp(input_tenor), dim=0).squeeze()

    def inference(self, feat_in: torch.Tensor, traj_in: torch.Tensor):
        """
        forward function
        :param feat_in: input feature tensor, torch.Tensor, [batch_size, feat_channels]
        :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
        :return: [batch_size, M]
        """
        return self.forward(feat_in, traj_in)

    # def loss(self, feat_in, traj_in, traj_gt):
    #     """
    #     compute loss
    #     :param feat_in: input feature, torch.Tensor, [batch_size, feat_channels]
    #     :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
    #     :param traj_gt: gt trajectories, torch.Tensor, [batch_size, horizon * 2]
    #     :return:
    #     """
    #     # batch_size = traj_in.shape[0]

    #     # compute ground truth score
    #     score_gt = F.softmax(-distance_metric(traj_in, traj_gt)/self.temper, dim=1)
    #     score_pred = self.forward(feat_in, traj_in)

    #     # return F.mse_loss(score_pred, score_gt, reduction='sum')
    #     logprobs = - torch.log(score_pred)

    #     # loss = torch.sum(torch.mul(logprobs, score_gt)) / batch_size
    #     loss = torch.sum(torch.mul(logprobs, score_gt))
    #     # if reduction == 'mean':
    #     #     loss = torch.sum(torch.mul(logprobs, score_gt)) / batch_size
    #     # else:
    #     #     loss = torch.sum(torch.mul(logprobs, score_gt))
    #     return loss

if __name__ == "__main__":
    feat_in = 64
    horizon = 30
    layer = TrajScoreSelection(feat_in, horizon)

    batch_size = 4

    feat_tensor = torch.randn((batch_size, feat_in))
    traj_in = torch.randn((batch_size, 50, horizon * 2))
    traj_gt = torch.randn((batch_size, horizon * 2))

    traj_in[:, 0, :] = traj_gt

    # forward
    score = layer(feat_tensor, traj_in)
    print("shape of score: ", score.size())

    # loss
    loss = layer.loss(feat_tensor, traj_in, traj_gt)
    print("Pass")
