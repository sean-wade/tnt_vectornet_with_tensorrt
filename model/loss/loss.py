'''
Author: zhanghao
LastEditTime: 2023-04-13 18:46:27
FilePath: /my_vectornet_github/model/loss/loss.py
LastEditors: zhanghao
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def distance_metric(traj_candidate: torch.Tensor, traj_gt: torch.Tensor):
    """
    compute the distance between the candidate trajectories and gt trajectory
        :param traj_candidate: torch.Tensor, [batch_size, M, horizon * 2] or [M, horizon * 2]
        :param traj_gt: torch.Tensor, [batch_size, horizon * 2] or [1, horizon * 2]
        :return: distance, torch.Tensor, [batch_size, M] or [1, M]
    """
    assert traj_gt.dim() == 2, "Error dimension in ground truth trajectory"
    if traj_candidate.dim() == 3:
        # batch case
        pass

    elif traj_candidate.dim() == 2:
        traj_candidate = traj_candidate.unsqueeze(1)
    else:
        raise NotImplementedError

    assert traj_candidate.size()[2] == traj_gt.size()[1], "Miss match in prediction horizon!"

    _, M, horizon_2_times = traj_candidate.size()
    dis = torch.pow(traj_candidate - traj_gt.unsqueeze(1), 2).view(-1, M, int(horizon_2_times / 2), 2)

    dis, _ = torch.max(torch.sum(dis, dim=3), dim=2)

    return dis


class VectorLoss(nn.Module):
    """
        The loss function for train vectornet, Loss = L_traj + alpha * L_node
        where L_traj is the negative Gaussian log-likelihood loss, L_node is the huber loss
    """
    def __init__(self, alpha=1.0, aux_loss=False, reduction='sum'):
        super(VectorLoss, self).__init__()

        self.alpha = alpha
        self.aux_loss = aux_loss
        if reduction in ["mean", "sum"]:
            self.reduction = reduction
        else:
            raise NotImplementedError("[VectorLoss]: The reduction has not been implemented!")

    def forward(self, pred, gt, aux_pred=None, aux_gt=None):
        batch_size = pred.size()[0]
        loss = 0.0

        l_traj = F.mse_loss(pred, gt, reduction='sum')
        # vars = torch.ones_like(pred) * 0.5
        # l_traj = F.gaussian_nll_loss(pred, gt, vars, reduction="sum")
        if self.reduction == 'mean':
            l_traj /= batch_size

        loss += l_traj
        if self.aux_loss:
            # return nll loss if pred is None
            if not isinstance(aux_pred, torch.Tensor) or not isinstance(aux_gt, torch.Tensor):
                return loss
            assert aux_pred.size() == aux_gt.size(), "[VectorLoss]: The dim of prediction and ground truth don't match!"

            l_node = F.smooth_l1_loss(aux_pred, aux_gt, reduction="sum")
            if self.reduction == 'mean':
                l_node /= batch_size
            loss += self.alpha * l_node
        return loss


class TNTLoss(nn.Module):
    """
        The loss function for train TNT, loss = a1 * Targe_pred_loss + a2 * Traj_reg_loss + a3 * Score_loss
    """
    def __init__(self,
                 lambda1,
                 lambda2,
                 lambda3,
                 m,
                 k,
                 temper=0.01,
                 aux_loss=False,
                 reduction='sum',
                 device=torch.device("cpu")):
        """
        lambda1, lambda2, lambda3: the loss coefficient;
        temper: the temperature for computing the score gt;
        aux_loss: with the auxiliary loss or not;
        reduction: loss reduction, "sum" or "mean" (batch mean);
        """
        super(TNTLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.m = m
        self.k = k

        self.aux_loss = aux_loss
        self.reduction = reduction
        self.temper = temper

        self.device = device

    def forward(self, pred_dict, gt_dict, aux_pred=None, aux_gt=None):
        """
            pred_dict: the dictionary containing model prediction,
                {
                    "target_prob":  the predicted probability of each target candidate,
                    "offset":       the predicted offset of the target position from the gt target candidate,
                    "traj_with_gt": the predicted trajectory with the gt target position as the input,
                    "traj":         the predicted trajectory without the gt target position,
                    "score":        the predicted score for each predicted trajectory,
                }
            gt_dict: the dictionary containing the prediction gt,
                {
                    "target_prob":  the one-hot gt of traget candidate;
                    "offset":       the gt for the offset of the nearest target candidate to the target position;
                    "y":            the gt trajectory of the target agent;
                }
        """
        batch_size = len(gt_dict['target_prob'])
        loss = 0.0

        cls_loss_total, offset_loss_total, reg_loss_total, score_loss_total, aux_loss_total = 0.0, 0.0, 0.0, 0.0, 0.0
        for bb in range(batch_size):
            cls_loss = F.binary_cross_entropy(
                pred_dict[bb]['target_prob'], gt_dict['target_prob'][bb].float(), reduction='none')
            cls_loss = cls_loss.sum()
            cls_loss_total += cls_loss

            gt_idx = gt_dict['target_prob'][bb].squeeze().nonzero()[0]
            offset = pred_dict[bb]['offset'][gt_idx].squeeze()

            offset_loss = F.smooth_l1_loss(offset, gt_dict['offset'][bb], reduction='sum')
            offset_loss_total += offset_loss

            reg_loss = F.smooth_l1_loss(pred_dict[bb]['traj_with_gt'], gt_dict['y'][bb], reduction='sum')
            reg_loss_total += reg_loss

            score_gt = F.softmax(-distance_metric(pred_dict[bb]['traj'], gt_dict['y'][bb])/self.temper, dim=0).detach()
            score_loss = F.binary_cross_entropy(pred_dict[bb]['score'], score_gt.squeeze(), reduction='sum')
            score_loss_total += score_loss

            if self.aux_loss and aux_pred and aux_gt:
                aux_loss = F.smooth_l1_loss(aux_pred[bb], aux_gt[bb], reduction="sum")
                aux_loss_total += aux_loss

        loss = self.lambda1 * (cls_loss_total + offset_loss_total) + \
            self.lambda2 * reg_loss_total + \
            self.lambda3 * score_loss_total + \
            aux_loss_total

        loss_dict = {
            "tar_cls_loss": cls_loss_total, 
            "tar_offset_loss": offset_loss_total, 
            "traj_loss": reg_loss_total, 
            "score_loss": score_loss_total
        }

        return loss, loss_dict