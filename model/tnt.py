'''
Author: zhanghao
LastEditTime: 2023-04-27 14:37:06
FilePath: /my_vectornet_github/model/tnt.py
LastEditors: zhanghao
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.vectornet_backbone import VectorNetBackbone
from model.layers.target_prediction import TargetPred
from model.layers.motion_etimation import MotionEstimation
from model.layers.scoring_and_selection import TrajScoreSelection
from model.loss import distance_metric


class TNT(nn.Module):
    def __init__(self,
                 in_channels=8,
                 horizon=30,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 with_aux=False,
                 aux_width=64,
                 target_pred_hid=64,
                 m=50,
                 motion_esti_hid=64,
                 score_sel_hid=64,
                 temperature=0.01,
                 k=6,
                 device=torch.device("cpu")
                 ):
        super(TNT, self).__init__()
        self.horizon = horizon
        self.m = m
        self.k = k

        self.with_aux = with_aux

        self.device = device

        # feature extraction backbone
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layers=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            aux_mlp_width=aux_width,
            device=device
        )

        self.target_pred_layer = TargetPred(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=m,
            device=device
        )
        self.motion_estimator = MotionEstimation(
            in_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=motion_esti_hid
        )
        self.traj_score_layer = TrajScoreSelection(
            feat_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=score_sel_hid,
            temper=temperature,
            device=self.device
        )
        self._init_weight()

    def forward(self, data):
        batch_candidates = [d["candidate"] for d in data]
        batch_gts = [d["target_gt"].view(1, 2) for d in data]

        batch_global_feat, batch_aux_out, batch_aux_gt = self.backbone(data)

        batch_preds = []
        for global_feat, target_candidate, target_gt in zip(batch_global_feat, batch_candidates, batch_gts):
            target_feat = global_feat[:, 0]
            
            # 1. gt-traj generate.
            traj_with_gt = self.motion_estimator(target_feat, target_gt)

            # 2. target prob offset generate.
            target_prob, offset = self.target_pred_layer(target_feat, target_candidate)

            # 3. select top 50 target to generate trajs.
            _, indices = target_prob.topk(self.m, dim=0)
            target_pred_se, offset_pred_se = target_candidate[indices], offset[indices]
            target_loc_se = (target_pred_se + offset_pred_se).view(self.m, 2)

            # 4. top 50 target generate trajs.
            trajs = self.motion_estimator(target_feat, target_loc_se)

            # 5. caculate traj scores.
            score = self.traj_score_layer(target_feat, trajs)

            pred_result = {
                "target_prob": target_prob,
                "offset": offset,
                "traj_with_gt": traj_with_gt,
                "traj": trajs,
                "score": score
            }

            batch_preds.append(pred_result)

        return {"pred": batch_preds, "aux_out" : batch_aux_out, "aux_gt" : batch_aux_gt}


    def inference(self, data):
        batch_candidates = [d["candidate"] for d in data]

        batch_global_feat, _, _ = self.backbone(data)

        batch_trajs, batch_traj_probs = [], []
        for global_feat, target_candidate in zip(batch_global_feat, batch_candidates):
            target_feat = global_feat[:, 0]

            # print("target_feat: \n", target_feat)

            # 2. target prob offset generate.
            target_prob, offset = self.target_pred_layer(target_feat, target_candidate)

            # 3. select top 50 target to generate trajs.
            _, indices = target_prob.topk(self.m, dim=0)
            target_pred_se, offset_pred_se = target_candidate[indices], offset[indices]
            target_loc_se = (target_pred_se + offset_pred_se).view(self.m, 2)

            # 4. top 50 target generate trajs.
            trajs = self.motion_estimator(target_feat, target_loc_se)

            # 5. caculate traj scores.
            score = self.traj_score_layer(target_feat, trajs)

            # print("trajs: \n", trajs)
            # print("score: \n", score)

            traj_final_k, traj_final_k_prob = self.traj_selection(trajs, score)
            traj_final_k_prob = traj_final_k_prob.view(self.k)
            # traj_final_k_prob = traj_final_k_prob / traj_final_k_prob.sum()

            batch_trajs.append(traj_final_k.view(self.k, self.horizon, 2))
            batch_traj_probs.append(traj_final_k_prob)

        return batch_trajs, batch_traj_probs

    def traj_selection(self, traj_in, score, threshold=4):
        score_descend, order = score.sort(descending=True)
        traj_pred = traj_in[order]
        traj_selected = traj_pred[:self.k].clone()
        traj_prob = score_descend[:6].clone()

        debug_index_selected = [0]
        traj_cnt = 1
        thres = threshold
        while traj_cnt < self.k:
            for j in range(1, self.m):
                dis = distance_metric(traj_selected[:traj_cnt], traj_pred[j].unsqueeze(0))
                if not torch.any(dis < thres):
                    traj_selected[traj_cnt] = traj_pred[j].clone()
                    traj_prob[traj_cnt] = score_descend[j]
                    traj_cnt += 1
                    debug_index_selected.append(j)
                if traj_cnt >= self.k:
                    break
            else:
                thres /= 2.0

        # print("\ndebug_index_selected: ", debug_index_selected)

        return traj_selected, traj_prob

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


if __name__ == "__main__":
    from tqdm import tqdm
    from dataset.sg_dataloader import SGTrajDataset, collate_list, collate_list_cuda
    from torch.utils.data import Dataset, DataLoader

    dataset = SGTrajDataset(data_root='mini_data/train/', in_mem=True)
    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, collate_fn=collate_list_cuda)

    device = torch.device('cuda:0')
    model = TNT(in_channels=6, device=device, with_aux=True)
    model.to(device)
    # model.eval()

    import time
    s = time.time()
    for i in range(10):
        for data in tqdm(loader):
            # data = data_to_device(data, device)
            output = model(data)
            # output = model.inference(data)

            # training
            # for pred in output['pred']:
            #     print(pred.shape)
            # for aux_out in output['aux_out']:
            #     print(aux_out.shape)

            ## val
            # for out in output:
            #     print(out[0].shape)
            # print("--------")
        print("epoch %d done..."%(i+1))
    e = time.time()
    print("using %.2f s"%(e - s))
    