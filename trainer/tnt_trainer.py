'''
Author: zhanghao
LastEditTime: 2023-04-23 16:03:32
FilePath: /my_vectornet_github/trainer/tnt_trainer.py
LastEditors: zhanghao
Description: 
'''
import os
import numpy as np
from tqdm import tqdm
from loguru import logger
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel
except:
    pass

from trainer.basic_trainer import Trainer
from trainer.optim_schedule import ScheduledOptim
from dataset.util.vis_utils_v2 import Visualizer
from model.tnt import TNT
from model.loss import TNTLoss


class TNTTrainer(Trainer):
    """
    TNT Trainer, train the TNT with specified hyperparameters and configurations
    """
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 collate_fn,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 num_global_graph_layer=1,
                 horizon: int = 30,
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=15,
                 lr_update_freq=5,
                 lr_decay_rate=0.3,
                 aux_loss: bool = False,
                 with_cuda: bool = False,
                 cuda_device=None,
                 multi_gpu=False,
                 log_freq: int = 2,
                 save_folder: str = "",
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True
                 ):
        super(TNTTrainer, self).__init__(
            trainset=trainset,
            evalset=evalset,
            testset=testset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            warmup_epoch=warmup_epoch,
            with_cuda=with_cuda,
            cuda_device=cuda_device,
            multi_gpu=multi_gpu,
            log_freq=log_freq,
            save_folder=save_folder,
            verbose=verbose,
            enable_log=True
        )

        # init or load model
        self.horizon = horizon
        self.aux_loss = aux_loss

        self.lambda1 = 0.1
        self.lambda2 = 1.0
        self.lambda3 = 0.1

        model_name = TNT
        self.model = model_name(
            self.trainset.num_features if hasattr(self.trainset, 'num_features') else self.testset.num_features,
            self.horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device
        )
        self.criterion = TNTLoss(
            self.lambda1, 
            self.lambda2, 
            self.lambda3,
            self.model.m, 
            self.model.k, 
            temper=0.01,
            aux_loss=self.aux_loss,
            reduction='sum',
            device=self.device
        )

        # init optimizer
        self.optim = AdamW(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optm_schedule = ScheduledOptim(
            self.optim,
            self.lr,
            n_warmup_epoch=self.warmup_epoch,
            update_rate=lr_update_freq,
            decay_rate=lr_decay_rate
        )

        # resume from model file or maintain the original
        if model_path:
            self.load(model_path, 'm')
        # load ckpt
        elif ckpt_path:
            self.load(ckpt_path, 'c')

        self.model = self.model.to(self.device)
        if self.multi_gpu:
            self.model = DistributedDataParallel(self.model)
            self.model, self.optimizer = amp.initialize(self.model, self.optim, opt_level="O0")
            if self.verbose:
                logger.info("[TNTTrainer]: Train the mode with multiple GPUs: {}.".format(self.cuda_id))
        else:
            if self.verbose:
                logger.info("[TNTTrainer]: Train the mode with single device on {}.".format(self.device))

        # record the init learning rate
        if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
            self.write_log("LR", self.lr, 0)


    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        # data_iter = tqdm(
        #     enumerate(dataloader),
        #     desc="{}_Ep_{}: loss: {:.5f}; avg_loss: {:.5f}".format("train" if training else "eval",
        #                                                             epoch,
        #                                                             0.0,
        #                                                             avg_loss),
        #     total=len(dataloader),
        #     bar_format="{l_bar}{r_bar}"
        # )
        learning_rate = self.lr
        # for i, data in data_iter:
        for i, data in enumerate(dataloader):
            n_graph = len(data)
            data = self.data_to_device(data)

            if training:
                self.optm_schedule.zero_grad()
                loss, loss_dict = self.compute_loss(data)

                if self.multi_gpu:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optim.step()

                if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                    self.write_log("Loss/Train_Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Loss/Target_Cls_Loss",
                                loss_dict["tar_cls_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Loss/Target_Offset_Loss",
                                loss_dict["tar_offset_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Loss/Traj_Loss",
                                loss_dict["traj_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Loss/Score_Loss",
                                loss_dict["score_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Loss/Aux_Loss",
                                loss_dict["aux_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
            else:
                with torch.no_grad():
                    loss, loss_dict = self.compute_loss(data)

                    # writing loss
                    if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                        self.write_log("Eval/Eval_Loss", loss.item() / n_graph, i + epoch * len(dataloader))

            num_sample += n_graph
            avg_loss += loss.detach().item()

            if training and i % self.log_freq == 0:
                # print log info
                desc_str = "Epoch-[{}], iter-[{}/{}]: loss: {:.2f}, avg_loss: {:.2f}, target_cls_loss: {:.2f}, target_offset_loss: {:.2f}, traj_loss: {:.2f}, score_loss: {:.2f}, aux_loss: {:.2f}, lr: {}".format(
                    epoch, i, len(dataloader),
                    loss.detach().item() / n_graph,
                    avg_loss / num_sample,
                    loss_dict["tar_cls_loss"].detach().item() / n_graph,
                    loss_dict["tar_offset_loss"].detach().item() / n_graph,
                    loss_dict["traj_loss"].detach().item() / n_graph,
                    loss_dict["score_loss"].detach().item() / n_graph,
                    loss_dict["aux_loss"].detach().item() / n_graph,
                    learning_rate
                )
                # data_iter.set_description(desc=desc_str, refresh=True)
                logger.info(desc_str)

        if training:
            if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                learning_rate = self.optm_schedule.step_and_update_lr()
                self.write_log("LR", learning_rate, epoch)
        else:    # add by zhanghao.
            cur_loss = avg_loss / num_sample
            if not self.min_eval_loss:
                self.min_eval_loss = cur_loss
            elif cur_loss < self.min_eval_loss:
                self.save(epoch, cur_loss)

            metric = self.eval_save_model("best")
            self.write_log("Eval/minADE", metric["minADE"], epoch)
            self.write_log("Eval/minFDE", metric["minFDE"], epoch)
            self.write_log("Eval/MR", metric["MR"], epoch)

            desc_str = "Validation-[{}]: loss: {:.2f}, min_eval_loss: {:.2f}, minADE: {:.2f}, minFDE: {:.2f}, MR: {:.2f}\n".format(
                        epoch,
                        cur_loss,
                        self.min_eval_loss,
                        metric["minADE"],
                        metric["minFDE"],
                        metric["MR"],
                    )
            # data_iter.set_description(desc=desc_str, refresh=True)
            logger.info(desc_str)

        return avg_loss / num_sample

    def compute_loss(self, data):
        out = self.model(data)
        gt = {
                "target_prob": [dd["candidate_gt"] for dd in data],
                "offset": [dd["offset_gt"] for dd in data],
                "y": [dd["y"].view(self.horizon, 2).cumsum(axis=0).view(1,-1) for dd in data]
        }
        return self.criterion(out["pred"], gt, out["aux_out"], out["aux_gt"])

    def test(self,
             miss_threshold=2.0,
             compute_metric=False,
             convert_coordinate=True,
             plot=False,
             save_pred=False):
        """
        test the testset,
            :param miss_threshold: float, the threshold for the miss rate, default 2.0m
            :param compute_metric: bool, whether compute the metric
            :param convert_coordinate: bool, True: under original coordinate, False: under the relative coordinate
            :param save_pred: store the prediction or not, store in the Argoverse benchmark format
        """
        self.model.eval()
        forecasted_trajectories, gt_trajectories, forecasted_probabilities = {}, {}, {}
        k = self.model.k
        horizon = self.model.horizon

        with torch.no_grad():
            for data in tqdm(self.test_loader, "Test inference..."):
                batch_size = len(data)
                data = self.data_to_device(data)

                # inference and transform dimension
                if self.multi_gpu:
                    out = self.model.module.inference(data)
                else:
                    out = self.model.inference(data)

                if len(out) == 2:
                    out, traj_prob = out
                    traj_prob = [pp.cpu().numpy().tolist() for pp in traj_prob]
                else:
                    traj_prob = [1.0] * 6

                # record the prediction and ground truth
                for batch_id in range(batch_size):
                    seq_id = data[batch_id]["seq_id"]
                    if convert_coordinate:
                        rot = data[batch_id]["rot"]
                        orig = data[batch_id]["orig"]

                        # pred = self.convert_coord(out[batch_id].squeeze().cpu().numpy(), orig, rot)
                        # forecasted_trajectories[seq_id] = pred[np.newaxis, :]

                        forecasted_trajectories[seq_id] = [self.convert_coord(pred_y_k, orig, rot)
                                                        if convert_coordinate else pred_y_k
                                                        for pred_y_k in out[batch_id].squeeze().cpu().numpy()]

                        gt = data[batch_id]["y"].view(-1, 2).cumsum(axis=0).cpu().numpy()
                        gt_trajectories[seq_id] = self.convert_coord(gt, orig, rot)
                        forecasted_probabilities[seq_id] = traj_prob[batch_id]
                    else:
                        forecasted_trajectories[seq_id] = out[batch_id].cpu().numpy()
                        gt_trajectories[seq_id] = data[batch_id]["y"].view(-1, 2).cumsum(axis=0).cpu().numpy()
                        forecasted_probabilities[seq_id] = traj_prob[batch_id]

        print(forecasted_trajectories)
        print(forecasted_probabilities)

        # compute the metric
        if compute_metric:
            metric_results = get_displacement_errors_and_miss_rate(
                forecasted_trajectories,
                gt_trajectories,
                k,
                horizon,
                miss_threshold,
                forecasted_probabilities
            )
            logger.info("[TNTTrainer]: The test result: {}".format(metric_results))
            if save_pred:
                with open(self.save_folder + "/result.txt", "a") as fff:
                    fff.write(str(metric_results))

        if save_pred:
            os.makedirs(self.save_folder + "/pd/", exist_ok=True)
            os.makedirs(self.save_folder + "/gt/", exist_ok=True)
            for k, v in forecasted_trajectories.items():
                with open(self.save_folder + "/pd/%s.txt"%k, "a") as fff:
                    for vv in v:
                        fff.write(str(vv))
                        fff.write("\n\n")
            for k, v in gt_trajectories.items():
                with open(self.save_folder + "/gt/%s.txt"%k, "w") as fff:
                    fff.write(str(v))

        if plot:
            from dataset.sg_dataloader import collate_list
            os.makedirs(self.save_folder + "/fig", exist_ok=True)
            vis = Visualizer(convert_coordinate=convert_coordinate)
            self.plot_loader = self.loader(self.testset, batch_size=1, num_workers=2, collate_fn=collate_list)
            for data in tqdm(self.plot_loader, desc="Ploting and saving..."):
                seq_id = data[0]["seq_id"]
                vis.draw_once(data[0], forecasted_trajectories[seq_id], gt_trajectories[seq_id], forecasted_probabilities[seq_id])
                if save_pred:
                    png_path = self.save_folder + "/fig/tnt_" + str(seq_id) + ".png"
                    plt.savefig(png_path)
                    plt.close()
                else:
                    plt.show()


    def convert_coord(self, traj, orig, rot):
        traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
        return traj_converted


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    from dataset.sg_dataloader import SGTrajDataset, collate_list

    device = torch.device('cuda:0')

    train_dataset = SGTrajDataset(data_root='/mnt/data/SGTrain/rosbag/medium/train', in_mem=False)
    val_dataset = SGTrajDataset(data_root='/mnt/data/SGTrain/rosbag/medium/val', in_mem=False)
    trainer = TNTTrainer(trainset=train_dataset, 
                        evalset=val_dataset, 
                        testset=val_dataset, 
                        num_workers=4,
                        collate_fn=collate_list, 
                        with_cuda=True, 
                        cuda_device=0, 
                        batch_size=128,
                        save_folder="./work_dir/tnt2/",
                        lr=0.01,
                        weight_decay=0.02,
                        warmup_epoch=20,
                        lr_update_freq=20,
                        lr_decay_rate=0.9,
                        aux_loss=False,
                        model_path="work_dir/tnt/best_TNT.pth"
                        )

    for iter_epoch in range(200):
        trainer.train(iter_epoch)
        trainer.eval(iter_epoch)

    trainer.test(miss_threshold=2.0,
            compute_metric=True,
            convert_coordinate=False,
            plot=True,
            save_pred=True)
