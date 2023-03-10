'''
Author: zhanghao
LastEditTime: 2023-03-10 17:44:51
FilePath: /vectornet/trainer/vectornet_trainer.py
LastEditors: zhanghao
Description: 
'''
import os
import numpy as np
from tqdm import tqdm
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
from model.vectornet import VectorNet
from model.loss import VectorLoss


class VectorNetTrainer(Trainer):
    """
    VectorNetTrainer, train the vectornet with specified hyperparameters and configurations
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
        """
        trainer class for vectornet
            :param train_loader: see parent class
            :param eval_loader: see parent class
            :param test_loader: see parent class
            :param lr: see parent class
            :param betas: see parent class
            :param weight_decay: see parent class
            :param warmup_steps: see parent class
            :param with_cuda: see parent class
            :param multi_gpu: see parent class
            :param log_freq: see parent class
            :param model_path: str, the path to a trained model
            :param ckpt_path: str, the path to a stored checkpoint to be resumed
            :param verbose: see parent class
        """
        super(VectorNetTrainer, self).__init__(
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

        model_name = VectorNet
        # model_name = OriginalVectorNet
        self.model = model_name(
            self.trainset.num_features,
            self.horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device
        )
        self.criterion = VectorLoss(aux_loss=self.aux_loss, reduction="sum")

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
                print("[TNTTrainer]: Train the mode with multiple GPUs: {}.".format(self.cuda_id))
        else:
            if self.verbose:
                print("[TNTTrainer]: Train the mode with single device on {}.".format(self.device))

        # record the init learning rate
        if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
            self.write_log("LR", self.lr, 0)


    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        data_iter = tqdm(
            enumerate(dataloader),
            desc="{}_Ep_{}: loss: {:.5f}; avg_loss: {:.5f}".format("train" if training else "eval",
                                                                    epoch,
                                                                    0.0,
                                                                    avg_loss),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:
            n_graph = len(data)
            data = self.data_to_device(data)

            if training:
                self.optm_schedule.zero_grad()
                loss = self.compute_loss(data)

                if self.multi_gpu:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optim.step()
                if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                    self.write_log("Train Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))

            else:
                with torch.no_grad():
                    loss = self.compute_loss(data)

                    if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                        self.write_log("Eval/Eval Loss", loss.item() / n_graph, i + epoch * len(dataloader))

            num_sample += n_graph
            avg_loss += loss.detach().item()

            # print log info
            desc_str = "[Info: Device_{}: {}_Ep_{}: loss: {:.5f}; avg_loss: {:.5f}]".format(
                self.cuda_id,
                "train" if training else "eval",
                epoch,
                loss.item() / n_graph,
                avg_loss / num_sample)
            data_iter.set_description(desc=desc_str, refresh=True)

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

        return avg_loss / num_sample


    def compute_loss(self, data):
        
        if self.model.training:
            # training
            out = self.model(data)
            pred = torch.cat(out["pred"])
            gt = torch.cat([dd["y"] for dd in data]).reshape(len(data), -1)
            aux_gt = torch.cat(out["aux_gt"]) if len(out["aux_gt"]) > 0 else None
            aux_out = torch.cat(out["aux_out"]) if len(out["aux_out"]) > 0 else None
            return self.criterion(pred, gt, aux_out, aux_gt)
        else:
            out = self.model(data)
            pred = torch.cat(out)
            gt = torch.cat([dd["y"] for dd in data]).reshape(len(data), -1)
            return self.criterion(pred, gt)


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
        forecasted_trajectories, gt_trajectories = {}, {}
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

                # record the prediction and ground truth
                for batch_id in range(batch_size):
                    seq_id = data[batch_id]["seq_id"]
                    if convert_coordinate:
                        rot = data[batch_id]["rot"]
                        orig = data[batch_id]["orig"]
                        pred = self.convert_coord(out[batch_id].squeeze().cpu().numpy(), orig, rot)
                        forecasted_trajectories[seq_id] = pred[np.newaxis, :]
                        gt = data[batch_id]["y"].view(-1, 2).cumsum(axis=0).cpu().numpy()
                        gt_trajectories[seq_id] = self.convert_coord(gt, orig, rot)
                    else:
                        forecasted_trajectories[seq_id] = out[batch_id].cpu().numpy()
                        gt_trajectories[seq_id] = data[batch_id]["y"].view(-1, 2).cumsum(axis=0).cpu().numpy()

        # compute the metric
        if compute_metric:
            metric_results = get_displacement_errors_and_miss_rate(
                forecasted_trajectories,
                gt_trajectories,
                1,
                horizon,
                miss_threshold
            )
            print("[VectornetTrainer]: The test result: {};".format(metric_results))
            if save_pred:
                with open(self.save_folder + "/result.txt", "a") as fff:
                    fff.write(str(metric_results))

        if save_pred:
            os.makedirs(self.save_folder + "/pd/", exist_ok=True)
            os.makedirs(self.save_folder + "/gt/", exist_ok=True)
            for k, v in forecasted_trajectories.items():
                with open(self.save_folder + "/pd/%s.txt"%k, "w") as fff:
                    fff.write(str(v[0]))
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
                vis.draw_once(data[0], forecasted_trajectories[seq_id], gt_trajectories[seq_id])
                if save_pred:
                    png_path = self.save_folder + "/fig/vectornet_" + str(seq_id) + ".png"
                    plt.savefig(png_path)
                    plt.close()
                else:
                    plt.show()


    def convert_coord(self, traj, orig, rot):
        traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
        return traj_converted


if __name__ == "__main__":
    from model.vectornet import VectorNet
    from dataset.sg_dataloader import SGTrajDataset, collate_list

    device = torch.device('cuda:0')

    dataset = SGTrajDataset(data_root='/mnt/data/SGTrain/rosbag/train_feature', in_mem=True)
    trainer = VectorNetTrainer(trainset=dataset, 
                                evalset=dataset, 
                                testset=dataset, 
                                collate_fn=collate_list, 
                                with_cuda=True, 
                                cuda_device=0, 
                                batch_size=64,
                                save_folder="./work_dir/vectornet/",
                                lr=0.005,
                                weight_decay=0.01,
                                warmup_epoch=20,
                                lr_update_freq=20,
                                lr_decay_rate=0.9,
                                aux_loss=True,
                                )

    for iter_epoch in range(100):
        trainer.train(iter_epoch)
        trainer.eval(iter_epoch)

    trainer.test(miss_threshold=2.0,
            compute_metric=True,
            convert_coordinate=False,
            plot=True,
            save_pred=True)
