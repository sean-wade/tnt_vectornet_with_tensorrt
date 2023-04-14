'''
Author: zhanghao
LastEditTime: 2023-04-14 16:43:54
FilePath: /my_vectornet_github/tools/train_tnt.py
LastEditors: zhanghao
Description: 
'''
import os
import sys
import json
import argparse
from loguru import logger
from datetime import datetime
from trainer.tnt_trainer import TNTTrainer
from dataset.sg_dataloader import SGTrajDataset, collate_list

@logger.catch
def train(n_gpu, args):
    logger.info("Start training tnt...")
    logger.info("Configs: {}".format(args))
    train_set = SGTrajDataset(args.data_root + "/train/", in_mem=args.on_memory)
    val_set = SGTrajDataset(args.data_root + "/val/", in_mem=args.on_memory)

    # init output dir
    time_stamp = datetime.now().strftime("%m_%d_%H_%M")
    output_dir = os.path.join(args.output_dir, time_stamp)
    if not args.multi_gpu or (args.multi_gpu and n_gpu == 0):
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            raise Exception("The output folder does exists and is not empty! Check the folder.")
        else:
            os.makedirs(output_dir)

            # # dump the args
            # with open(os.path.join(output_dir, 'conf.json'), 'w') as fp:
            #     json.dump(vars(args), fp, indent=4, separators=(", ", ": "))

    # init trainer
    trainer = TNTTrainer(
        trainset=train_set,
        evalset=val_set,
        testset=val_set,
        collate_fn=collate_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        warmup_epoch=args.warmup_epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_update_freq=args.lr_update_freq,
        weight_decay=args.adam_weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        num_global_graph_layer=args.num_glayer,
        aux_loss=args.aux_loss,
        with_cuda=args.with_cuda,
        cuda_device=n_gpu,
        multi_gpu=args.multi_gpu,
        save_folder=output_dir,
        log_freq=args.log_freq,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None
    )

    # training
    for iter_epoch in range(args.n_epoch):
        trainer.train(iter_epoch)
        trainer.eval(iter_epoch)

    trainer.eval_save_model("final")
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", required=False, type=str, default="/mnt/data/SGTrain/rosbag/medium/",
                        help="root dir for datasets")
    parser.add_argument("-o", "--output_dir", required=False, type=str, default="work_dir/tnt/",
                        help="dir to save checkpoint and model")

    parser.add_argument("-l", "--num_glayer", type=int, default=1,
                        help="number of global graph layers")
    parser.add_argument("-a", "--aux_loss", action="store_true", default=True,
                        help="Training with the auxiliary recovery loss")

    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="number of batch_size")
    parser.add_argument("-e", "--n_epoch", type=int, default=200,
                        help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0,
                        help="dataloader worker size")

    parser.add_argument("-c", "--with_cuda", action="store_true", default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("-m", "--multi_gpu", action="store_true", default=False,
                        help="training with distributed data parallel: true, or false")
    parser.add_argument("-r", "--local_rank", default=0, type=int,
                        help="the default id of gpu")

    parser.add_argument("--log_freq", type=int, default=2,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate of adam")
    parser.add_argument("-we", "--warmup_epoch", type=int, default=10,
                        help="the number of warmup epoch with initial learning rate, after the learning rate decays")
    parser.add_argument("-luf", "--lr_update_freq", type=int, default=20,
                        help="learning rate decay frequency for lr scheduler")
    parser.add_argument("-ldr", "--lr_decay_rate", type=float, default=0.8, help="lr scheduler decay rate")

    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("-rc", "--resume_checkpoint", type=str, help="resume a checkpoint for fine-tune")
    parser.add_argument("-rm", "--resume_model", type=str, help="resume a model state for fine-tune")

    args = parser.parse_args()
    train(args.local_rank, args)
