'''
Author: zhanghao
LastEditTime: 2023-04-28 10:05:02
FilePath: /my_vectornet_github/tools/test_tnt.py
LastEditors: zhanghao
Description: 
'''
import os
import glob
import json
import argparse
from loguru import logger
from datetime import datetime
from trainer.tnt_trainer import TNTTrainer
from trainer.utils.logger import setup_logger
from dataset.sg_dataloader import SGTrajDataset, collate_list


def test(args):
    time_stamp = datetime.now().strftime("%m_%d_%H_%M")
    args.save_dir = os.path.join(args.save_dir, time_stamp) + "_tnt"

    setup_logger(
            args.save_dir,
            distributed_rank=0,
            filename="train_log.txt",
            mode="a",
        )
    logger.info("Start testing tnt...")
    logger.info("Configs: {}".format(args))
    
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_pred:
        os.makedirs(args.save_dir + "/fig", exist_ok=True)

    # print(str(vars(args)))
    # with open(args.save_dir + "/result.txt", "a") as fff:
    #     fff.write(str(vars(args)) + "\n\n")
    
    test_path_list = glob.glob(args.data_root + "/*/" + args.split)
    print(test_path_list)
    
    # data loading
    try:
        test_set = SGTrajDataset(test_path_list, in_mem=args.on_memory)
    except:
        raise Exception("Failed to load the data, please check the dataset!")

    # init trainer
    trainer = TNTTrainer(
        trainset=test_set,
        evalset=test_set,
        testset=test_set,
        collate_fn=collate_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aux_loss=True,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=args.save_dir,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None
    )

    trainer.test(miss_threshold=2.0, 
                save_pred=args.save_pred, 
                convert_coordinate=True,
                compute_metric=True,
                plot=not args.noplot
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", type=str, default="/home/jovyan/workspace/DATA/TRAJ_ALL_AGENTS_0427/", help="root dir for datasets")
    parser.add_argument("-s", "--split", type=str, default="val")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")
    parser.add_argument("-c", "--with_cuda", action="store_true", default=True, help="training with CUDA: true, or false")
    parser.add_argument("-cd", "--cuda_device", type=int, default=0, help="CUDA device ids")

    parser.add_argument("-rc", "--resume_checkpoint", type=str,
                        # default="checkpoint_iter26.ckpt",
                        help="resume a checkpoint for fine-tune")
    parser.add_argument("-rm", "--resume_model", type=str,
                        default="weights/sg_best_TNT.pth",
                        help="resume a model state for fine-tune")

    parser.add_argument("-sd", "--save_dir", type=str, default="work_dir/test/"),
    parser.add_argument("-sv", "--save_pred", action="store_true", default=True)
    parser.add_argument("-npl", "--noplot", action="store_true", default=False)
    parser.add_argument("--on_memory", type=bool, default=False, help="Loading on memory: true or false")
    args = parser.parse_args()
    test(args)
