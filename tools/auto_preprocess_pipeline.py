'''
Author: zhanghao
LastEditTime: 2023-04-27 15:53:38
FilePath: /my_vectornet_github/tools/auto_preprocess_pipeline.py
LastEditors: zhanghao
Description: 
    自动遍历某目录下的所有文件夹，找到包含
        traj_data
    内容的子文件夹，生成 all-agents 训练数据
'''
import glob
import argparse
from dataset.sg_preprocess_all_agents import process_with_folders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rr", "--regex_root", type=str, default="/mnt/data/SGTrain/rosbag/*/*/traj_data/")
    parser.add_argument("-d", "--dest", type=str, default="/mnt/data/SGTrain/rosbag/sg_all_agent/")
    parser.add_argument("-v", "--viz", action='store_true', default=False)
    args = parser.parse_args()

    folders = glob.glob(args.regex_root)
    for fd in folders:
        print(fd)
    process_with_folders(folders, args.dest, args)