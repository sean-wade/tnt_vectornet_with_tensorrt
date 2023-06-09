'''
Author: zhanghao
LastEditTime: 2023-06-08 16:41:57
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
    parser.add_argument("-rr", "--regex_root", type=str, default="/home/jovyan/vol-2/wangyisong/tnt_data/*/traj_data/")
    parser.add_argument("-d", "--dest", type=str, default="/home/jovyan/workspace/DATA/TRAJ_DATASET/EXP6_Heading_Diamond_ALL/")
    parser.add_argument("-v", "--viz", action='store_true', default=False)
    parser.add_argument("-norm", "--normalized", action='store_true', default=True)
    args = parser.parse_args()
    
    print(args)

    folders = glob.glob(args.regex_root)
    print(len(folders))
    for fd in folders:
        print(fd)
    
    process_with_folders(folders, args.dest, args)