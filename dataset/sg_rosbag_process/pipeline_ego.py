'''
Author: zhanghao
LastEditTime: 2023-03-02 19:34:31
FilePath: /sg_generate_vectorize_data_rosbag/pipeline_ego.py
LastEditors: zhanghao
Description: 
    作用: 
        给定一个或多个 rosbag 包(rosbag中必须包含如下6个 topic)，自动解析数据，并生成 ego 轨迹 feature 数据保存
            /perception/land_semantics/detected
            /perception/lidar/tracked
            /perception/traffic_light/detected
            /sensor/ins/fusion
            /sensor/ins/rawimu
    使用: 
        python pipeline_ego.py --root_path /mnt/data/SGTrain/rosbag/bag2 --save_path /mnt/data/SGTrain/rosbag/bag2
        参数含义：
            --root_path    存放 rosbag 的路径(rosbag中必须包含如下6个 topic)
                                                                
            --save_path    存放提取的 sensor_data 和生成的 traj_data 路径
            --start_seq    每个 bag 包提取出的 traj_data feature 文件名前加的 prefix number, 用于合并时防止重复
            --save_viz     是否绘制图像并保存
            --filter_scene  加入场景筛选  使用 --filter_scene lc lk ...
'''
import os
import math
import glob
import json
import shutil
import rosbag
import argparse
import numpy as np
import pymap3d as pm
from tqdm import tqdm
from time import time
from numba import jit
from multiprocessing import Process

from get_sensor_data_from_bag import create_dir, process_one_bag
from generate_ego_traj_feature import TrajDataGenerater
from visualize_traj import plot_save_tjs


def single_pipeline(bag_path, data_path, seq_add=0, save_viz=False, dir_name='',filter_scene = []):
    print("Start processing : ", bag_path)
    print("Saving to : ", data_path)
    process_one_bag(bag_path, data_path)
    tdg = TrajDataGenerater(data_path, 2,filter_scene=filter_scene)
    tdg.process(seq_add, dir_name)
    # print(save_viz)
    if save_viz:
        plot_save_tjs(data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str,
                        help="path to bag dir or bag path")
    parser.add_argument("--save_path", type=str,
                        help="dir to save the processed results")
    parser.add_argument("--start_seq", type=int, default=1,
                        help="save feature name start seq")
    parser.add_argument("--save_viz", action='store_true',
                        default=False, help="save plot image")
    parser.add_argument("--filter_scene",nargs='+',default=[],help="filter scene:lane keep,lane change")

    args = parser.parse_args()

    if args.root_path.endswith(".bag"):
        dir_name = os.path.basename(args.root_path)[:-4]
        save_path = os.path.join(args.save_path, dir_name)
        create_dir(save_path)
        print("PROCESS ONLY ONE BAG %s without calib" % (dir_name))
        single_pipeline(args.root_path, save_path,
                        save_viz=args.save_viz, dir_name=dir_name,filter_scene=args.filter_scene)
    else:
        ps = glob.glob(os.path.join(args.root_path, "*.bag"))
        assert len(ps), "no bag was found !"
        print("Found %d bags!" % (len(ps)))
        for idx, rp in enumerate(ps):
            dir_name = os.path.basename(rp)[:-4]
            save_path = os.path.join(args.save_path, dir_name)
            create_dir(save_path)

            print("=====PROCESSING %d/%d BAG NAME : %s=====" %
                  (idx, len(ps), dir_name))
            p = Process(target=single_pipeline, args=(
                rp, save_path, 10000*(idx + args.start_seq), args.save_viz))
            p.start()
