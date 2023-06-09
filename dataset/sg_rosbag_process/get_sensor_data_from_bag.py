'''
Author: zhanghao
LastEditTime: 2023-03-02 17:16:53
FilePath: /sg_generate_vectorize_data_rosbag/get_sensor_data_from_bag.py
LastEditors: zhanghao
Description: 
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


def create_dir(d):
    if os.path.exists(d):
        print("Dir %s exists, will delete and creat a new one !" % (d))
        shutil.rmtree(d)
    os.makedirs(d)


def get_timestr(sec, nsec):
    nsec_str = str(nsec)
    nsec_str = '0' * (9 - len(nsec_str)) + nsec_str
    return str(sec) + nsec_str


def encode_msg(msg):
    bboxes = []
    classes = []
    trackids = []
    for idx,d in enumerate(msg.vehicles + msg.vrus + msg.traffic_barriers):
        velo_x = d.movement_prop.linear_velocity_mps.x if hasattr(d,'movement_prop') else 0
        velo_y = d.movement_prop.linear_velocity_mps.y if hasattr(d,'movement_prop') else 0
        box = [
            d.static_prop.center_point_m.x, 
            d.static_prop.center_point_m.y, 
            d.static_prop.center_point_m.z,
            d.static_prop.dimension_m.x, 
            d.static_prop.dimension_m.y, 
            d.static_prop.dimension_m.z, 
            velo_x, 
            velo_y,
            d.static_prop.rotation_deg.z / 180.0 * np.pi, 
            d.confidence
        ]
        classes.append(d.type)
        trackids.append(d.tracker.uuid)
        bboxes.append(box)
    classes = np.array(classes,dtype=np.str)
    trackids = np.array(trackids,dtype=np.str)
    bboxes = np.array(bboxes).reshape(len(classes), 10)
    return bboxes, classes, trackids


def save_track(msg, save_path):
    bboxes, classes, trackids = encode_msg(msg)
    with open(save_path,"w") as f:
        for i in range(len(classes)):
            box = bboxes[i]
            cls = classes[i]
            tid = trackids[i]
            string = "%f %f %f %f %f %f %f %f %f %f %s %s\n" % \
            (box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7],box[8],box[9],cls,tid)
            f.writelines(string)
        f.close()


def save_lane(msg, save_path):
    lanes = {"lanes" : []}
    for lane in msg.lanes:
        # print("msg.lanes = ", msg.lanes)
        points_color  = lane.points[0].color
        points_type   = lane.points[0].type
        xyzs    = [[p.position.x, p.position.y, p.position.z] for p in lane.points]
        cur_lane = {
            "lane_id" : lane.lane_id,
            "lane_type" : lane.type,
            "confidence" : lane.confidence,
            "points_color" : points_color,
            "points_type" : points_type,
            "points" : xyzs,
        }
        lanes["lanes"].append(cur_lane)
    with open(save_path, 'w') as ff:
        json.dump(lanes, ff, indent=4, separators=(',', ':'))


def save_tl(msg, save_path):
    """
    traffic light encode:
        [
            [color1, distance1],
            [color2, distance2],
            [color3, distance3],
            [color4, distance4],
            [color4, number],
        ]
    """
    digit_num = -1 if msg.results[0].digit.number=="NN" else msg.results[0].digit.number
    tls = [
        [msg.results[0].light_straight.color.color, msg.results[0].light_straight.distance_m],
        [msg.results[0].light_left.color.color,     msg.results[0].light_left.distance_m],
        [msg.results[0].light_right.color.color,    msg.results[0].light_right.distance_m],
        [msg.results[0].light_uturn.color.color,    msg.results[0].light_uturn.distance_m],
        [msg.results[0].digit.color.color,          digit_num]
    ]
    with open(save_path, "w") as f:
        for tl in tls:
            string = "%d %s\n" %  (tl[0], tl[1])
            f.writelines(string)
        f.close()


def eulerAngles2rotationMat(theta, degree=True):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return: Rotate matrix
    RPY角, 是ZYX欧拉角, 依次绕定轴XYZ转动[rx, ry, rz]
    """
    if degree:
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def get_ins2world_pose(rpy_deg, xyz_shift):
    mat = np.identity(4)
    R = eulerAngles2rotationMat(rpy_deg, degree=True)
    mat[:3,:3] = R
    mat[:3,-1] = xyz_shift
    return mat


def process_one_bag(bag_rp, save_path):
    bag_src = rosbag.Bag(bag_rp)
    bag_time_len = bag_src.get_end_time() - bag_src.get_start_time()
    print("\nBag record time : ", bag_time_len)

    os.makedirs(os.path.join(save_path, "ins_to_global"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "lidar_obj_tracked"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "lane_detected"), exist_ok=True)
    # os.makedirs(os.path.join(save_path, "vehicle_status"),exist_ok=True)
    os.makedirs(os.path.join(save_path, "tl_detected"), exist_ok=True)
    tic = time()

    is_first_ins = True
    pose_ins2world = np.identity(4)
    first_longitude, first_latitude, first_altitude = -1, -1, -1
    for idx, topic in tqdm(enumerate(bag_src.read_messages()), total=bag_src.get_message_count()):
        if topic.topic.startswith("/perception/lidar/tracked"):
            msg = topic.message
            timestamp_str = get_timestr(msg.header.stamp.secs, msg.header.stamp.nsecs)
            save_path_trk = os.path.join(save_path, "lidar_obj_tracked", timestamp_str + ".txt")
            save_track(msg, save_path_trk)

            save_path_rt = os.path.join(save_path, "ins_to_global/" + timestamp_str + ".txt") 
            np.savetxt(save_path_rt, pose_ins2world)

        if topic.topic.startswith("/perception/land_semantics/detected"):
            msg = topic.message
            timestamp_str = get_timestr(msg.header.stamp.secs, msg.header.stamp.nsecs)
            save_path_lane = os.path.join(save_path, "lane_detected", timestamp_str + ".json")
            save_lane(msg, save_path_lane)

        if topic.topic.startswith("/perception/traffic_light/detected"):
            msg = topic.message
            timestamp_str = get_timestr(msg.header.stamp.secs, msg.header.stamp.nsecs)
            save_path_tl = os.path.join(save_path, "tl_detected", timestamp_str + ".txt")
            save_tl(msg, save_path_tl)

        if topic.topic.startswith("/sensor/ins/fusion"):
            msg = topic.message
            timestamp_str = get_timestr(msg.header.stamp.secs, msg.header.stamp.nsecs)
            if is_first_ins:
                first_latitude, first_longitude, first_altitude = msg.latitude, msg.longitude, msg.altitude
                is_first_ins = False

            e,n,u = pm.geodetic2enu(msg.latitude, msg.longitude, msg.altitude, 
                                first_latitude, first_longitude, first_altitude)
            positions = [e,n,u]
            rpys = [msg.roll, msg.pitch, msg.yaw]
            pose_ins2world = get_ins2world_pose(rpys, positions)

    toc = time()
    print("Totally %f sec used " % (toc-tic))


parser = argparse.ArgumentParser()
parser.add_argument("--root_path",type=str, help="path to bag dir or bag path")
parser.add_argument("--save_path",type=str, help="dir to save the processed results")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.root_path.endswith(".bag"):
        dir_name = os.path.basename(args.root_path)[:-4]
        save_path = os.path.join(args.save_path, dir_name)
        create_dir(save_path)
        print("PROCESS ONLY ONE BAG %s without calib" % (dir_name))
        process_one_bag(args.root_path, save_path)
    else:
        ps = glob.glob(os.path.join(args.root_path,"*.bag"))
        assert len(ps), "no bag was found !"
        print("Found %d bags!" % (len(ps)))
        for idx, rp in enumerate(ps):
            dir_name = os.path.basename(rp)[:-4]
            save_path = os.path.join(args.save_path, dir_name)
            create_dir(save_path)

            print("=====PROCESSING %d/%d BAG NAME : %s=====" % (idx, len(ps), dir_name))
            p = Process(target=process_one_bag, args=(rp, save_path))
            p.start()
