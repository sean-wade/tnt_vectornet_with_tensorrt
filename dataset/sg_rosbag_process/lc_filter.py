import os
import sys
import numpy as np
import math
import time

class LaneChangeFilter:
    def __init__(self,data) -> None:
        self.data = data
        self.obs_horizon = 20

    def process(self):
        #print("lane change filter process")
        # get the origin and compute the oritentation of the target agent
        orig = self.data['trajs'][0][self.obs_horizon -
                                1][3:5].copy().astype(np.float32)
        pre = (orig - self.data['trajs'][0][self.obs_horizon-4][3:5]) / 2.0
        theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2

        if self.data["lane_raw"]:
            lane_theta = self.cal_near_lane(orig)
        if(abs(lane_theta-theta)< np.pi/6):
            theta = lane_theta

        rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)

        if not self.has_front_obj:
            print("no front obj")
            return False
        dest = self.data['trajs'][0][-1][3:5].copy().astype(np.float32)
        point_rot = np.matmul(
                rot, (dest - orig).T).T
        
        if abs(point_rot[0]) > 0.5:
            return True
        
        return False

    def cal_near_lane(self,orig):
        min_dis = 10000
        min_id = None
        for i,l in enumerate(self.data["lane_raw"]):
            point = l["points"][0]
            dis= math.sqrt(pow((point[0]-orig[0]),2)+pow((point[1]-orig[1]),2))
            if dis < min_dis:
                min_id = i
        index = min(4,len(l["points"]))
        pre = (self.data["lane_raw"][min_id]["points"][index] - point)/2.0
        theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
        return theta

    def has_front_obj(self,orig,rot):
        data = self.data
        obj_front_list = []
        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_horizon-1 not in step:
                continue
            index = np.where(step == self.obs_horizon-1)
            obj_pos = traj[index[0][0]][3:5].astype(np.float32)
            obj2ego = np.matmul(rot, (obj_pos - orig).T).T
            if abs(obj2ego[0]) < 2 and obj2ego[1] > 0:
                obj_front_list.append(obj2ego)
        return obj_front_list
