'''
Author: zhanghao
LastEditTime: 2023-06-29 09:59:44
FilePath: /my_vectornet_github/dataset/data_augment.py
LastEditors: zhanghao
Description: 
'''
import torch
import numpy as np
import random


class TrainAugmentation:
    def __init__(self):
        self.augment = Compose([
            RandomMaskTargetTopx(prob=0.1, max_topx=15),
            RandomMaskTargetMiddle(prob=0.1, max_num=3, horizon=18),
            RandomVectorNoise(prob=0.1),
        ])

    def __call__(self, data):
        return self.augment(data)



class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        for aug in self.augmentations:
            data = aug(data)
        return data


class RandomMaskTargetTopx(object):
    """
    Random mask 0-x frame constantly
    """
    def __init__(self, prob=0.2, max_topx=15):
        self.prob = prob
        self.max_topx = max_topx

    def __call__(self, data):
        if np.random.random() < self.prob:
            mask_num = np.random.randint(self.max_topx)
            data['x'][:mask_num, :-2] = 0
            data['aug_mask_topx'] = mask_num
        else:
            data['aug_mask_topx'] = 0
        
        # print('aug_mask_topx:', data['aug_mask_topx'])
        return data 


class RandomMaskTargetMiddle(object):
    """
    Random mask x frames in 0-18 frames
    """
    def __init__(self, prob=0.2, max_num=4, horizon=18):
        self.prob = prob
        self.max_num = max_num  # max_num=4 : 1,2,3
        self.select_pool = [x for x in range(horizon)]

    def __call__(self, data):
        if np.random.random() < self.prob:
            mask_num = np.random.randint(1, self.max_num)
            selected_idx = random.sample(self.select_pool, k=mask_num)
            data['x'][selected_idx, :-2] = 0
            data['aug_mask_middle'] = torch.tensor(selected_idx)
        else:
            data['aug_mask_middle'] = torch.tensor([])
        
        # print('mask_middle:', data['mask_middle'])
        return data 


# class RandomMaskAgent(object):
#     """
#     Random mask 1 agent, to debate whether to use this.
#     """
#     def __init__(self, prob=0.2):
#         self.prob = prob

#     def __call__(self, data):
#         if np.random.random() < self.prob:
#             obj_num = data["traj_num"]
#             mask_idx = np.random.randint(1, obj_num)
#             data['x'][data['x'][:,-2] == mask_idx, :-2] = 0
#             data['identifier'][mask_idx] = 0
#         return data 


class RandomVectorNoise(object):
    """
    Random add noise to vector: xs/ys/dx/dy.
    """
    def __init__(self, prob=0.2, scale=0.1):
        self.prob = prob
        self.scale = scale  # scale random 0.0-1.0 -> 1.0/scale

    def __call__(self, data):
        if np.random.random() < self.prob:
            noises = (torch.rand((data['x'].shape[0], 4)) - 0.5) * self.scale
            data['x'][:, :4] = data['x'][:, :4] + noises.to(data['x'].device)
            data['aug_vector_noise'] = True
        else:
            data['aug_vector_noise'] = False
        return data 

