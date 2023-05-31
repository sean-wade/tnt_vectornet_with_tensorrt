/*
 * @Author: zhanghao
 * @LastEditTime: 2023-04-25 15:22:15
 * @FilePath: /my_vectornet_github/tensorrt_deploy/tnt_trt/cpp/utils.h
 * @LastEditors: zhanghao
 * @Description:
 */
#include <map>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

using namespace nvinfer1;

std::map<std::string, Weights> loadWeights(const std::string file);

float calculateDistance(float* traj_data, int idx1, int idx2, int horizon = 30);

// void postProcessCPU(float* traj_score, float* traj_pred, std::vector<int>& select_index);
void postProcessCPUV2(
    float* traj_score, float* traj_pred, std::vector<int>& select_index, std::vector<int>& score_index);
