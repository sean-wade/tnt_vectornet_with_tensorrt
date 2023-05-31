/*
 * @Author: zhanghao
 * @LastEditTime: 2023-04-23 15:52:55
 * @FilePath: /cpp/utils.cpp
 * @LastEditors: zhanghao
 * @Description:
 */
#include "utils.h"
#include "layer.h"

using namespace nvinfer1;

float calculateDistance(float* traj_data, int idx1, int idx2, int horizon)
{
    float max_diff = -999.0;
    for (int i = 0; i < horizon; i++)
    {
        float x1  = traj_data[idx1 * horizon * 2 + 2 * i];
        float y1  = traj_data[idx1 * horizon * 2 + 2 * i + 1];
        float x2  = traj_data[idx2 * horizon * 2 + 2 * i];
        float y2  = traj_data[idx2 * horizon * 2 + 2 * i + 1];
        float dis = std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2);
        if (dis > max_diff)
        {
            max_diff = dis;
        }
    }
    return max_diff;
    // return std::sqrt(max_diff);
}

std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "[INFO]: Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open Weight file
    std::ifstream input(file);
    assert(input.is_open() && "[ERROR]: Unable to load weight file...");

    // Read number of weights
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    // Loop through number of line, actually the number of weights & biases
    while (count--)
    {
        // TensorRT weights
        Weights  wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        // Read name and type of weights
        std::string w_name;
        input >> w_name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            // Change hex values to uint32 (for higher values)
            input >> std::hex >> val[x];
        }
        // std::cout << w_name << "\n";
        wt.values = val;
        wt.count  = size;

        // Add weight values against its name (key)
        weightMap[w_name] = wt;
    }
    std::cout << "[INFO]: Loading weights done." << std::endl;
    return weightMap;
}

void postProcessCPUV2(
    float* traj_score, float* traj_pred, std::vector<int>& select_index, std::vector<int>& score_index)
{
    int index[TNT_TARGET_SELECT_NUM];
    thrust::sequence(thrust::host, index, index + TNT_TARGET_SELECT_NUM);
    thrust::sort_by_key(thrust::host, traj_score, traj_score + TNT_TARGET_SELECT_NUM, index, thrust::greater<float>());

    select_index.clear();
    score_index.clear();
    select_index.emplace_back(index[0]);
    score_index.emplace_back(0);
    float threshold = TNT_PSEUDO_NMS_THRESH;
    while (select_index.size() < TNT_FINAL_OUT_TRAJ_NUM && threshold >= 0.5)
    {
        for (int i = 1; i < TNT_TARGET_SELECT_NUM; i++)
        {
            bool near_all_select = true;
            for (auto j : select_index)
            {
                if (i != j)
                {
                    float cur_dis = calculateDistance(traj_pred, index[i], index[j], TNT_HORIZON);
                    if (cur_dis < threshold)
                        near_all_select = false;
                }
                else
                {
                    near_all_select = false;
                }
            }
            if (near_all_select)
            {
                select_index.push_back(index[i]);
                score_index.emplace_back(i);
            }
            if (select_index.size() >= TNT_FINAL_OUT_TRAJ_NUM)
                break;
        }
        if (select_index.size() >= TNT_FINAL_OUT_TRAJ_NUM)
            break;
        else
            threshold /= 2.0;
    }
    for (int j = select_index.size(); j < TNT_FINAL_OUT_TRAJ_NUM; j++)
    {
        select_index.emplace_back(index[j]);
        score_index.emplace_back(j);
    }

    // for (auto k : select_index)
    // {
    //     printf("%d, ", k);
    // }
    // printf("\n");
}