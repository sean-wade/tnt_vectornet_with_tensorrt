/*
 * @Author: zhanghao
 * @LastEditTime: 2023-04-18 15:49:44
 * @FilePath: /cpp/tnt.h
 * @LastEditors: zhanghao
 * @Description:
 */
#include <map>
#include <vector>
#include <fstream>
#include <chrono>
#include <iostream>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "layer.h"

using namespace nvinfer1;
static Logger gLogger;

struct TNTPredictTraj
{
    std::vector<std::vector<float>> pred_trajs;
    std::vector<float>              scores;
};

struct TrajfeatureInputData
{
    int feats_num     = 0;
    int cluster_num   = 0;
    int candidate_num = 0;

    float* feature;
    float* cluster;
    float* cluster_count;
    float* id_embedding;
    float* candidate_points;

    void load_from_file(std::string file_path){};
};

struct TNTOptions
{
    std::string weights_path;
    std::string engine_path;

    // std::vector<std::string> engineInputTensorNames  = {"feats", "cluster", "cluster_count", "id_embedding"};
    // std::vector<std::string> engineOutputTensorNames = {"pred"};

    int  gpu_id   = 0;
    bool ues_fp16 = false;
};

class TNT
{
public:
    TNT()  = default;
    ~TNT() = default;

    /*
     * @description:
     * @param {TNTOptions&} options
     * @return {*}
     */
    bool Init(const TNTOptions& options);

    /*
     * @description:
     * @param {TrajfeatureInputData&} input_data
     * @return {*}
     */
    bool Process(TrajfeatureInputData& input_data, TNTPredictTraj& pred_data);

private:
    bool createTNTEngine();
    bool serializeEngineToFile();
    bool deserializeEngineFromFile();

private:
    TNTOptions         m_options_;
    ICudaEngine*       m_engine_;
    IExecutionContext* m_context_;
};