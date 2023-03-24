/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-24 16:00:03
 * @FilePath: /vectornetx/vectornet.h
 * @LastEditors: zhanghao
 * @Description:
 */
/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-24 16:00:02
 * @FilePath: /vectornetx/vectornet.h
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

struct TrajPredictData
{
    std::vector<float> predict_points;
};

struct TrajfeatureInputData
{
    int feats_num   = 0;
    int cluster_num = 0;

    float* feature;
    float* cluster;
    float* cluster_count;
    float* id_embedding;
    float* candidate_points;

    void load_from_file(std::string file_path){};
};

struct VectorNetOptions
{
    std::string weights_path;
    std::string engine_path;

    // std::vector<std::string> engineInputTensorNames  = {"feats", "cluster", "cluster_count", "id_embedding"};
    // std::vector<std::string> engineOutputTensorNames = {"pred"};

    int  gpu_id   = 0;
    bool ues_fp16 = false;
};

class VectorNet
{
public:
    VectorNet()  = default;
    ~VectorNet() = default;

    /*
     * @description:
     * @param {VectorNetOptions&} options
     * @return {*}
     */
    bool Init(const VectorNetOptions& options);

    /*
     * @description:
     * @param {TrajfeatureInputData&} input_data
     * @return {*}
     */
    bool Process(TrajfeatureInputData& input_data, TrajPredictData& pred_data);

private:
    bool createVectornetEngine();
    bool serializeEngineToFile();
    bool deserializeEngineFromFile();

private:
    VectorNetOptions   m_options_;
    ICudaEngine*       m_engine_;
    IExecutionContext* m_context_;
};