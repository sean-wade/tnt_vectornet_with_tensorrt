/*
 * @Author: zhanghao
 * @LastEditTime: 2023-05-25 12:06:28
 * @FilePath: /cpp/tnt.h
 * @LastEditors: zhanghao
 * @Description:
 */
#include "layer.h"
#include "utils.h"

using namespace nvinfer1;
static Logger gLogger;

struct TrajData
{
    float pred_traj[TNT_HORIZON * 2];
    float score;
};

struct TNTPredictTrajs
{
    std::vector<TrajData> pred_trajs;

    void print()
    {
        for (int i = 0; i < pred_trajs.size(); i++)
        {
            printf("score = [%f]: \n", pred_trajs[i].score);
            for (int j = 0; j < TNT_HORIZON * 2; j++)
            {
                printf("%f, ", pred_trajs[i].pred_traj[j]);
            }
            printf("\n\n");
        }
    }
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
    bool Process(TrajfeatureInputData& input_data, TNTPredictTrajs& pred_data);

private:
    bool createTNTEngine();
    bool serializeEngineToFile();
    bool deserializeEngineFromFile();

private:
    TNTOptions         m_options_;
    ICudaEngine*       m_engine_;
    IExecutionContext* m_context_;
};