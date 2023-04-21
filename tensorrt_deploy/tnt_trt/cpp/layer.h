/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-17 14:20:27
 * @FilePath: /vectornetx/layernorm.cpp
 * @LastEditors: zhanghao
 * @Description: vectornet layers implemention.
 * Including : {scattermax, layernorm, l2norm, mlp, sub_graph, global_graph ......}
 */
#pragma once
#include <math.h>
#include <map>
#include <chrono>
#include <fstream>
#include <iostream>
#include "logging.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "layerNormPlugin/layerNormPlugin.h"
#include "scatterMaxPlugin/scatterMaxPlugin.h"
#include "scatterMaxNoPaddingPlugin/scatterMaxNoPaddingPlugin.h"

using namespace nvinfer1;

static const int BATCH_SIZE = 1;

static const int TNT_HORIZON           = 30;
static const int TNT_CANDIDATE_NUM     = 900; // only surpport fixed now.
static const int TNT_TARGET_SELECT_NUM = 50;

static const int   INPUT_CHANNEL               = 6;
static const int   SUB_GRAPH_HIDDEN_CHANNEL    = 64;
static const int   GLOBAL_GRAPH_HIDDEN_CHANNEL = 64;
static const int   TRAJ_PRED_HIDDEN_CHANNEL    = 64;
static const int   MOTION_EST_HIDDEN_CHANNEL   = 64;
static const int   TRAJ_SCORE_HIDDEN_CHANNEL   = 64;
static const float EPSILON_L2NORM              = 1e-12f;
static const float GRAPH_SCORE_NORM            = 8.0f; // sqrt(GLOBAL_GRAPH_HIDDEN_CHANNEL)
// static const int   FINAL_PRED_CHANNEL    = 50;

static const void printDims(ITensor* const data)
{
    // Attention !!!
    //    Tensorrt-7.2, printDims will cause nan value in ITensor !!!
    //    Tensorrt-8.0, ok.
    Dims dims   = data->getDimensions();
    int  nbDims = dims.nbDims;
    for (int d = 0; d < nbDims; d++)
        std::cout << dims.d[d] << " ";
    std::string sss;
    if (data->getType() == DataType::kHALF)
        sss = "float16";
    if (data->getType() == DataType::kFLOAT)
        sss = "float32";
    std::cout << sss << " ";
    std::cout << std::endl;
}

static ILayer* RepeatConcatLayer(INetworkDefinition* network, ITensor* tensorA, ITensor* tensorB, int repeat_num)
{
    Dims dimsA = tensorA->getDimensions();

    // Repeat
    ISliceLayer* repeatA =
        network->addSlice(*tensorA, Dims4(0, 0, 0, 0), Dims4(repeat_num, dimsA.d[1], 1, 1), Dims4(1, 1, 1, 1));
    repeatA->setMode(SliceMode::kWRAP);

    // Cancat
    ITensor*             inputTensorsCat[] = {repeatA->getOutput(0), tensorB};
    IConcatenationLayer* concat            = network->addConcatenation(inputTensorsCat, 2);

    return concat;
}

static IPluginV2Layer* ScatterMaxPlugin(
    INetworkDefinition* network,
    ITensor*            feats,
    ITensor*            cluster,
    ITensor*            cluster_num,
    int                 hidden_dim = 64,
    bool                padding    = true,
    const std::string&  name       = "scatter_max1")
{
    if (padding)
    {
        auto                         creator        = getPluginRegistry()->getPluginCreator("ScatterMax", "1");
        const PluginFieldCollection* pfc            = creator->getFieldNames();
        IPluginV2*                   pluginObj      = creator->createPlugin(name.c_str(), pfc);
        ITensor*                     inputScatter[] = {feats, cluster, cluster_num};
        auto                         scatter_max    = network->addPluginV2(inputScatter, 3, *pluginObj);
        pluginObj->destroy();
        return scatter_max;
    }
    else
    {
        auto                         creator        = getPluginRegistry()->getPluginCreator("ScatterMaxNoPadding", "1");
        const PluginFieldCollection* pfc            = creator->getFieldNames();
        IPluginV2*                   pluginObj      = creator->createPlugin(name.c_str(), pfc);
        ITensor*                     inputScatter[] = {feats, cluster, cluster_num};
        auto                         scatter_max    = network->addPluginV2(inputScatter, 3, *pluginObj);
        pluginObj->destroy();
        return scatter_max;
    }
}

static IPluginV2Layer* LayerNormPlugin(
    INetworkDefinition*             network,
    ITensor*                        input,
    std::map<std::string, Weights>& weightMap,
    const std::string&              lname,
    int                             hidden_dim = 64)
{
    auto creator = getPluginRegistry()->getPluginCreator("LayerNorm", "1");

    const PluginFieldCollection* pfc       = creator->getFieldNames();
    IPluginV2*                   pluginObj = creator->createPlugin(lname.c_str(), pfc);

    auto     norm1_weights = network->addConstant(Dims{1, hidden_dim}, weightMap[lname + ".weight"]);
    auto     norm1_bias    = network->addConstant(Dims{1, hidden_dim}, weightMap[lname + ".bias"]);
    ITensor* inputNorm1[]  = {input, norm1_weights->getOutput(0), norm1_bias->getOutput(0)};
    auto     norm1         = network->addPluginV2(inputNorm1, 3, *pluginObj);

    pluginObj->destroy();
    return norm1;
}

static ILayer* MlpBlock(
    INetworkDefinition*             network,
    std::map<std::string, Weights>& weightMap,
    ITensor&                        input,
    int                             out_channel = 64,
    int                             hidden_dim  = 64,
    bool                            short_cut   = false,
    std::string                     lname       = "")
{
    IFullyConnectedLayer* fc1 = network->addFullyConnected(
        input, hidden_dim, weightMap[lname + "linear1.weight"], weightMap[lname + "linear1.bias"]);
    assert(fc1);

    IPluginV2Layer* norm1 = LayerNormPlugin(network, fc1->getOutput(0), weightMap, lname + "norm1", hidden_dim);
    assert(norm1);

    IActivationLayer* relu1 = network->addActivation(*norm1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IFullyConnectedLayer* fc2 = network->addFullyConnected(
        *relu1->getOutput(0), out_channel, weightMap[lname + "linear2.weight"], weightMap[lname + "linear2.bias"]);
    assert(fc2);

    IPluginV2Layer* norm2 = LayerNormPlugin(network, fc2->getOutput(0), weightMap, lname + "norm2", out_channel);
    assert(norm2);

    if (short_cut)
    {
        IFullyConnectedLayer* fc3 = network->addFullyConnected(
            input, out_channel, weightMap[lname + "shortcut.0.weight"], weightMap[lname + "shortcut.0.bias"]);
        assert(fc3);

        IPluginV2Layer* norm3 =
            LayerNormPlugin(network, fc3->getOutput(0), weightMap, lname + "shortcut.1", out_channel);
        assert(norm3);

        IElementWiseLayer* ew_add =
            network->addElementWise(*norm3->getOutput(0), *norm2->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew_add);

        IActivationLayer* relu2 = network->addActivation(*ew_add->getOutput(0), ActivationType::kRELU);
        assert(relu2);

        return relu2;
    }
    else
    {
        IElementWiseLayer* ew_add = network->addElementWise(input, *norm2->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew_add);

        IActivationLayer* relu2 = network->addActivation(*ew_add->getOutput(0), ActivationType::kRELU);
        assert(relu2);

        return relu2;
    }
}

static ILayer* L2Normalize(INetworkDefinition* network, ITensor* input)
{
    auto square   = network->addElementWise(*input, *input, ElementWiseOperation::kPROD);
    auto norm2    = network->addReduce(*square->getOutput(0), ReduceOperation::kSUM, 2, 1);
    auto sqrtNorm = network->addUnary(*norm2->getOutput(0), UnaryOperation::kSQRT);

    auto epsTensor = network->addConstant(Dims4{1, 1, 1, 1}, Weights{DataType::kFLOAT, &EPSILON_L2NORM, 1});
    auto sqrtNormEps =
        network->addElementWise(*sqrtNorm->getOutput(0), *epsTensor->getOutput(0), ElementWiseOperation::kSUM);

    IElementWiseLayer* div_norm =
        network->addElementWise(*input, *sqrtNormEps->getOutput(0), ElementWiseOperation::kDIV);
    return div_norm;
}

static ILayer* SubGraph(
    INetworkDefinition*             network,
    std::map<std::string, Weights>& weightMap,
    ITensor*                        feats,
    ITensor*                        cluster,
    ITensor*                        cluster_count,
    int                             out_channel = 64,
    int                             hidden_dim  = 64,
    bool                            short_cut   = false,
    std::string                     lname       = "subgraph.")
{
    // 1. MLP + Scatter + Concat
    ILayer* mlp1 = MlpBlock(network, weightMap, *feats, hidden_dim, hidden_dim, true, lname + "layer_seq.glp_0.");
    IPluginV2Layer* scatter1 =
        ScatterMaxPlugin(network, mlp1->getOutput(0), cluster, cluster_count, hidden_dim, true, "scatter_max1");
    ITensor*             inputTensorsCat1[] = {mlp1->getOutput(0), scatter1->getOutput(0)};
    IConcatenationLayer* concat1            = network->addConcatenation(inputTensorsCat1, 2);
    concat1->setAxis(1);
    assert(mlp1);
    assert(scatter1);
    assert(concat1);

    // 2. MLP + Scatter + Concat
    ILayer* mlp2 =
        MlpBlock(network, weightMap, *concat1->getOutput(0), hidden_dim, hidden_dim, true, lname + "layer_seq.glp_1.");
    IPluginV2Layer* scatter2 =
        ScatterMaxPlugin(network, mlp2->getOutput(0), cluster, cluster_count, hidden_dim, true, "scatter_max2");
    ITensor*             inputTensorsCat2[] = {mlp2->getOutput(0), scatter2->getOutput(0)};
    IConcatenationLayer* concat2            = network->addConcatenation(inputTensorsCat2, 2);
    concat2->setAxis(1);
    assert(mlp2);
    assert(scatter2);
    assert(concat2);

    // 3. MLP + Scatter + Concat
    ILayer* mlp3 =
        MlpBlock(network, weightMap, *concat2->getOutput(0), hidden_dim, hidden_dim, true, lname + "layer_seq.glp_2.");
    IPluginV2Layer* scatter3 =
        ScatterMaxPlugin(network, mlp3->getOutput(0), cluster, cluster_count, hidden_dim, true, "scatter_max3");
    ITensor*             inputTensorsCat3[] = {mlp3->getOutput(0), scatter3->getOutput(0)};
    IConcatenationLayer* concat3            = network->addConcatenation(inputTensorsCat3, 2);
    concat3->setAxis(1);
    assert(mlp3);
    assert(scatter3);
    assert(concat3);

    // 4. Linear
    IFullyConnectedLayer* linear = network->addFullyConnected(
        *concat3->getOutput(0), out_channel, weightMap[lname + "linear.weight"], weightMap[lname + "linear.bias"]);
    assert(linear);

    // 5. Scatter
    IPluginV2Layer* scatter4 =
        ScatterMaxPlugin(network, linear->getOutput(0), cluster, cluster_count, hidden_dim, false, "scatter_max4");
    assert(scatter4);

    // 6.  normalize
    ILayer* l2norm = L2Normalize(network, scatter4->getOutput(0));
    assert(l2norm);

    return l2norm;
}

static ILayer* GlobalGraph(
    INetworkDefinition*             network,
    std::map<std::string, Weights>& weightMap,
    ITensor*                        sub_graph_out,
    ITensor*                        id_embedding,
    int                             global_graph_channel = 64,
    std::string                     lname                = "global_graph.")
{
    ITensor*             inputTensorsGlobal[] = {sub_graph_out, id_embedding};
    IConcatenationLayer* concat_embedding     = network->addConcatenation(inputTensorsGlobal, 2);
    concat_embedding->setAxis(1);

    IFullyConnectedLayer* q_linear = network->addFullyConnected(
        *concat_embedding->getOutput(0),
        global_graph_channel,
        weightMap[lname + "layers.glp_0.q_lin.weight"],
        weightMap[lname + "layers.glp_0.q_lin.bias"]);
    assert(q_linear);

    IFullyConnectedLayer* k_linear = network->addFullyConnected(
        *concat_embedding->getOutput(0),
        global_graph_channel,
        weightMap[lname + "layers.glp_0.k_lin.weight"],
        weightMap[lname + "layers.glp_0.k_lin.bias"]);
    assert(k_linear);

    IFullyConnectedLayer* v_linear = network->addFullyConnected(
        *concat_embedding->getOutput(0),
        global_graph_channel,
        weightMap[lname + "layers.glp_0.v_lin.weight"],
        weightMap[lname + "layers.glp_0.v_lin.bias"]);
    assert(v_linear);

    IShuffleLayer* q_shuffle = network->addShuffle(*q_linear->getOutput(0));
    q_shuffle->setReshapeDimensions(Dims4{1, 1, -1, global_graph_channel});
    IShuffleLayer* k_shuffle = network->addShuffle(*k_linear->getOutput(0));
    k_shuffle->setReshapeDimensions(Dims4{1, 1, -1, global_graph_channel});
    IShuffleLayer* v_shuffle = network->addShuffle(*v_linear->getOutput(0));
    v_shuffle->setReshapeDimensions(Dims4{1, 1, -1, global_graph_channel});

    IMatrixMultiplyLayer* scores = network->addMatrixMultiply(
        *q_shuffle->getOutput(0), MatrixOperation::kNONE, *k_shuffle->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(scores);

    IConstantLayer* normScore =
        network->addConstant(Dims4{1, 1, 1, 1}, Weights{scores->getOutput(0)->getType(), &GRAPH_SCORE_NORM, 1});
    IElementWiseLayer* scores_normed =
        network->addElementWise(*scores->getOutput(0), *normScore->getOutput(0), ElementWiseOperation::kDIV);
    assert(normScore);

    ISoftMaxLayer* attention_score = network->addSoftMax(*scores_normed->getOutput(0));
    attention_score->setAxes(1 << 3);
    assert(attention_score);

    IMatrixMultiplyLayer* global_out = network->addMatrixMultiply(
        *attention_score->getOutput(0), MatrixOperation::kNONE, *v_shuffle->getOutput(0), MatrixOperation::kNONE);
    assert(global_out);

    // printDims(global_out->getOutput(0));

    IShuffleLayer* global_out_reshape = network->addShuffle(*global_out->getOutput(0));
    global_out_reshape->setReshapeDimensions(Dims4{-1, global_graph_channel, 1, 1});

    ISliceLayer* slice_target = network->addSlice(
        *global_out_reshape->getOutput(0), Dims4(0, 0, 0, 0), Dims4(1, global_graph_channel, 1, 1), Dims4(1, 1, 1, 1));
    assert(slice_target);

    return slice_target;
}

static ILayer* TargetPredLayer(
    INetworkDefinition*             network,
    std::map<std::string, Weights>& weightMap,
    ITensor*                        target_global_feat,
    ITensor*                        candidate_points,
    int                             candidate_num,
    int                             target_select_num = 50,
    int                             hidden_channel    = 64,
    std::string                     lname             = "target_pred_layer.")
{
    ILayer* feats_candidate_concat = RepeatConcatLayer(network, target_global_feat, candidate_points, candidate_num);

    ILayer* prob_mlp = MlpBlock(
        network,
        weightMap,
        *feats_candidate_concat->getOutput(0),
        hidden_channel,
        hidden_channel,
        true,
        lname + "prob_mlp.0.");

    IFullyConnectedLayer* prob_fc = network->addFullyConnected(
        *prob_mlp->getOutput(0), 1, weightMap[lname + "prob_mlp.1.weight"], weightMap[lname + "prob_mlp.1.bias"]);

    ISoftMaxLayer* candit_prob = network->addSoftMax(*prob_fc->getOutput(0));
    candit_prob->setAxes(1 << 0);

    ILayer* offset_mlp = MlpBlock(
        network,
        weightMap,
        *feats_candidate_concat->getOutput(0),
        hidden_channel,
        hidden_channel,
        true,
        lname + "mean_mlp.0.");

    IFullyConnectedLayer* candit_offset = network->addFullyConnected(
        *offset_mlp->getOutput(0), 2, weightMap[lname + "mean_mlp.1.weight"], weightMap[lname + "mean_mlp.1.bias"]);

    ITopKLayer* topk = network->addTopK(*candit_prob->getOutput(0), TopKOperation::kMAX, target_select_num, 0X01);

    IGatherLayer* topk_target = network->addGather(*candidate_points, *topk->getOutput(1), 0);
    IGatherLayer* topk_offset = network->addGather(*candit_offset->getOutput(0), *topk->getOutput(1), 0);

    IElementWiseLayer* topk_target_pred =
        network->addElementWise(*topk_target->getOutput(0), *topk_offset->getOutput(0), ElementWiseOperation::kSUM);

    IShuffleLayer* topk_target_pred_reshape = network->addShuffle(*topk_target_pred->getOutput(0));
    topk_target_pred_reshape->setReshapeDimensions(Dims4{target_select_num, 2, 1, 1});

    return topk_target_pred_reshape;
}

static ILayer* MotionEstimationLayer(
    INetworkDefinition*             network,
    std::map<std::string, Weights>& weightMap,
    ITensor*                        target_global_feat,
    ITensor*                        topk_target_pred,
    int                             horizon           = 30,
    int                             target_select_num = 50,
    int                             hidden_channel    = 64,
    std::string                     lname             = "motion_estimator.")
{
    ILayer* feats_target_concat = RepeatConcatLayer(network, target_global_feat, topk_target_pred, target_select_num);

    ILayer* traj_pred_mlp = MlpBlock(
        network,
        weightMap,
        *feats_target_concat->getOutput(0),
        hidden_channel,
        hidden_channel,
        true,
        lname + "traj_pred.0.");

    IFullyConnectedLayer* traj_pred_fc = network->addFullyConnected(
        *traj_pred_mlp->getOutput(0),
        horizon * 2,
        weightMap[lname + "traj_pred.1.weight"],
        weightMap[lname + "traj_pred.1.bias"]);

    return traj_pred_fc;
}

static ILayer* TrajScoreLayer(
    INetworkDefinition*             network,
    std::map<std::string, Weights>& weightMap,
    ITensor*                        target_global_feat,
    ITensor*                        traj_topk_pred,
    int                             target_select_num = 50,
    int                             hidden_channel    = 64,
    std::string                     lname             = "traj_score_layer.")
{
    ILayer* feats_traj_concat = RepeatConcatLayer(network, target_global_feat, traj_topk_pred, target_select_num);

    ILayer* traj_score_mlp = MlpBlock(
        network,
        weightMap,
        *feats_traj_concat->getOutput(0),
        hidden_channel,
        hidden_channel,
        true,
        lname + "score_mlp.0.");

    IFullyConnectedLayer* traj_score_fc = network->addFullyConnected(
        *traj_score_mlp->getOutput(0),
        1,
        weightMap[lname + "score_mlp.1.weight"],
        weightMap[lname + "score_mlp.1.bias"]);

    ISoftMaxLayer* traj_score = network->addSoftMax(*traj_score_fc->getOutput(0));
    traj_score->setAxes(1 << 0);

    return traj_score;
}

// static const float SCALING_ONE = 1.0;
// static const float SHIFT_ZERO = 0.0;
// static const float POWER_TWO = 2.0;
// static const float EPS = 0.0000001;
// ITensor* LayerNorm_MY(
//     INetworkDefinition*             network,
//     ITensor&                        input,
//     std::map<std::string, Weights>& weightMap,
//     const std::string&              lname,
//     int                             hidden_dim = 64)
// {
//     // maybe a better implementation
//     // https://github.com/NVIDIA/TensorRT/blob/master/plugin/common/common.cuh#212
//     IReduceLayer* mean = network->addReduce(input, ReduceOperation::kAVG, 2, true);
//     assert(mean);

//     IElementWiseLayer* sub_mean = network->addElementWise(input, *mean->getOutput(0),
//     ElementWiseOperation::kSUB); assert(sub_mean);

//     // for (int i = 0; i < sub_mean->getOutput(0)->getDimensions().nbDims; i++)
//     // {
//     //     std::cout << sub_mean->getOutput(0)->getDimensions().d[i] << ",";
//     // }

//     // implement pow2 with scale
//     Weights scale{DataType::kFLOAT, &SCALING_ONE, 1};
//     Weights shift{DataType::kFLOAT, &SHIFT_ZERO, 1};
//     Weights power{DataType::kFLOAT, &POWER_TWO, 1};
//     auto    pow2 = network->addScaleNd(*sub_mean->getOutput(0), ScaleMode::kUNIFORM, shift, scale, power, 1);
//     assert(pow2);

//     auto pow_mean = network->addReduce(*pow2->getOutput(0), ReduceOperation::kAVG, 2, true);
//     assert(pow_mean);

//     auto eps = network->addConstant(Dims4{1, 1, 1, 1}, Weights{DataType::kFLOAT, &EPSILON_L2NORM, 1});
//     assert(eps);

//     auto add_eps = network->addElementWise(*pow_mean->getOutput(0), *eps->getOutput(0),
//     ElementWiseOperation::kSUM); assert(add_eps);

//     auto sqrt = network->addUnary(*add_eps->getOutput(0), UnaryOperation::kSQRT);
//     assert(sqrt);

//     auto div = network->addElementWise(*sub_mean->getOutput(0), *sqrt->getOutput(0), ElementWiseOperation::kDIV);
//     assert(div);

//     float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * hidden_dim));
//     for (int i = 0; i < hidden_dim; i++) { pval[i] = 1.0; }
//     Weights norm1_power{DataType::kFLOAT, pval, hidden_dim};

//     auto affine = network->addScaleNd(
//         *div->getOutput(0),
//         ScaleMode::kCHANNEL,
//         weightMap[lname + ".bias"],
//         weightMap[lname + ".weight"],
//         norm1_power,
//         1);
//     assert(affine);

//     return affine->getOutput(0);
// }

// TRT network->addSlice must specify the size with constant value.
// static ILayer* RepeatConcatLayer(INetworkDefinition* network, ITensor* tensorA, ITensor* tensorB)
// {
//     Dims dimsA = tensorA->getDimensions();
//     Dims dimsB = tensorB->getDimensions();

//     // printDims(tensorA);
//     // printDims(tensorB);
//     int repeat_num = dimsB.d[0] > 0 ? dimsB.d[0] : 1;

//     // Repeat
//     ISliceLayer* repeatA =
//         network->addSlice(*tensorA, Dims4(0, 0, 0, 0), Dims4(repeat_num, dimsA.d[1], 1, 1), Dims4(1, 1, 1, 1));
//     repeatA->setMode(SliceMode::kWRAP);

//     // printDims(repeatA->getOutput(0));

//     // Cancat
//     ITensor*             inputTensorsCat[] = {repeatA->getOutput(0), tensorB};
//     IConcatenationLayer* concat            = network->addConcatenation(inputTensorsCat, 1);

//     return concat;
// }