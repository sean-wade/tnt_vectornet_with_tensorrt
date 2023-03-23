/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-17 14:20:27
 * @FilePath: /vectornetx/layernorm.cpp
 * @LastEditors: zhanghao
 * @Description:
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

static const float EPSILON          = 1e-12f;
static const float GRAPH_SCORE_NORM = 8.0f;

// Attention: printDims will cause nan values in data!!!
static const void printDims(ITensor* const data)
{
    Dims dims   = data->getDimensions();
    int  nbDims = dims.nbDims;
    for (int d = 0; d < nbDims; d++)
        std::cout << dims.d[d] << " "; // << dims.d[1] << " " << dims.d[2] << " " << dims.d[3] << std::endl;
    std::string sss;
    if (data->getType() == DataType::kHALF)
        sss = "float16";
    if (data->getType() == DataType::kFLOAT)
        sss = "float32";
    std::cout << sss << " ";
    std::cout << std::endl;
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
    IFullyConnectedLayer* fc1 =
        network->addFullyConnected(input, hidden_dim, weightMap[lname + "linear1.weight"], weightMap[lname + "linear1.bias"]);
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
        IFullyConnectedLayer* fc3 =
            network->addFullyConnected(input, out_channel, weightMap[lname + "shortcut.0.weight"], weightMap[lname + "shortcut.0.bias"]);
        assert(fc3);

        IPluginV2Layer* norm3 = LayerNormPlugin(network, fc3->getOutput(0), weightMap, lname + "shortcut.1", out_channel);
        assert(norm3);

        IElementWiseLayer* ew_add = network->addElementWise(*norm3->getOutput(0), *norm2->getOutput(0), ElementWiseOperation::kSUM);
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

    auto epsTensor   = network->addConstant(Dims4{1, 1, 1, 1}, Weights{DataType::kFLOAT, &EPSILON, 1});
    auto sqrtNormEps = network->addElementWise(*sqrtNorm->getOutput(0), *epsTensor->getOutput(0), ElementWiseOperation::kSUM);

    // // printDims(sqrtNormEps->getOutput(0));

    IElementWiseLayer* div_norm = network->addElementWise(*input, *sqrtNormEps->getOutput(0), ElementWiseOperation::kDIV);
    return div_norm;

    // auto scaleLayer =
    //     network->addScaleNd(*inputTensor, ScaleMode::kUNIFORM, Dims{mInputDims.nbDims}, broadcastLayer->getOutput(0), nullptr);

    // return scaleLayer;
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
    ILayer*              mlp1     = MlpBlock(network, weightMap, *feats, hidden_dim, hidden_dim, true, lname + "layer_seq.glp_0.");
    IPluginV2Layer*      scatter1 = ScatterMaxPlugin(network, mlp1->getOutput(0), cluster, cluster_count, hidden_dim, true, "scatter_max1");
    ITensor*             inputTensorsCat1[] = {mlp1->getOutput(0), scatter1->getOutput(0)};
    IConcatenationLayer* concat1            = network->addConcatenation(inputTensorsCat1, 2);
    concat1->setAxis(1);
    assert(mlp1);
    assert(scatter1);
    assert(concat1);

    // 2. MLP + Scatter + Concat
    ILayer*         mlp2 = MlpBlock(network, weightMap, *concat1->getOutput(0), hidden_dim, hidden_dim, true, lname + "layer_seq.glp_1.");
    IPluginV2Layer* scatter2 = ScatterMaxPlugin(network, mlp2->getOutput(0), cluster, cluster_count, hidden_dim, true, "scatter_max2");
    ITensor*        inputTensorsCat2[] = {mlp2->getOutput(0), scatter2->getOutput(0)};
    IConcatenationLayer* concat2       = network->addConcatenation(inputTensorsCat2, 2);
    concat2->setAxis(1);
    assert(mlp2);
    assert(scatter2);
    assert(concat2);

    // 3. MLP + Scatter + Concat
    ILayer*         mlp3 = MlpBlock(network, weightMap, *concat2->getOutput(0), hidden_dim, hidden_dim, true, lname + "layer_seq.glp_2.");
    IPluginV2Layer* scatter3 = ScatterMaxPlugin(network, mlp3->getOutput(0), cluster, cluster_count, hidden_dim, true, "scatter_max3");
    ITensor*        inputTensorsCat3[] = {mlp3->getOutput(0), scatter3->getOutput(0)};
    IConcatenationLayer* concat3       = network->addConcatenation(inputTensorsCat3, 2);
    concat3->setAxis(1);
    assert(mlp3);
    assert(scatter3);
    assert(concat3);

#ifdef DEBUG_BEFORE_FINAL_SCATTER
    return mlp3;
#else

    // 4. Linear
    IFullyConnectedLayer* linear = network->addFullyConnected(
        *concat3->getOutput(0), out_channel, weightMap[lname + "linear.weight"], weightMap[lname + "linear.bias"]);
    assert(linear);

    // 5. Scatter
    IPluginV2Layer* scatter4 = ScatterMaxPlugin(network, linear->getOutput(0), cluster, cluster_count, hidden_dim, false, "scatter_max4");
    assert(scatter4);

    // 6.  normalize
    ILayer* l2norm = L2Normalize(network, scatter4->getOutput(0));
    assert(l2norm);

    return l2norm;
#endif
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

    // IShuffleLayer* q_shuffle = network->addShuffle(*q_linear->getOutput(0));
    // q_shuffle->setReshapeDimensions(Dims3{1, -1, global_graph_channel});
    // IShuffleLayer* k_shuffle = network->addShuffle(*k_linear->getOutput(0));
    // k_shuffle->setReshapeDimensions(Dims3{1, -1, global_graph_channel});
    // IShuffleLayer* v_shuffle = network->addShuffle(*v_linear->getOutput(0));
    // v_shuffle->setReshapeDimensions(Dims3{1, -1, global_graph_channel});

    // IMatrixMultiplyLayer* scores =
    //     network->addMatrixMultiply(*q_shuffle->getOutput(0), MatrixOperation::kNONE, *k_shuffle->getOutput(0),
    //     MatrixOperation::kTRANSPOSE);
    // assert(scores);

    // float norm_coeff    = sqrt(global_graph_channel);
    // auto  normScore     = network->addConstant(Dims3{1, 1, 1}, Weights{DataType::kFLOAT, &norm_coeff, 1});
    // auto  scores_normed = network->addElementWise(*scores->getOutput(0), *normScore->getOutput(0), ElementWiseOperation::kDIV);
    // assert(scores_normed);

    // ISoftMaxLayer* attention_score = network->addSoftMax(*scores_normed->getOutput(0));
    // attention_score->setAxes(4);
    // assert(attention_score);

    // // printDims(attention_score->getOutput(0));
    // // printDims(v_shuffle->getOutput(0));

    // IMatrixMultiplyLayer* global_out = network->addMatrixMultiply(
    //     *attention_score->getOutput(0), MatrixOperation::kNONE, *v_shuffle->getOutput(0), MatrixOperation::kNONE);
    // assert(global_out);

    // // ISliceLayer* slice_target =
    // //     network->addSlice(*global_out->getOutput(0), Dims3(0, 0, 0), Dims3(1, 1, global_graph_channel), Dims3(1, 1, 1));
    // // assert(slice_target);

    // return global_out;

    // // 下面的方法无法切片操作， Slice 后都是 nan IShuffleLayer* q_shuffle = network->addShuffle(*q_linear->getOutput(0));
    IShuffleLayer* q_shuffle = network->addShuffle(*q_linear->getOutput(0));
    q_shuffle->setReshapeDimensions(Dims4{1, 1, -1, global_graph_channel});
    IShuffleLayer* k_shuffle = network->addShuffle(*k_linear->getOutput(0));
    k_shuffle->setReshapeDimensions(Dims4{1, 1, -1, global_graph_channel});
    IShuffleLayer* v_shuffle = network->addShuffle(*v_linear->getOutput(0));
    v_shuffle->setReshapeDimensions(Dims4{1, 1, -1, global_graph_channel});

    IMatrixMultiplyLayer* scores =
        network->addMatrixMultiply(*q_shuffle->getOutput(0), MatrixOperation::kNONE, *k_shuffle->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(scores);

    IConstantLayer*    normScore = network->addConstant(Dims4{1, 1, 1, 1}, Weights{scores->getOutput(0)->getType(), &GRAPH_SCORE_NORM, 1});
    IElementWiseLayer* scores_normed = network->addElementWise(*scores->getOutput(0), *normScore->getOutput(0), ElementWiseOperation::kDIV);
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

    ISliceLayer* slice_target =
        network->addSlice(*global_out_reshape->getOutput(0), Dims4(0, 0, 0, 0), Dims4(1, global_graph_channel, 1, 1), Dims4(1, 1, 1, 1));
    assert(slice_target);

    return slice_target;
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

//     IElementWiseLayer* sub_mean = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
//     assert(sub_mean);

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

//     auto eps = network->addConstant(Dims4{1, 1, 1, 1}, Weights{DataType::kFLOAT, &EPSILON, 1});
//     assert(eps);

//     auto add_eps = network->addElementWise(*pow_mean->getOutput(0), *eps->getOutput(0), ElementWiseOperation::kSUM);
//     assert(add_eps);

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