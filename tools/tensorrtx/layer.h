/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-17 14:20:27
 * @FilePath: /vectornetx/layernorm.cpp
 * @LastEditors: zhanghao
 * @Description:
 */
#pragma once
#include <map>
#include <chrono>
#include <fstream>
#include <iostream>
#include "logging.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "layerNormPlugin/layerNormPlugin.h"
#include "scatterMaxPlugin/scatterMaxPlugin.h"

using namespace nvinfer1;

IPluginV2Layer* ScatterMax_Plugin(
    INetworkDefinition* network,
    ITensor*            feats,
    ITensor*            cluster,
    int                 hidden_dim = 64,
    const std::string&  name = "scatter_max1")
{
    auto creator = getPluginRegistry()->getPluginCreator("ScatterMax", "1");

    const PluginFieldCollection* pfc = creator->getFieldNames();
    IPluginV2*                   pluginObj = creator->createPlugin(name.c_str(), pfc);

    ITensor* inputScatter[] = {feats, cluster};
    auto     scatter_max1 = network->addPluginV2(inputScatter, 2, *pluginObj);

    pluginObj->destroy();
    return scatter_max1;
}

IPluginV2Layer* LayerNorm_Plugin(
    INetworkDefinition*             network,
    ITensor*                        input,
    std::map<std::string, Weights>& weightMap,
    const std::string&              lname,
    int                             hidden_dim = 64)
{
    auto creator = getPluginRegistry()->getPluginCreator("LayerNorm", "1");

    const PluginFieldCollection* pfc = creator->getFieldNames();
    IPluginV2*                   pluginObj = creator->createPlugin(lname.c_str(), pfc);

    auto     norm1_weights = network->addConstant(Dims{1, hidden_dim}, weightMap[lname + ".weight"]);
    auto     norm1_bias = network->addConstant(Dims{1, hidden_dim}, weightMap[lname + ".bias"]);
    ITensor* inputNorm1[] = {input, norm1_weights->getOutput(0), norm1_bias->getOutput(0)};
    auto     norm1 = network->addPluginV2(inputNorm1, 3, *pluginObj);

    pluginObj->destroy();
    return norm1;
}

static ILayer* MLP_block(
    INetworkDefinition*             network,
    std::map<std::string, Weights>& weightMap,
    ITensor&                        input,
    int                             out_channel = 64,
    int                             hidden_dim = 64,
    bool                            short_cut = false,
    std::string                     lname = "")
{
    IFullyConnectedLayer* fc1 = network->addFullyConnected(
        input, hidden_dim, weightMap[lname + "linear1.weight"], weightMap[lname + "linear1.bias"]);
    assert(fc1);

    IPluginV2Layer* norm1 = LayerNorm_Plugin(network, fc1->getOutput(0), weightMap, lname + "norm1", hidden_dim);
    assert(norm1);

    IActivationLayer* relu1 = network->addActivation(*norm1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IFullyConnectedLayer* fc2 = network->addFullyConnected(
        *relu1->getOutput(0), out_channel, weightMap[lname + "linear2.weight"], weightMap[lname + "linear2.bias"]);
    assert(fc2);

    IPluginV2Layer* norm2 = LayerNorm_Plugin(network, fc2->getOutput(0), weightMap, lname + "norm2", out_channel);
    assert(norm2);

    if (short_cut)
    {
        IFullyConnectedLayer* fc3 = network->addFullyConnected(
            input, out_channel, weightMap[lname + "shortcut.0.weight"], weightMap[lname + "shortcut.0.bias"]);
        assert(fc3);

        IPluginV2Layer* norm3 =
            LayerNorm_Plugin(network, fc3->getOutput(0), weightMap, lname + "shortcut.1", out_channel);
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

//     auto eps = network->addConstant(Dims4{1, 1, 1, 1}, Weights{DataType::kFLOAT, &EPS, 1});
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