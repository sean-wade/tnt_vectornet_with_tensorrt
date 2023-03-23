#include "cuda_fp16.h"
#include "common/serialize.hpp"
#include "scatterMaxPlugin.h"

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::ScatterMaxPlugin;
using nvinfer1::plugin::ScatterMaxPluginCreator;

namespace
{
char const* kSCATTER_MAX_PLUGIN_NAME{"ScatterMax"};
char const* kSCATTER_MAX_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(float)};
} // namespace

ScatterMaxPlugin::ScatterMaxPlugin(std::string const& name, bool padding) : mName(name)
{
}

IPluginV2DynamicExt* ScatterMaxPlugin::clone() const noexcept
{
    try
    {
        auto plugin = new ScatterMaxPlugin(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t ScatterMaxPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType ScatterMaxPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

DimsExprs
ScatterMaxPlugin::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
    // {
    //     DimsExprs outputDims;
    //     outputDims.nbDims = 4;
    //     outputDims.d[0]   = inputs[2].d[0];
    //     outputDims.d[1]   = inputs[0].d[1];
    //     outputDims.d[2]   = inputs[0].d[2];
    //     outputDims.d[3]   = inputs[0].d[3];
    //     return outputDims;
    // }
}

bool ScatterMaxPlugin::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return ((inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && (inOut[0].format == TensorFormat::kLINEAR))
               || ((inOut[0].type == DataType::kINT8)
                   && (inOut[0].format == TensorFormat::kCHW4 || inOut[0].format == TensorFormat::kCHW32));
    case 1:
    case 2: return (inOut[pos].type == inOut[0].type) || ((inOut[0].type == DataType::kINT8) && (inOut[pos].type == DataType::kHALF));
    case 3: return (inOut[pos].type == inOut[0].type) && (inOut[pos].format == inOut[0].format);
    default: // should NOT be here!
        return false;
    }
    return false;
}

void ScatterMaxPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in,
    int32_t                        nbInputs,
    DynamicPluginTensorDesc const* out,
    int32_t                        nbOutputs) noexcept
{
}

size_t
ScatterMaxPlugin::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs)
    const noexcept
{
    return 0;
}

int32_t ScatterMaxPlugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const*      inputs,
    void* const*            outputs,
    void*                   workspace,
    cudaStream_t            stream) noexcept
{
    // int32_t gridSize = inputDesc[0].dims.d[0];
    // int32_t nHiddenSize = 1;
    // for (int32_t i = 0; i < inputDesc[0].dims.nbDims; ++i)
    // {
    //     std::cout << "inputDesc[0].dims[ " << i << "] = " << inputDesc[0].dims.d[i] << std::endl;
    // }
    // for (int32_t i = 0; i < inputDesc[1].dims.nbDims; ++i)
    // {
    //     std::cout << "inputDesc[1].dims[ " << i << "] = " << inputDesc[1].dims.d[i] << std::endl;
    // }
    // for (int32_t i = 0; i < outputDesc[0].dims.nbDims; ++i)
    // {
    //     std::cout << "outputDesc[0].dims[ " << i << "] = " << outputDesc[0].dims.d[i] << std::endl;
    // }

    int32_t nFeatsNum   = inputDesc[0].dims.d[0]; // -1 (e.g. 396)
    int32_t nHiddenSize = inputDesc[0].dims.d[1]; // 64
    int32_t nClusterNum = inputDesc[2].dims.d[0]; // -1 (e.g.  11)
    int32_t status      = -1;

    switch (inputDesc[0].type)
    {
    case DataType::kFLOAT:
    {
        auto const feature       = static_cast<float const*>(inputs[0]);
        auto const cluster       = static_cast<float const*>(inputs[1]);
        auto const cluster_count = static_cast<float const*>(inputs[2]);
        auto       output        = static_cast<float*>(outputs[0]);

        // std::cout << "nFeatsNum = " << nFeatsNum << "\n";
        // std::cout << "nHiddenSize = " << nHiddenSize << "\n";
        // std::cout << "nClusterNum = " << nClusterNum << "\n";
        status = computeScatterMaxCUDA<float>(nFeatsNum, nHiddenSize, nClusterNum, feature, cluster, cluster_count, output, stream);
        break;
    }
    case DataType::kHALF:
    {
        auto const feature       = static_cast<half const*>(inputs[0]);
        auto const cluster       = static_cast<half const*>(inputs[1]);
        auto const cluster_count = static_cast<half const*>(inputs[2]);
        auto       output        = static_cast<half*>(outputs[0]);

        status = computeScatterMaxCUDA<half>(nFeatsNum, nHiddenSize, nClusterNum, feature, cluster, cluster_count, output, stream);
        break;
    }
    default:
    {
        PLUGIN_FAIL("DataType not implemented yet");
        break;
    }
    }
    return status;
}

void ScatterMaxPlugin::destroy() noexcept
{
    delete this;
}

int32_t ScatterMaxPlugin::initialize() noexcept
{
    return 0;
}

void ScatterMaxPlugin::terminate() noexcept
{
}

size_t ScatterMaxPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void ScatterMaxPlugin::serialize(void* buffer) const noexcept
{
}

void ScatterMaxPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNameSpace = pluginNamespace;
}

char const* ScatterMaxPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

char const* ScatterMaxPlugin::getPluginType() const noexcept
{
    return kSCATTER_MAX_PLUGIN_NAME;
}

char const* ScatterMaxPlugin::getPluginVersion() const noexcept
{
    return kSCATTER_MAX_PLUGIN_VERSION;
}

PluginFieldCollection    ScatterMaxPluginCreator::mFC{};
std::vector<PluginField> ScatterMaxPluginCreator::mPluginAttributes;

ScatterMaxPluginCreator::ScatterMaxPluginCreator()
{
    mPluginAttributes.clear();
    // mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kCHAR, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

ScatterMaxPluginCreator::~ScatterMaxPluginCreator()
{
}

IPluginV2* ScatterMaxPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;

        return new ScatterMaxPlugin(name);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* ScatterMaxPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        PLUGIN_VALIDATE(serialData != nullptr);
        return new ScatterMaxPlugin(name);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* ScatterMaxPluginCreator::getPluginName() const noexcept
{
    return kSCATTER_MAX_PLUGIN_NAME;
}

char const* ScatterMaxPluginCreator::getPluginVersion() const noexcept
{
    return kSCATTER_MAX_PLUGIN_VERSION;
}

PluginFieldCollection const* ScatterMaxPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
