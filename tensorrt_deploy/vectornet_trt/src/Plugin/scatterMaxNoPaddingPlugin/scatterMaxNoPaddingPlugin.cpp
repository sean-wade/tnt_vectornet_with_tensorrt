#include "cuda_fp16.h"
#include "common/serialize.hpp"
#include "scatterMaxNoPaddingPlugin.h"

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::ScatterMaxNoPaddingPlugin;
using nvinfer1::plugin::ScatterMaxNoPaddingPluginCreator;

namespace
{
char const* kSCATTER_MAX_PLUGIN_NAME{"ScatterMaxNoPadding"};
char const* kSCATTER_MAX_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(float)};
} // namespace

ScatterMaxNoPaddingPlugin::ScatterMaxNoPaddingPlugin(std::string const& name, bool padding) : mName(name)
{
}

IPluginV2DynamicExt* ScatterMaxNoPaddingPlugin::clone() const noexcept
{
    try
    {
        auto plugin = new ScatterMaxNoPaddingPlugin(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t ScatterMaxNoPaddingPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType ScatterMaxNoPaddingPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

DimsExprs ScatterMaxNoPaddingPlugin::getOutputDimensions(
    int32_t          outputIndex,
    DimsExprs const* inputs,
    int32_t          nbInputs,
    IExprBuilder&    exprBuilder) noexcept
{
    DimsExprs outputDims;
    outputDims.nbDims = 4;
    outputDims.d[0]   = inputs[2].d[0];
    outputDims.d[1]   = inputs[0].d[1];
    outputDims.d[2]   = inputs[0].d[2];
    outputDims.d[3]   = inputs[0].d[3];
    return outputDims;
}

bool ScatterMaxNoPaddingPlugin::supportsFormatCombination(
    int32_t                 pos,
    PluginTensorDesc const* inOut,
    int32_t                 nbInputs,
    int32_t                 nbOutputs) noexcept
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

void ScatterMaxNoPaddingPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in,
    int32_t                        nbInputs,
    DynamicPluginTensorDesc const* out,
    int32_t                        nbOutputs) noexcept
{
}

size_t ScatterMaxNoPaddingPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs,
    int32_t                 nbInputs,
    PluginTensorDesc const* outputs,
    int32_t                 nbOutputs) const noexcept
{
    return 0;
}

int32_t ScatterMaxNoPaddingPlugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const*      inputs,
    void* const*            outputs,
    void*                   workspace,
    cudaStream_t            stream) noexcept
{
    int32_t nFeatsNum   = inputDesc[0].dims.d[0];
    int32_t nHiddenSize = inputDesc[0].dims.d[1];
    int32_t nClusterNum = inputDesc[2].dims.d[0];
    int32_t status      = -1;

    switch (inputDesc[0].type)
    {
    case DataType::kFLOAT:
    {
        auto const feature       = static_cast<float const*>(inputs[0]);
        auto const cluster       = static_cast<float const*>(inputs[1]);
        auto const cluster_count = static_cast<float const*>(inputs[2]);
        auto       output        = static_cast<float*>(outputs[0]);

        status =
            computeScatterMaxNoPaddingCUDA<float>(nFeatsNum, nHiddenSize, nClusterNum, feature, cluster, cluster_count, output, stream);
        break;
    }
    case DataType::kHALF:
    {
        auto const feature       = static_cast<half const*>(inputs[0]);
        auto const cluster       = static_cast<half const*>(inputs[1]);
        auto const cluster_count = static_cast<half const*>(inputs[2]);
        auto       output        = static_cast<half*>(outputs[0]);

        status = computeScatterMaxNoPaddingCUDA<half>(nFeatsNum, nHiddenSize, nClusterNum, feature, cluster, cluster_count, output, stream);
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

void ScatterMaxNoPaddingPlugin::destroy() noexcept
{
    delete this;
}

int32_t ScatterMaxNoPaddingPlugin::initialize() noexcept
{
    return 0;
}

void ScatterMaxNoPaddingPlugin::terminate() noexcept
{
}

size_t ScatterMaxNoPaddingPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void ScatterMaxNoPaddingPlugin::serialize(void* buffer) const noexcept
{
}

void ScatterMaxNoPaddingPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNameSpace = pluginNamespace;
}

char const* ScatterMaxNoPaddingPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

char const* ScatterMaxNoPaddingPlugin::getPluginType() const noexcept
{
    return kSCATTER_MAX_PLUGIN_NAME;
}

char const* ScatterMaxNoPaddingPlugin::getPluginVersion() const noexcept
{
    return kSCATTER_MAX_PLUGIN_VERSION;
}

PluginFieldCollection    ScatterMaxNoPaddingPluginCreator::mFC{};
std::vector<PluginField> ScatterMaxNoPaddingPluginCreator::mPluginAttributes;

ScatterMaxNoPaddingPluginCreator::ScatterMaxNoPaddingPluginCreator()
{
    mPluginAttributes.clear();
    // mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kCHAR, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

ScatterMaxNoPaddingPluginCreator::~ScatterMaxNoPaddingPluginCreator()
{
}

IPluginV2* ScatterMaxNoPaddingPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;

        return new ScatterMaxNoPaddingPlugin(name);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* ScatterMaxNoPaddingPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        PLUGIN_VALIDATE(serialData != nullptr);
        return new ScatterMaxNoPaddingPlugin(name);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* ScatterMaxNoPaddingPluginCreator::getPluginName() const noexcept
{
    return kSCATTER_MAX_PLUGIN_NAME;
}

char const* ScatterMaxNoPaddingPluginCreator::getPluginVersion() const noexcept
{
    return kSCATTER_MAX_PLUGIN_VERSION;
}

PluginFieldCollection const* ScatterMaxNoPaddingPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
