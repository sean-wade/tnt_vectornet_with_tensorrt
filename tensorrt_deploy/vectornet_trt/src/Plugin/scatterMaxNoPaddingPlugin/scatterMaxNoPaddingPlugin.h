/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-20 19:43:24
 * @FilePath: /vectornetx/scatterMaxPlugin/scatterMaxPlugin.h
 * @LastEditors: zhanghao
 * @Description: Scatter_max_pooling without padding plugin.
 */
#ifndef TRT_SCATTERMAX_NOPADDING_PLUGIN_H
#define TRT_SCATTERMAX_NOPADDING_PLUGIN_H

#include "common/plugin.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "scatterMaxNoPaddingKernel.h"

namespace nvinfer1
{
namespace plugin
{
class ScatterMaxNoPaddingPlugin : public IPluginV2DynamicExt
{
public:
    ScatterMaxNoPaddingPlugin() = delete;
    ScatterMaxNoPaddingPlugin(std::string const& name, bool padding = true);
    ~ScatterMaxNoPaddingPlugin() override = default;

    // Method inherited from IPluginV2
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    size_t      getSerializationSize() const noexcept override;
    void        serialize(void* buffer) const noexcept override;
    void        destroy() noexcept override;
    void        setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // Method inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs
         getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void
    configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
        override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs)
        const noexcept override;
    int32_t enqueue(
        PluginTensorDesc const* inputDesc,
        PluginTensorDesc const* outputDesc,
        void const* const*      inputs,
        void* const*            outputs,
        void*                   workspace,
        cudaStream_t            stream) noexcept override;

private:
    const std::string mName;
    std::string       mNameSpace;
};

class ScatterMaxNoPaddingPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    ScatterMaxNoPaddingPluginCreator();
    ~ScatterMaxNoPaddingPluginCreator();
    char const*                  getPluginName() const noexcept override;
    char const*                  getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV2*                   createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2*                   deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection    mFC;
    static std::vector<PluginField> mPluginAttributes;
};
REGISTER_TENSORRT_PLUGIN(ScatterMaxNoPaddingPluginCreator);
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SCATTERMAX_NOPADDING_PLUGIN_H
