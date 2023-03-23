#include <stdio.h>
#include <thrust/count.h>
#include "scatterMaxNoPaddingKernel.h"

template <typename T>
__global__ void ScatterMaxNoPaddingKernel(int featsNum, T const* feature, T const* cluster, T const* clusterCounts, T* output)
{
    int past_feat_nums = 0;
    for (int i = 0; i < blockIdx.x; i++)
    {
        past_feat_nums += int(clusterCounts[i]);
    }
    int cur_start_index = past_feat_nums * blockDim.x + threadIdx.x;

    int cur_cluster_count = int(clusterCounts[int(blockIdx.x)]);

    float max_value = -1e5;
    for (int i = 0; i < cur_cluster_count; i++)
    {
        if (static_cast<float>(feature[cur_start_index + i * blockDim.x]) > max_value)
        {
            max_value = feature[cur_start_index + i * blockDim.x];
        }
    }

    int32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index]       = max_value;
}

template <typename T>
int32_t computeScatterMaxNoPaddingCUDA(
    int32_t const featsNum,
    int32_t const hiddenSize,
    int32_t const clusterNum,
    T const*      feature,
    T const*      cluster,
    T const*      cluster_count,
    T*            output,
    cudaStream_t  stream)
{
    (ScatterMaxNoPaddingKernel<T>)<<<clusterNum, hiddenSize, 0, stream>>>(featsNum, feature, cluster, cluster_count, output);
    cudaStreamSynchronize(stream);
    return 0;
}

template int
computeScatterMaxNoPaddingCUDA<float>(int const, int const, int const, float const*, float const*, float const*, float*, cudaStream_t);
template int
computeScatterMaxNoPaddingCUDA<half>(int const, int const, int const, half const*, half const*, half const*, half*, cudaStream_t);
