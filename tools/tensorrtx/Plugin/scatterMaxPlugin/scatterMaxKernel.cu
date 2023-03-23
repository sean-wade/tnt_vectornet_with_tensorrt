#include <stdio.h>
#include <thrust/count.h>
#include "scatterMaxKernel.h"

template <typename T>
__global__ void ScatterMaxKernel(int featsNum, T const* feature, T const* cluster, T const* clusterCounts, T* output)
{
    // printf(">>>>>>>>>>>> blockIdx.x = [%d], threadIdx.x = [%d]\n", int(blockIdx.x), int(threadIdx.x));
    // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    // {
    //     for (int i = 0; i < gridDim.x; i++)
    //     {
    //         //
    //         printf(">>>>>>>>>>>> clusterCounts[%d] = %d\n", i, clusterCounts[i]);
    //     }
    // }

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

// template <typename T>
// __global__ void CalcClusterNumKernel(T const* cluster, int* clusterCounts)
// {
//     int32_t const index = threadIdx.x + blockDim.x * blockIdx.x;
//     atomicAdd(clusterCounts + int(cluster[index]), 1);
//     __syncthreads();
//     // printf(">>>> >>>> cluster[%d] = %d\n", index, int(cluster[index]));
// }

// __global__ void reset_zero(int* clusterCounts, int clusterNum)
// {
//     for (int j = 0; j < clusterNum; j++)
//     {
//         clusterCounts[j] = 0;
//     }
// }

template <typename T>
__global__ void ScatterCopyKernel(T const* output_cluster, T const* cluster, T* output)
{
    int32_t const o_index = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t const c_index = threadIdx.x + blockDim.x * int(cluster[blockIdx.x]);

    output[o_index] = output_cluster[c_index];
}

template <typename T>
int32_t computeScatterMaxCUDA(
    int32_t const featsNum,
    int32_t const hiddenSize,
    int32_t const clusterNum,
    T const*      feature,
    T const*      cluster,
    T const*      cluster_count,
    T*            output,
    cudaStream_t  stream)
{
    T* output_cluster = NULL;
    cudaMalloc(&output_cluster, size_t(clusterNum * hiddenSize * sizeof(T)));

    // printf(">>>> clusterNum = %d\n", clusterNum);

    (ScatterMaxKernel<T>)<<<clusterNum, hiddenSize, 0, stream>>>(featsNum, feature, cluster, cluster_count, output_cluster);

    cudaStreamSynchronize(stream);

    (ScatterCopyKernel<T>)<<<featsNum, hiddenSize, 0, stream>>>(output_cluster, cluster, output);

    cudaStreamSynchronize(stream);

    cudaFree(output_cluster);
    return 0;
}

template int computeScatterMaxCUDA<float>(int const, int const, int const, float const*, float const*, float const*, float*, cudaStream_t);
template int computeScatterMaxCUDA<half>(int const, int const, int const, half const*, half const*, half const*, half*, cudaStream_t);
