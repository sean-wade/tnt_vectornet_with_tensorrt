/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-21 11:30:22
 * @FilePath: /vectornetx/Plugin/scatterMaxPlugin/scatterMaxKernel.h
 * @LastEditors: zhanghao
 * @Description:
 */
#ifndef TRT_SCATTERMAX_KERNEL_H
#define TRT_SCATTERMAX_KERNEL_H

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <cuda.h>
#include "cuda_fp16.h"

using half = __half;

template <typename T>
int32_t computeScatterMaxCUDA(
    int32_t const featsNum,
    int32_t const hiddenSize,
    T const*      feature,
    T const*      cluster,
    T*            output,
    cudaStream_t  stream);

#endif // TRT_SCATTERMAX_KERNEL_H
