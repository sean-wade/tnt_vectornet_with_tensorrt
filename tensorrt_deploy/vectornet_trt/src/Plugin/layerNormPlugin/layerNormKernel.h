/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-24 15:37:57
 * @FilePath: /vectornetx/Plugin/layerNormPlugin/layerNormKernel.h
 * @LastEditors: zhanghao
 * @Description:
 */
#ifndef TRT_LAYERNORM_KERNEL_H
#define TRT_LAYERNORM_KERNEL_H

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <cuda.h>
#include "cuda_fp16.h"

#include "common/checkMacrosPlugin.h"

using half = __half;

template <typename T>
int32_t computeLayerNorm(
    int32_t const gridSize,
    int32_t const nHiddenDimension,
    T const*      input,
    T const*      gamma,
    T const*      beta,
    T*            output,
    float const   epsilon,
    cudaStream_t  stream);

int32_t computeLayerNormQDQ(
    int32_t const gridSize,
    int32_t const nHiddenDimension,
    int8_t const* input,
    __half const* gamma,
    __half const* beta,
    int8_t*       output,
    float const   dqScaleIn,
    float const   qScale,
    float const   epsilon,
    cudaStream_t  stream);

#endif // TRT_LAYERNORM_KERNEL_H
