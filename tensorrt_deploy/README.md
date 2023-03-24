<!--
 * @Author: zhanghao
 * @LastEditTime: 2023-03-24 17:41:10
 * @FilePath: /vectornetx/README.md
 * @LastEditors: zhanghao
 * @Description: 
-->
# Vectornet TRT

VectorNet tensorrt impletemention.


## Requirements
// 1. TensorRT-8.0.1-1
// 2. CUDA 11.1


## Usage

```
* 1. get vectornet.wts from python code.
    
    python tensorrt_deploy/vectornet_export_wts.py

* 2. modify config options in test.cpp

* 3. compile && run test

    mkdir build

    cd build

    cmake ..

    make

    ./test_vectornet
```

// 4. get the output values and the time using.


## Note
This code cannot run in tensorrt-7.2.3, IShuffle layer will return nan-value.(Don't know why)
