<!--
 * @Author: zhanghao
 * @Date: 2023-03-08 14:32:01
 * @LastEditors: zhanghao
 * @LastEditTime: 2023-03-10 21:52:05
 * @FilePath: /vectornet/README.md
 * @Description: 
-->
# Reimplement VectorNet on Custom dataset

Paper: [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)

Migrate from the [TNT implemention](https://github.com/Henry1iu/TNT-Trajectory-Prediction)

## Features
Contain follow features:
- [x] my custom data feature preprocessor
- [x] remove torch-geometric requirements
- [x] batchify the data and compute subgraph in pipeline
- [x] better visualize the evaluation result
- [x] overfit the tiny sample dataset
- [x] simplify the inference pipeline for deploy
- [ ] deploy through TensorRT or other framework

## Viz on my custom dataset.
* Warning: My custom dataset does not have HDMap, so I only use the perception lane for input. This is only for experiment !!!

![](docs/viz.png) 


## Deploy

First, organize the inference computing pipeline as follows:
![](docs/VectorNet计算图.png) 
