# Reimplement VectorNet on Custom dataset

Paper: [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)

Migrate from the [TNT implemention](https://github.com/Henry1iu/TNT-Trajectory-Prediction)

## Features
Contain follow features:
- [x] my custom data feature preprocessor
- [x] remove torch-geometric requirements
- [x] batchify the data and compute subgraph in pipeline
- [x] better visualize the evaluation result
- [x] use tensorboard to viz the loss & metric
- [x] overfit the tiny sample dataset
- [x] simplify the inference pipeline for deploy
- [ ] deploy through TensorRT or other framework

## Viz on my custom dataset.
* Warning: My custom dataset does not have HDMap, so I only use the perception lane for input. This is only for experiment !!!

![](docs/viz.png) 


## Tensorboard Viz
![](docs/vectornet_metric.png) 

## Deploy

First, organize the inference computing pipeline as follows:
![](docs/VectorNet计算图.png) 
