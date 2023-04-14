<!--
 * @Author: zhanghao
 * @LastEditTime: 2023-04-14 14:01:29
 * @FilePath: /my_vectornet_github/assets/README.md
 * @LastEditors: zhanghao
 * @Description:
-->

# Reimplement VectorNet on Custom dataset

Paper: [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)

Thanks to this [TNT implemention](https://github.com/Henry1iu/TNT-Trajectory-Prediction)

## I. Features

Contain follow features:

- [x] my custom data feature preprocessor
- [x] remove torch-geometric requirements
- [x] batchify the data and compute subgraph in pipeline
- [x] better visualize the evaluation result
- [x] use tensorboard to viz the loss & metric
- [x] overfit the tiny sample dataset
- [x] simplify the inference pipeline for deploy
- [x] export onnx by replace scatter_max with fake_op
- [x] deploy through TensorRT or other framework

## II. Train & Test

### 1. Add python path

```
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. Train

```
python tools/train_vectornet.py -d ./mini_data -b 128 --lr 0.005
```

### 3. Test

```
python tools/test_vectornet.py -d ./mini_data -b 128 -rm weights/sg_best_v1.pth
```

## III. Plot

> **ATTENTION**: My custom dataset does not have HDMap, so I only use the perception lane for input. This is only for experiment !!!

![](docs/viz.png)

## IV.Tensorboard

![](docs/vectornet_metric.png)

## V. Metric

On my private dataset, the metrics are:

```
{"minADE": 0.3962065457291247, "minFDE": 0.9303791035665404, "MR": 0.11696658097686376}
```

## VI. Deploy

### 1. Compute graph

Organize the inference computing pipeline as follows:
![](docs/VectorNet计算图.png)

### 2. TRT deploy

Details can be found in tensorrt_deploy directory.

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

- Precision Compare:

```
 TensorRT output            Pytorch output
-0.000751,0.598084        [-0.0008,  0.5981],
0.002925,0.618462         [ 0.0029,  0.6184],
0.002822,0.607659         [ 0.0028,  0.6076],
0.003941,0.613367         [ 0.0039,  0.6133],
0.004445,0.619251         [ 0.0044,  0.6192],
0.005261,0.606338         [ 0.0053,  0.6063],
0.004013,0.606201         [ 0.0040,  0.6062],
0.001496,0.615865         [ 0.0015,  0.6158],
0.002494,0.603205         [ 0.0025,  0.6032],
0.001168,0.611155         [ 0.0012,  0.6111],
0.002630,0.602360         [ 0.0026,  0.6023],
0.002455,0.606869         [ 0.0025,  0.6068],
0.003524,0.609820         [ 0.0035,  0.6098],
0.003280,0.604363         [ 0.0033,  0.6043],
0.002711,0.606441         [ 0.0027,  0.6064],
0.003065,0.599403         [ 0.0031,  0.5994],
0.004386,0.600533         [ 0.0044,  0.6005],
0.004443,0.609845         [ 0.0044,  0.6098],
0.002879,0.600789         [ 0.0029,  0.6008],
0.005783,0.596618         [ 0.0058,  0.5966],
0.005589,0.601561         [ 0.0056,  0.6015],
0.005739,0.593500         [ 0.0057,  0.5935],
0.006239,0.596110         [ 0.0062,  0.5961],
0.005204,0.605715         [ 0.0052,  0.6057],
0.006330,0.603428         [ 0.0063,  0.6034],
0.004776,0.596250         [ 0.0048,  0.5962],
0.007388,0.584935         [ 0.0074,  0.5849],
0.006047,0.601648         [ 0.0060,  0.6016],
0.007334,0.602399         [ 0.0073,  0.6024],
0.008742,0.594365         [ 0.0087,  0.5943]

```

- Speed Test

![](docs/time.jpg)

### 3. Libtorch deploy

> **ABANDONED!!!**
> Only test on libtorch-cpu(version must > 1.12.0), average 0.4 ms/forward.

a. Export jit traced module.

```
python assets/tools/export/vectornet_export_v2.py
```

b. Compile cpp code.

```
cd assets/tools/export/cpp_libtorch
mkdir build && cd build
cmake ..
make
```

c. Excute

```
./vectornet ../../models/traced_vectornet.pt
```
