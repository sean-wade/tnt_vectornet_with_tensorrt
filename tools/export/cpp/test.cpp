/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-13 14:51:53
 * @FilePath: /vectornet/tools/export/cpp/test.cpp
 * @LastEditors: zhanghao
 * @Description:
 *      目前可以 cpu libtorch 运行，速度大约 0.4 ms / forward.
 */

#include <memory>
#include <iostream>
#include <torch/script.h>
// #include "sglog/sglog.h"
// #include "sgtime/sgtime.h"

// using namespace jarvis;
int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try
    {
        // 使用以下命令从文件中反序列化脚本模块: torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    torch::Tensor x = torch::randn({10, 6});
    float cc[] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    float pp[] = {2};
    torch::Tensor cluster = torch::from_blob(cc, {10}).to(torch::kInt64);
    torch::Tensor id_embedding = torch::randn({2, 2});
    torch::Tensor poly_num = torch::from_blob(pp, {1});

    auto device_ = torch::kCPU;
    std::vector<torch::jit::IValue> torch_inputs;
    torch_inputs.push_back(std::move(x.to(device_)));
    torch_inputs.push_back(std::move(cluster.to(device_)));
    torch_inputs.push_back(std::move(id_embedding.to(device_)));
    torch_inputs.push_back(std::move(poly_num.to(device_)));
    at::Tensor torch_output_tensor_;

    // SG_INFO("Inference start...");
    std::cout << "Start\n";
    for (int i = 0; i < 10000; i++)
    {
        torch_output_tensor_ = module.forward(torch_inputs).toTensor().to(device_);
    }
    // SG_INFO("Inference done...");
    std::cout << torch_output_tensor_ << std::endl;
    std::cout << "OK\n";
}