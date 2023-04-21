/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-24 16:09:44
 * @FilePath: /vectornetx/vectornet.cpp
 * @LastEditors: zhanghao
 * @Description:
 */
#include <unistd.h>
#include "vectornet.h"

using namespace nvinfer1;

std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "[INFO]: Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open Weight file
    std::ifstream input(file);
    assert(input.is_open() && "[ERROR]: Unable to load weight file...");

    // Read number of weights
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    // Loop through number of line, actually the number of weights & biases
    while (count--)
    {
        // TensorRT weights
        Weights  wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        // Read name and type of weights
        std::string w_name;
        input >> w_name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            // Change hex values to uint32 (for higher values)
            input >> std::hex >> val[x];
        }
        std::cout << w_name << "\n";
        wt.values = val;
        wt.count  = size;

        // Add weight values against its name (key)
        weightMap[w_name] = wt;
    }
    std::cout << "[INFO]: Loading weights done." << std::endl;
    return weightMap;
}

bool VectorNet::Init(const VectorNetOptions& options)
{
    m_options_ = options;

    bool engine_exists = access(m_options_.engine_path.c_str(), F_OK) == 0;
    if (engine_exists)
    {
        printf("Engine file exist. Build from serialized engine : [%s]\n", m_options_.engine_path.c_str());
        if (!deserializeEngineFromFile())
        {
            printf("Engine build failed!\n");
            return false;
        }
    }
    else
    {
        printf("Engine file dose not exist! TRT start building! This will take a while...\n");
        printf("Weights path: %s\n", m_options_.weights_path.c_str());

        if (!createVectornetEngine())
        {
            printf("Create Vectornet Engine failed!\n");
            return false;
        }
        if (!serializeEngineToFile())
        {
            printf("Serialize Vectornet Engine failed!\n");
            return false;
        }
        delete m_engine_;
        if (!deserializeEngineFromFile())
        {
            printf("Engine rebuild failed!\n");
            return false;
        }
    }
    printf("Engine build success!\n");

    return true;
}

bool VectorNet::Process(TrajfeatureInputData& input_data, TrajPredictData& pred_data)
{
    void* buffers[5];
    float out[FINAL_PRED_CHANNEL];

    const int inputIndex1 = m_engine_->getBindingIndex("feats");
    const int inputIndex2 = m_engine_->getBindingIndex("cluster");
    const int inputIndex3 = m_engine_->getBindingIndex("cluster_count");
    const int inputIndex4 = m_engine_->getBindingIndex("id_embedding");

    m_context_->setBindingDimensions(inputIndex1, Dims4(input_data.feats_num, INPUT_CHANNEL, 1, 1));
    m_context_->setBindingDimensions(inputIndex2, Dims4(input_data.feats_num, 1, 1, 1));
    m_context_->setBindingDimensions(inputIndex3, Dims4(input_data.cluster_num, 1, 1, 1));
    m_context_->setBindingDimensions(inputIndex4, Dims4(input_data.cluster_num, 2, 1, 1));

    cudaMalloc(&buffers[inputIndex1], input_data.feats_num * INPUT_CHANNEL * sizeof(float));
    cudaMalloc(&buffers[inputIndex2], input_data.feats_num * 1 * sizeof(float));
    cudaMalloc(&buffers[inputIndex3], input_data.cluster_num * 1 * sizeof(float));
    cudaMalloc(&buffers[inputIndex4], input_data.cluster_num * 2 * sizeof(float));

    const int outputIndex = m_engine_->getBindingIndex("pred");
    cudaMalloc(&buffers[outputIndex], FINAL_PRED_CHANNEL * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(
        buffers[inputIndex1], input_data.feature, input_data.feats_num * INPUT_CHANNEL * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[inputIndex2], input_data.cluster, input_data.feats_num * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        buffers[inputIndex3], input_data.cluster_count, input_data.cluster_num * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        buffers[inputIndex4], input_data.id_embedding, input_data.cluster_num * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);

    m_context_->enqueue(BATCH_SIZE, buffers, stream, nullptr);

    float output[FINAL_PRED_CHANNEL];
    cudaMemcpyAsync(output, buffers[outputIndex], FINAL_PRED_CHANNEL * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex1]);
    cudaFree(buffers[inputIndex2]);
    cudaFree(buffers[inputIndex3]);
    cudaFree(buffers[inputIndex4]);
    cudaFree(buffers[outputIndex]);

    for (int i = 0; i < FINAL_PRED_CHANNEL; i++)
    {
        pred_data.predict_points.push_back(output[i]);
    }
    return true;
}

bool VectorNet::deserializeEngineFromFile()
{
    cudaSetDevice(m_options_.gpu_id);
    char*  trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(m_options_.engine_path, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else
    {
        return false;
    }
    IRuntime* runtime = createInferRuntime(gLogger);

    m_engine_  = runtime->deserializeCudaEngine(trtModelStream, size);
    m_context_ = m_engine_->createExecutionContext();

    return m_engine_ && m_context_;
}

bool VectorNet::createVectornetEngine()
{
    printf("Engine create start!\n");
    cudaSetDevice(m_options_.gpu_id);
    std::map<std::string, Weights> weightMap = loadWeights(m_options_.weights_path);

    IBuilder*       builder = createInferBuilder(gLogger);
    IBuilderConfig* config  = builder->createBuilderConfig();

    INetworkDefinition* network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    ITensor* feats         = network->addInput("feats", DataType::kFLOAT, Dims4{-1, INPUT_CHANNEL, 1, 1});
    ITensor* cluster       = network->addInput("cluster", DataType::kFLOAT, Dims4{-1, 1, 1, 1});
    ITensor* cluster_count = network->addInput("cluster_count", DataType::kFLOAT, Dims4{-1, 1, 1, 1});
    ITensor* id_embedding  = network->addInput("id_embedding", DataType::kFLOAT, Dims4{-1, 2, 1, 1});

    ILayer* sub_graph_out = SubGraph(network, weightMap, feats, cluster, cluster_count, HIDDEN_CHANNEL, HIDDEN_CHANNEL, true, "subgraph.");
    ILayer* global_graph_target =
        GlobalGraph(network, weightMap, sub_graph_out->getOutput(0), id_embedding, GLOBAL_GRAPH_CHANNEL, "global_graph.");

    ILayer* pred_mlp = MlpBlock(
        network, weightMap, *global_graph_target->getOutput(0), TRAJ_PRED_MLP_CHANNEL, TRAJ_PRED_MLP_CHANNEL, false, "traj_pred_mlp.0.");

    IFullyConnectedLayer* pred_fc = network->addFullyConnected(
        *pred_mlp->getOutput(0), FINAL_PRED_CHANNEL, weightMap["traj_pred_mlp.1.weight"], weightMap["traj_pred_mlp.1.bias"]);

    pred_fc->getOutput(0)->setName("pred");
    network->markOutput(*pred_fc->getOutput(0));

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1 << 24);
    if (m_options_.ues_fp16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        printf("Using FP16 mode, this may take several minutes... Please wait......\n");
    }
    else
    {
        printf("Using FP32 mode.\n");
    }

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("feats", OptProfileSelector::kMIN, Dims4(1, INPUT_CHANNEL, 1, 1));
    profile->setDimensions("feats", OptProfileSelector::kOPT, Dims4(128, INPUT_CHANNEL, 1, 1));
    profile->setDimensions("feats", OptProfileSelector::kMAX, Dims4(1024, INPUT_CHANNEL, 1, 1));

    profile->setDimensions("cluster", OptProfileSelector::kMIN, Dims4(1, 1, 1, 1));
    profile->setDimensions("cluster", OptProfileSelector::kOPT, Dims4(128, 1, 1, 1));
    profile->setDimensions("cluster", OptProfileSelector::kMAX, Dims4(1024, 1, 1, 1));

    profile->setDimensions("cluster_count", OptProfileSelector::kMIN, Dims4(1, 1, 1, 1));
    profile->setDimensions("cluster_count", OptProfileSelector::kOPT, Dims4(32, 1, 1, 1));
    profile->setDimensions("cluster_count", OptProfileSelector::kMAX, Dims4(256, 1, 1, 1));

    profile->setDimensions("id_embedding", OptProfileSelector::kMIN, Dims4(1, 2, 1, 1));
    profile->setDimensions("id_embedding", OptProfileSelector::kOPT, Dims4(32, 2, 1, 1));
    profile->setDimensions("id_embedding", OptProfileSelector::kMAX, Dims4(256, 2, 1, 1));
    config->addOptimizationProfile(profile);

    IHostMemory* engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0)
    {
        printf("[Error]: Failed building serialized engine!\n");
        return false;
    }

    IRuntime* runtime{createInferRuntime(gLogger)};
    m_engine_ = runtime->deserializeCudaEngine(engineString->data(), engineString->size());

    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    printf("Succeeded build serialized engine!\n");
    return true;
}

bool VectorNet::serializeEngineToFile()
{
    IHostMemory* modelStream = m_engine_->serialize();

    std::ofstream p(m_options_.engine_path, std::ios::binary);
    if (!p)
    {
        printf("Could not open plan output file\n");
        return false;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

    printf("Successfully created TensorRT engine...\n");
    return true;
}
