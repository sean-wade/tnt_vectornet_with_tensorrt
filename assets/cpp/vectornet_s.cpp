#include <map>
#include <fstream>
#include <chrono>
#include <iostream>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "layer.h"

using namespace nvinfer1;
static Logger gLogger;

const std::string WTS_FILE    = "/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/vectornet.wts";
const std::string ENGINE_FILE = "../data/vectornet.engine";

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

ICudaEngine* createVectornetEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    std::cout << "[INFO]: Creating MLP using TensorRT..." << std::endl;
    // Load Weights from relevant file
    std::map<std::string, Weights> weightMap = loadWeights(WTS_FILE);

    // INetworkDefinition *network = builder->createNetworkV2(0U);
    INetworkDefinition* network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    ITensor* feats         = network->addInput("feats", DataType::kFLOAT, Dims4{-1, INPUT_CHANNEL, 1, 1});
    ITensor* cluster       = network->addInput("cluster", DataType::kFLOAT, Dims4{-1, 1, 1, 1});
    ITensor* cluster_count = network->addInput("cluster_count", DataType::kFLOAT, Dims4{-1, 1, 1, 1});
    ITensor* id_embedding  = network->addInput("id_embedding", DataType::kFLOAT, Dims4{-1, 2, 1, 1});
    assert(feats);
    assert(cluster);
    assert(cluster_count);
    assert(id_embedding);

    ILayer* sub_graph_out = SubGraph(network, weightMap, feats, cluster, cluster_count, HIDDEN_CHANNEL, HIDDEN_CHANNEL, true, "subgraph.");
    ILayer* global_graph_target =
        GlobalGraph(network, weightMap, sub_graph_out->getOutput(0), id_embedding, GLOBAL_GRAPH_CHANNEL, "global_graph.");

    ILayer* pred_mlp = MlpBlock(
        network, weightMap, *global_graph_target->getOutput(0), TRAJ_PRED_MLP_CHANNEL, TRAJ_PRED_MLP_CHANNEL, false, "traj_pred_mlp.0.");
    assert(pred_mlp);

    IFullyConnectedLayer* pred_fc = network->addFullyConnected(
        *pred_mlp->getOutput(0), FINAL_PRED_CHANNEL, weightMap["traj_pred_mlp.1.weight"], weightMap["traj_pred_mlp.1.bias"]);
    assert(pred_fc);

    pred_fc->getOutput(0)->setName("pred");
    network->markOutput(*pred_fc->getOutput(0));

    // Set configurations
    builder->setMaxBatchSize(1);
    // Set workspace size
    config->setMaxWorkspaceSize(1 << 24);

    std::cout << "[INFO]: Define done." << std::endl;

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

    // TRT 7.2.0 version.
    // ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // assert(engine != nullptr);

    IHostMemory* engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0)
    {
        std::cout << "[Error]: Failed building serialized engine!" << std::endl;
        return nullptr;
    }
    std::cout << "[INFO]: Succeeded building serialized engine!" << std::endl;

    IRuntime*    runtime{createInferRuntime(gLogger)};
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());

    // TRT 7.2.0 version.
    // network->destroy();

    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    std::cout << "[INFO]: Create VectorNet Engine success." << std::endl;
    return engine;
}

void doInference(
    IExecutionContext& context,
    float*             feature,
    float*             cluster,
    float*             cluster_count,
    float*             id_embedding,
    float*             output,
    int                batchSize,
    int                feats_num,
    int                cluster_num)
{
    const ICudaEngine& engine = context.getEngine();

    assert(engine.getNbBindings() == 5);
    void* buffers[5];

    const int inputIndex1 = engine.getBindingIndex("feats");
    const int inputIndex2 = engine.getBindingIndex("cluster");
    const int inputIndex3 = engine.getBindingIndex("cluster_count");
    const int inputIndex4 = engine.getBindingIndex("id_embedding");
    context.setBindingDimensions(inputIndex1, Dims4(feats_num, INPUT_CHANNEL, 1, 1));
    context.setBindingDimensions(inputIndex2, Dims4(feats_num, 1, 1, 1));
    context.setBindingDimensions(inputIndex3, Dims4(cluster_num, 1, 1, 1));
    context.setBindingDimensions(inputIndex4, Dims4(cluster_num, 2, 1, 1));

    cudaMalloc(&buffers[inputIndex1], feats_num * INPUT_CHANNEL * sizeof(float));
    cudaMalloc(&buffers[inputIndex2], feats_num * 1 * sizeof(float));
    cudaMalloc(&buffers[inputIndex3], cluster_num * 1 * sizeof(float));
    cudaMalloc(&buffers[inputIndex4], cluster_num * 2 * sizeof(float));

    const int outputIndex = engine.getBindingIndex("pred");
    cudaMalloc(&buffers[outputIndex], FINAL_PRED_CHANNEL * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(buffers[inputIndex1], feature, feats_num * INPUT_CHANNEL * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[inputIndex2], cluster, feats_num * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[inputIndex3], cluster_count, cluster_num * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[inputIndex4], id_embedding, cluster_num * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);

    context.enqueue(batchSize, buffers, stream, nullptr);

    cudaMemcpyAsync(output, buffers[outputIndex], FINAL_PRED_CHANNEL * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex1]);
    cudaFree(buffers[inputIndex2]);
    cudaFree(buffers[inputIndex3]);
    cudaFree(buffers[inputIndex4]);
    cudaFree(buffers[outputIndex]);
}

void performInference()
{
    // stream to write model
    char*  trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(ENGINE_FILE, std::ios::binary);
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

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);

    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    int batch_size = 1;

    float feature[INPUT_CHANNEL * FEATS_NUM] = {-0.3801, -0.1300, 1.1666,  -1.1327, 0.6438, 0.6729,  -1.1299, -2.2857, 0.1849,  0.0493,
                                                -0.4179, -0.5331, 0.7467,  -1.0006, 1.4848, 0.2771,  0.1393,  -0.9162, -1.7744, 0.8850,
                                                -1.6748, 1.3581,  -0.4987, -0.7244, 0.7941, -0.4109, -0.3446, -0.5246, -0.8153, -0.5685,
                                                1.9105,  -0.1069, 0.7214,  0.5255,  0.3654, -0.3434, 0.7163,  -0.6460, 1.9680,  0.8964,
                                                0.3845,  3.4347,  -2.6291, -0.9330, 0.6411, 0.9983,  0.6731,  0.9110,  -2.0634, -0.5751,
                                                1.4070,  0.5285,  -0.1171, -0.1863, 2.1200, 1.3745,  0.9763,  -0.1193, -0.3343, -1.5933};
    float cluster[FEATS_NUM]                 = {0, 1, 1, 2, 2, 3, 3, 3, 3, 4};
    float cluster_count[CLUSTER_NUM]         = {1, 2, 2, 4, 1};
    float id_embedding[CLUSTER_NUM * 2]      = {-0.3330, -0.7534, 1.1834, 0.6447, -1.1398, 0.5933, 1.5586, 1.0459, 0.2039, 1.0544};

    float out[FINAL_PRED_CHANNEL];
    auto  start = std::chrono::system_clock::now();

    for (int k = 0; k < 1000; k++)
    {
        doInference(*context, feature, cluster, cluster_count, id_embedding, out, batch_size, FEATS_NUM, CLUSTER_NUM);
    }

    auto end = std::chrono::system_clock::now();
    std::cout << "\n[INFO]: Time taken by execution: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;

    // // free the captured space
    // context->destroy();
    // engine->destroy();
    // runtime->destroy();

    std::cout << "\n>>>> Input:\n\n";
    for (float i : feature)
    {
        std::cout << i << ", ";
    }
    std::cout << "\n>>>> Output:\n\n";
    for (float j : out)
    {
        std::cout << j << ", ";
    }
    std::cout << std::endl;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    /**
     * Create engine using TensorRT APIs
     *
     * @param maxBatchSize: for the deployed model configs
     * @param modelStream: shared memory to store serialized model
     */

    // Create builder with the help of logger
    IBuilder* builder = createInferBuilder(gLogger);

    // Create hardware configs
    IBuilderConfig* config = builder->createBuilderConfig();

    // Build an engine
    ICudaEngine* engine = createVectornetEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine into binary stream
    (*modelStream) = engine->serialize();

    // // free up the memory
    // engine->destroy();
    // builder->destroy();
}

void performSerialization()
{
    /**
     * Serialization Function
     */
    // Shared memory object
    IHostMemory* modelStream{nullptr};

    // Write model into stream
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);

    std::cout << "[INFO]: Writing engine into binary..." << std::endl;

    // Open the file and write the contents there in binary format
    std::ofstream p(ENGINE_FILE, std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

    // // Release the memory
    // modelStream->destroy();

    std::cout << "[INFO]: Successfully created TensorRT engine..." << std::endl;
    std::cout << "\n\tRun inference using `./vectornet -d`" << std::endl;
}

int checkArgs(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "[ERROR]: Arguments not right!" << std::endl;
        std::cerr << "./vectornet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./vectornet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    if (std::string(argv[1]) == "-s")
    {
        return 1;
    }
    else if (std::string(argv[1]) == "-d")
    {
        return 2;
    }
    return -1;
}

int main(int argc, char** argv)
{
    int args = checkArgs(argc, argv);
    if (args == 1)
        performSerialization();
    else if (args == 2)
        performInference();
    return 0;
}
