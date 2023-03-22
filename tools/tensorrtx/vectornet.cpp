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

const int INPUT_CHANNEL = 6;
const int OUTPUT_CHANNEL = 64;
const int HIDDEN_CHANNEL = 64;
const int FEATS_NUM = 10;

// print tensor dimensions
void printDims(ITensor* data)
{
    Dims dims = data->getDimensions();
    int  nbDims = dims.nbDims;
    for (int d = 0; d < nbDims; d++)
        std::cout << dims.d[d] << " "; // << dims.d[1] << " " << dims.d[2] << " " << dims.d[3] << std::endl;
    std::string sss;
    if (data->getType() == DataType::kHALF) sss = "float16";
    if (data->getType() == DataType::kFLOAT) sss = "float32";
    std::cout << sss << " ";
    std::cout << std::endl;
}

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
        std::cout << "\n";
        wt.values = val;
        wt.count = size;

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
    std::map<std::string, Weights> weightMap =
        loadWeights("/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/sub_graph/sub_graph.wts");

    // INetworkDefinition *network = builder->createNetworkV2(0U);
    INetworkDefinition* network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    ITensor* feats = network->addInput("feats", DataType::kFLOAT, Dims4{-1, INPUT_CHANNEL, 1, 1});
    ITensor* cluster = network->addInput("cluster", DataType::kFLOAT, Dims4{-1, 1, 1, 1});
    assert(feats);
    assert(cluster);

    // 1. MLP + Scatter + Concat
    auto mlp1 = MLP_block(network, weightMap, *feats, HIDDEN_CHANNEL, HIDDEN_CHANNEL, true, "layer_seq.glp_0.");
    IPluginV2Layer* scatter1 = ScatterMax_Plugin(network, mlp1->getOutput(0), cluster, HIDDEN_CHANNEL, "scatter_max1");
    ITensor*        inputTensorsCat1[] = {mlp1->getOutput(0), scatter1->getOutput(0)};
    IConcatenationLayer* concat1 = network->addConcatenation(inputTensorsCat1, 2);
    concat1->setAxis(1);
    assert(mlp1);
    assert(scatter1);
    assert(concat1);

    // 2. MLP + Scatter + Concat
    auto mlp2 =
        MLP_block(network, weightMap, *concat1->getOutput(0), HIDDEN_CHANNEL, HIDDEN_CHANNEL, true, "layer_seq.glp_1.");
    IPluginV2Layer* scatter2 = ScatterMax_Plugin(network, mlp2->getOutput(0), cluster, HIDDEN_CHANNEL, "scatter_max2");
    ITensor*        inputTensorsCat2[] = {mlp2->getOutput(0), scatter2->getOutput(0)};
    IConcatenationLayer* concat2 = network->addConcatenation(inputTensorsCat2, 2);
    concat2->setAxis(1);
    assert(mlp2);
    assert(scatter2);
    assert(concat2);

    // 3. MLP + Scatter + Concat
    auto mlp3 =
        MLP_block(network, weightMap, *concat2->getOutput(0), HIDDEN_CHANNEL, HIDDEN_CHANNEL, true, "layer_seq.glp_2.");
    IPluginV2Layer* scatter3 = ScatterMax_Plugin(network, mlp3->getOutput(0), cluster, HIDDEN_CHANNEL, "scatter_max3");
    ITensor*        inputTensorsCat3[] = {mlp3->getOutput(0), scatter3->getOutput(0)};
    IConcatenationLayer* concat3 = network->addConcatenation(inputTensorsCat3, 2);
    concat3->setAxis(1);
    assert(mlp3);
    assert(scatter3);
    assert(concat3);

    // 4. Linear
    IFullyConnectedLayer* linear = network->addFullyConnected(
        *concat3->getOutput(0), OUTPUT_CHANNEL, weightMap["linear.weight"], weightMap["linear.bias"]);
    assert(linear);

    // set output
    linear->getOutput(0)->setName("out");
    network->markOutput(*linear->getOutput(0));

    // Set configurations
    builder->setMaxBatchSize(1);
    // Set workspace size
    config->setMaxWorkspaceSize(1 << 24);

    std::cout << "[INFO]: Define done." << std::endl;

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("feats", OptProfileSelector::kMIN, Dims4(1, INPUT_CHANNEL, 1, 1));
    profile->setDimensions("feats", OptProfileSelector::kOPT, Dims4(2, INPUT_CHANNEL, 1, 1));
    profile->setDimensions("feats", OptProfileSelector::kMAX, Dims4(512, INPUT_CHANNEL, 1, 1));

    profile->setDimensions("cluster", OptProfileSelector::kMIN, Dims4(1, 1, 1, 1));
    profile->setDimensions("cluster", OptProfileSelector::kOPT, Dims4(2, 1, 1, 1));
    profile->setDimensions("cluster", OptProfileSelector::kMAX, Dims4(512, 1, 1, 1));
    config->addOptimizationProfile(profile);

    // Build CUDA Engine using network and configurations
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);

    std::cout << "[INFO]: Engine done." << std::endl;

    // Don't need the network any more
    // free captured memory
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) { free((void*)(mem.second.values)); }

    std::cout << "[INFO]: CreateMLPEngine success." << std::endl;
    return engine;
}

void doInference(
    IExecutionContext& context, float* feature, float* cluster, float* output, int batchSize, int feats_num)
{
    // Get engine from the context
    const ICudaEngine& engine = context.getEngine();

    // Pointers to feature and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex1 = engine.getBindingIndex("feats");
    const int inputIndex2 = engine.getBindingIndex("cluster");
    const int outputIndex = engine.getBindingIndex("out");
    context.setBindingDimensions(inputIndex1, Dims4(feats_num, INPUT_CHANNEL, 1, 1));
    context.setBindingDimensions(inputIndex2, Dims4(feats_num, 1, 1, 1));

    // Create GPU buffers on device -- allocate memory for input and output
    cudaMalloc(&buffers[inputIndex1], feats_num * INPUT_CHANNEL * sizeof(float));
    cudaMalloc(&buffers[inputIndex2], feats_num * 1 * sizeof(float));
    cudaMalloc(&buffers[outputIndex], feats_num * OUTPUT_CHANNEL * sizeof(float));

    // create CUDA stream for simultaneous CUDA operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // copy input from host (CPU) to device (GPU)  in stream
    cudaMemcpyAsync(
        buffers[inputIndex1], feature, feats_num * INPUT_CHANNEL * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[inputIndex2], cluster, feats_num * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);

    // execute inference using context provided by engine
    context.enqueue(batchSize, buffers, stream, nullptr);

    // copy output back from device (GPU) to host (CPU)
    cudaMemcpyAsync(
        output, buffers[outputIndex], feats_num * OUTPUT_CHANNEL * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // synchronize the stream to prevent issues
    //      (block CUDA and wait for CUDA operations to be completed)
    cudaStreamSynchronize(stream);

    // Release stream and buffers (memory)
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex1]);
    cudaFree(buffers[inputIndex2]);
    cudaFree(buffers[outputIndex]);
}

void performInference()
{
    // stream to write model
    char*  trtModelStream{nullptr};
    size_t size{0};

    // read model from the engine file
    std::ifstream file("../vectornet.engine", std::ios::binary);
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

    // create a runtime (required for deserialization of model) with NVIDIA's logger
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // deserialize engine for using the char-stream
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);

    // create execution context -- required for inference executions
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    int batch_size = 1;

    // array for output
    float out[OUTPUT_CHANNEL * batch_size * FEATS_NUM];
    float feature[INPUT_CHANNEL * batch_size * FEATS_NUM] = {
        // 0.0, 1, 2, 3, 4, 5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 121.0, 132.0, 103.0, 114.0, 105.0, 135.0};
        1.2855,  0.9948,  0.2046,  0.0493,  -0.1711, 0.2865,  0.7435, -1.0972, 0.6326,  -0.5128, -0.9912, 0.3193,
        0.3489,  -0.9907, 1.0864,  -0.0765, -1.4389, -1.3965, 0.5083, -1.5095, -0.3404, 0.0123,  0.6280,  -0.6943,
        -0.8992, 0.2768,  0.5403,  1.4955,  0.2885,  -1.9019, 0.4183, 0.0400,  2.1108,  1.0964,  0.2342,  0.8872,
        -0.7857, -0.9230, 0.6942,  -2.0588, -0.8065, 0.2856,  1.1959, 0.1580,  -1.1589, 1.0871,  1.1840,  0.1470,
        -0.0257, 0.0598,  -1.2005, -1.1679, 0.8444,  -0.3796, 0.2261, 0.0967,  -0.4144, 0.4793,  -0.7380, 0.8590};
    float cluster[batch_size * FEATS_NUM] = {0, 1, 1, 2, 2, 3, 3, 3, 3, 4};
    // float feature[INPUT_CHANNEL * batch_size * FEATS_NUM] = {0.0, 1, 2, 3, 4,
    // 5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}; float feature[INPUT_CHANNEL * batch_size * FEATS_NUM]; for
    // (int i = 0; i < INPUT_CHANNEL * batch_size * FEATS_NUM; i++)
    // {
    //     feature[i] = 0.0;
    // }

    // doInference(*context, feature, cluster, out, batch_size, FEATS_NUM);

    // time the execution
    auto start = std::chrono::system_clock::now();

    for (int k = 0; k < 1000; k++)
    {
        // do inference using the parameters
        doInference(*context, feature, cluster, out, batch_size, FEATS_NUM);
    }

    // time the execution
    auto end = std::chrono::system_clock::now();
    std::cout << "\n[INFO]: Time taken by execution: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // free the captured space
    context->destroy();
    engine->destroy();
    runtime->destroy();

    std::cout << "\nInput:\t";
    for (float i : feature) { std::cout << i << ", "; }
    std::cout << "\nOutput:\t";
    for (float j : out) { std::cout << j << ", "; }
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

    // free up the memory
    engine->destroy();
    builder->destroy();
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
    std::ofstream p("../vectornet.engine", std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

    // Release the memory
    modelStream->destroy();

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
    if (std::string(argv[1]) == "-s") { return 1; }
    else if (std::string(argv[1]) == "-d") { return 2; }
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
