cmake_minimum_required(VERSION 3.14)
project(vectornet_test)

set(CMAKE_CXX_STANDARD 14)

# find tensorrt
find_library(NVINFER NAMES nvinfer)
message("NVINFER: ${NVINFER}")

# find CUDA
find_package(CUDA)
message("CUDA Libs: ${CUDA_LIBRARIES}")
message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")

# find CUDNN
find_library(CUDNN_LIBRARY
    NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
    PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES lib lib64 bin
    DOC "CUDNN library."
)
message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")

# # head files include
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/Plugin
    ${PROJECT_SOURCE_DIR}/Plugin/layerNormPlugin
    ${PROJECT_SOURCE_DIR}/Plugin/scatterMaxPlugin
    ${CUDA_INCLUDE_DIRS}
)

# mytrt_plugin
file(GLOB_RECURSE PLUGIN_SOURCES ${PROJECT_SOURCE_DIR}/Plugin/*.cpp)
file(GLOB_RECURSE PLUGIN_CU_SOURCES ${PROJECT_SOURCE_DIR}/Plugin/*.cu)
cuda_add_library(mytrt_plugin SHARED ${PLUGIN_CU_SOURCES} ${PLUGIN_SOURCES})
target_link_libraries(mytrt_plugin nvinfer)

add_executable(test_vectornet test.cpp vectornet.cpp)

target_link_libraries(test_vectornet ${CUDA_LIBRARIES} nvinfer mytrt_plugin)
add_definitions(-O2 -pthread)
