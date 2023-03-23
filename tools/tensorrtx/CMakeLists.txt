cmake_minimum_required(VERSION 3.14)
project(vectornet)

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

## head files include 
include_directories(
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/Plugin 
    ${PROJECT_SOURCE_DIR}/Plugin/layerNormPlugin
    ${PROJECT_SOURCE_DIR}/Plugin/scatterMaxPlugin
    ${CUDA_INCLUDE_DIRS}
)

# layernorm_plugin
file(GLOB_RECURSE LN_PLUGIN_SOURCES ${PROJECT_SOURCE_DIR}/Plugin/layerNormPlugin/*.cpp ${PROJECT_SOURCE_DIR}/Plugin/common/*.cpp)
file(GLOB_RECURSE LN_PLUGIN_CU_SOURCES ${PROJECT_SOURCE_DIR}/Plugin/layerNormPlugin/*.cu)
cuda_add_library(layernorm_plugin SHARED ${LN_PLUGIN_CU_SOURCES} ${LN_PLUGIN_SOURCES})
target_link_libraries(layernorm_plugin nvinfer)

# scattermax_plugin
file(GLOB_RECURSE SC_PLUGIN_SOURCES ${PROJECT_SOURCE_DIR}/Plugin/scatterMaxPlugin/*.cpp ${PROJECT_SOURCE_DIR}/Plugin/common/*.cpp)
file(GLOB_RECURSE SC_PLUGIN_CU_SOURCES ${PROJECT_SOURCE_DIR}/Plugin/scatterMaxPlugin/*.cu)
cuda_add_library(scattermax_plugin SHARED ${SC_PLUGIN_CU_SOURCES} ${SC_PLUGIN_SOURCES})
target_link_libraries(scattermax_plugin nvinfer)

# scattermax_nopadding_plugin
file(GLOB_RECURSE SC_PLUGIN_SOURCES ${PROJECT_SOURCE_DIR}/Plugin/scatterMaxNoPaddingPlugin/*.cpp ${PROJECT_SOURCE_DIR}/Plugin/common/*.cpp)
file(GLOB_RECURSE SC_PLUGIN_CU_SOURCES ${PROJECT_SOURCE_DIR}/Plugin/scatterMaxNoPaddingPlugin/*.cu)
cuda_add_library(scattermax_nopadding_plugin SHARED ${SC_PLUGIN_CU_SOURCES} ${SC_PLUGIN_SOURCES})
target_link_libraries(scattermax_nopadding_plugin nvinfer)

# create link for executable files
add_executable(vectornet vectornet.cpp)

target_link_libraries(vectornet ${CUDA_LIBRARIES} nvinfer layernorm_plugin scattermax_plugin scattermax_nopadding_plugin)

add_definitions(-O2 -pthread)

