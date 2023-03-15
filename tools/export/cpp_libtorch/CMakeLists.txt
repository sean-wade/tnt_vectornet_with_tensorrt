cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(vectornet)


set(CMAKE_PREFIX_PATH "/home/zhanghao/envs/libtorch-shared-with-deps-latest/libtorch")
# set(CMAKE_PREFIX_PATH "/home/zhanghao/envs/libtorch-cxx11-abi-shared-with-deps-1.12.0+cu113/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(vectornet test.cpp)
target_link_libraries(vectornet ${TORCH_LIBRARIES})
set_property(TARGET vectornet PROPERTY CXX_STANDARD 14)

