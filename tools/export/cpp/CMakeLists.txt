cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(vectornet)

find_package(PkgConfig QUIET)
pkg_check_modules(
    SG_DEPEND
    sglog
    sgtime
)
message("depend inc: ${SG_DEPEND_INCLUDE_DIRS}")
message("depend lib: ${SG_DEPEND_LIBRARIES}")

# set(CMAKE_PREFIX_PATH "/home/zhanghao/envs/libtorch-shared-with-deps-latest/libtorch")
set(CMAKE_PREFIX_PATH "/home/zhanghao/envs/libtorch-cxx11-abi-shared-with-deps-1.12.0+cu113/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    ${SG_DEPEND_INCLUDE_DIRS}
)

add_executable(vectornet test.cpp)
target_link_libraries(vectornet ${TORCH_LIBRARIES} ${SG_DEPEND_LIBRARIES})
set_property(TARGET vectornet PROPERTY CXX_STANDARD 14)

