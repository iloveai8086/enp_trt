# 最低版本要求
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc")
# 项目信息
project(trt_demo LANGUAGES CXX CUDA)

# 添加CMAKE_MODULE_PATH，否则找不到FindTensorRT.cmake
# list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# 寻找TensorRT库
# find_package(TensorRT REQUIRED)

#if (TensorRT_FOUND)
#    message(STATUS "Found TensorRT ${TensorRT_VERSION} in ${TensorRT_ROOT_DIR}")
#    message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
#    message(STATUS "TensorRT include files: ${TensorRT_INCLUDE_DIRS}")
#else()
#    message(FATAL_ERROR "Cannot find TensorRT")
#
#endif()

include_directories(
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/include
        /usr/local/cuda-11.1/include)
link_directories(
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/lib
        /usr/local/cuda-11.1/lib64
)

# 添加可执行文件
add_executable(build src/build.cpp)

# 头文件
# target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS})
# 链接库
target_link_libraries(build nvinfer cudart)


# 添加可执行文件
add_executable(runtime src/runtime.cu)

# 头文件
# target_include_directories(runtime PRIVATE ${TensorRT_INCLUDE_DIRS})
# 链接库
target_link_libraries(runtime nvinfer cudart)

