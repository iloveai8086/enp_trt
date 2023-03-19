cmake_minimum_required(VERSION 3.11)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc")
project(TensorRT_YoloV5)
find_package(CUDA REQUIRED)
# set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/include
        /usr/local/cuda-11.1/include
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/samples/common
)
link_directories(
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/lib
        /usr/local/cuda-11.1/lib64
)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

cuda_add_library(yolo_plugin SHARED
        plugins/yoloPlugins.cpp
        plugins/yoloForward_nc.cu
        plugins/yoloPlugins.h
        )
target_include_directories(yolo_plugin PUBLIC)
target_link_libraries(yolo_plugin cudart nvinfer nvonnxparser)
# set_target_properties(yolo_plugin PROPERTIES CUDA_ARCHITECTURES "61;70;75;80")

cuda_add_library(yolo_utils SHARED
        utils/postprocess.cpp
        utils/preprocess.cu
        )
target_link_libraries(yolo_utils opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)
# set_target_properties(yolo_utils PROPERTIES CUDA_ARCHITECTURES "61;70;75;80")


cuda_add_executable(build
        build.cu
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/samples/common/logger.cpp
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/samples/common/sampleUtils.cpp
        plugins/yoloPlugins.h
        )
# target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(build yolo_plugin) # -Wl,--no-as-needed is needed to avoid linking errors
# target_link_libraries(build yolo_utils)
# set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75;80")

cuda_add_executable(runtime
        runtime.cu
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/samples/common/logger.cpp
        /media/ros/A666B94D66B91F4D/ros/learning/enpCV/tensorrt_cpp/utils/preprocess.h
        /media/ros/A666B94D66B91F4D/ros/learning/enpCV/tensorrt_cpp/utils/postprocess.h
        /media/ros/A666B94D66B91F4D/ros/learning/enpCV/tensorrt_cpp/utils/types.h
        )
# target_include_directories(runtime PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(runtime yolo_plugin yolo_utils)
# set_target_properties(runtime PROPERTIES CUDA_ARCHITECTURES "61;70;75;80")
