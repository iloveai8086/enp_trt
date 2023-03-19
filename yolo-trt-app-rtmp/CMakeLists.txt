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
# target_include_directories(yolo_plugin PUBLIC ${TensorRT_INCLUDE_DIRS})
# target_link_libraries(yolo_plugin PUBLIC ${TensorRT_LIBRARIES})
# set_target_properties(yolo_plugin PROPERTIES CUDA_ARCHITECTURES "61;70;75")
target_include_directories(yolo_plugin PUBLIC)
target_link_libraries(yolo_plugin cudart nvinfer nvonnxparser)

cuda_add_library(yolo_utils SHARED
        utils/postprocess.cpp
        utils/preprocess.cu
        )
target_link_libraries(yolo_utils opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)
# set_target_properties(yolo_utils PROPERTIES CUDA_ARCHITECTURES "61;70;75")


cuda_add_executable(build
        build.cu
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/samples/common/logger.cpp
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/samples/common/sampleUtils.cpp
        plugins/yoloPlugins.h
        )
# target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(build yolo_plugin) # -Wl,--no-as-needed is needed to avoid linking errors
target_link_libraries(build yolo_utils)
# set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")

# add subdirectory to build streamer
add_subdirectory(streamer)
# add library to build task
add_library(task SHARED
    task/border_cross.cpp
    task/gather.cpp
)
cuda_add_executable(runtime
    runtime.cu
    /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/samples/common/logger.cpp
        plugins/yoloPlugins.h
)
# target_include_directories(runtime PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(runtime yolo_plugin)
target_link_libraries(runtime yolo_utils)
target_link_libraries(runtime streamer task)
# set_target_properties(runtime PROPERTIES CUDA_ARCHITECTURES "61;70;75")

cuda_add_executable(runtime_thread
    runtime_thread.cu
    /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/samples/common/logger.cpp
plugins/yoloPlugins.h
)
# target_include_directories(runtime_thread PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(runtime_thread  yolo_plugin)
target_link_libraries(runtime_thread  yolo_utils)
target_link_libraries(runtime_thread  streamer task)
# set_target_properties(runtime_thread PROPERTIES CUDA_ARCHITECTURES "61;70;75")
