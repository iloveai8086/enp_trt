cmake_minimum_required(VERSION 3.11)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
project(ds_tracker VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# find_package(TensorRT REQUIRED)
include_directories(
        /home/ai/Downloads/TensorRT-8.5.2.2.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.2.2/include
        /usr/local/cuda-11.8/include
        /home/ai/Downloads/TensorRT-8.5.2.2.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.2.2/samples/common
)
link_directories(
        /home/ai/Downloads/TensorRT-8.5.2.2.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.2.2/lib
        /usr/local/cuda-11.8/lib64)
include_directories(/usr/local/cuda-11.8/include)   
     
# build yolo decode plugin
add_library(yolo_decode SHARED
    ./plugins/yoloPlugins.cpp
    ./plugins/yoloForward_nc.cu
    ./plugins/nvdsparsebbox_Yolo.cpp
)
# target_include_directories(yolo_decode PUBLIC ${TensorRT_INCLUDE_DIRS})
# target_link_libraries(yolo_decode PUBLIC ${TensorRT_LIBRARIES})
target_link_libraries(yolo_decode cudart nvinfer nvonnxparser)

set_target_properties(yolo_decode PROPERTIES CUDA_ARCHITECTURES "61;70;75")
target_include_directories(yolo_decode PRIVATE /usr/local/cuda-11.8/include)
target_include_directories(yolo_decode PRIVATE  /opt/nvidia/deepstream/deepstream/sources/includes)


#find gstream-1.0 gstreamer-rtsp-server-1.0, add include dirs and link libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-rtsp-server-1.0)

# NVDS_VERSION:=6.1
# 
# LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
# APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/
# APP_INSTALL_ROOT?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)
# set variables
set(LIB_INSTALL_DIR /opt/nvidia/deepstream/deepstream/lib/)
set(APP_INSTALL_DIR /opt/nvidia/deepstream/deepstream/bin/)
set(APP_INSTALL_ROOT /opt/nvidia/deepstream/deepstream)

link_directories(
    ${LIB_INSTALL_DIR}
    /usr/local/cuda/lib64
)
# add library to build task
add_library(task SHARED
    task/border_cross.cpp
    task/gather.cpp
)


add_executable(ds_detect ds_detect.cpp)
target_include_directories(ds_detect PRIVATE
    /usr/local/cuda-11.8/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${APP_INSTALL_ROOT}/sources/includes
)
target_link_libraries(ds_detect PRIVATE
    ${GSTREAMER_LIBRARIES} cudart nvdsgst_meta nvds_meta nvds_yml_parser task
)


add_executable(ds_track ds_track.cpp)
target_include_directories(ds_track PRIVATE
    /usr/local/cuda-11.8/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${APP_INSTALL_ROOT}/sources/includes
)
target_link_libraries(ds_track PRIVATE
    ${GSTREAMER_LIBRARIES} cudart nvdsgst_meta nvds_meta nvds_yml_parser task
)

add_executable(ds_app ds_app.cpp)
target_include_directories(ds_app PRIVATE
    /usr/local/cuda-11.8/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${APP_INSTALL_ROOT}/sources/includes
)
target_link_libraries(ds_app PRIVATE
    ${GSTREAMER_LIBRARIES} cudart nvdsgst_meta nvds_meta nvds_yml_parser task
)

add_executable(ds_app_multi ds_app_multi.cpp)
target_include_directories(ds_app_multi PRIVATE
    /usr/local/cuda-11.8/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${APP_INSTALL_ROOT}/sources/includes
)
target_link_libraries(ds_app_multi PRIVATE
    ${GSTREAMER_LIBRARIES} cudart nvdsgst_meta nvds_meta nvds_yml_parser task
)
