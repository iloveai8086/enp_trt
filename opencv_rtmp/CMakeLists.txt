cmake_minimum_required(VERSION 3.0)

project(opencv_rtmp LANGUAGES CXX)

# set CXX standard 14
set(CMAKE_CXX_STANDARD 14)

# find opencv
find_package(OpenCV REQUIRED)

if (OpenCV_FOUND)
    message(STATUS "OpenCV found")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()


# add subdirectory to build
add_subdirectory(streamer)

# add executable
add_executable(${PROJECT_NAME} main.cpp)

# link libraries
target_link_libraries(${PROJECT_NAME} streamer ${OpenCV_LIBS})