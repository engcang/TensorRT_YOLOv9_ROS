cmake_minimum_required(VERSION 3.12)
project(tensorrt_yolov9_ros)
### Set compiler
ADD_COMPILE_OPTIONS(-std=c++17 )
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -g -ggdb -O3") # -std=c++17 is commented to avoid conflict with CUDA
set(CMAKE_CXX_FLAGS_RELEASE "-Wall ${CMAKE_CXX_FLAGS}")

### Get packages
# catkin
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    message_generation
    message_runtime
)
# Find CUDA
find_package(CUDA REQUIRED)
# Find OpenCV
find_package(OpenCV REQUIRED)
# Include TensorRT
set(TENSORRT_INCLUD_DIR /usr/include/x86_64-linux-gnu) # TensorRT include directory, NvInfer.h etc.
set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
link_directories(${TENSORRT_LIB_DIR})

include_directories( # Include directories
    include
    ${catkin_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUD_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

### Generate messages
add_message_files(
  FILES
    bbox.msg
    bboxes.msg
)
generate_messages(
  DEPENDENCIES
    std_msgs
)

### Catkin package
catkin_package()

### Build
cuda_add_executable(${PROJECT_NAME}_node src/main.cpp include/preprocess.cu)
target_link_libraries(${PROJECT_NAME}_node # Link libraries
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)
