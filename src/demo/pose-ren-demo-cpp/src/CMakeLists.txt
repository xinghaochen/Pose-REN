set(EXE_NAME "PoseREN")
project(${EXE_NAME})
set(CAFFE_DIR "/home/workspace/Work-Linux/projects/my-release/caffe-pose")
set(CAFFE_LIBS "${CAFFE_DIR}/build/lib/libcaffe.so")
set(CMAKE_CXX_FLAGS "-O0 -fPIC -std=c++11")
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
find_package( OpenCV REQUIRED )
find_package(CUDA)
#find_package(Glog)

include_directories("${CAFFE_DIR}/include")
include_directories(${OpenCV_INCLUDE_DIRS})
MESSAGE( STATUS "OpenCV_HEADERS: " ${OpenCV_INCLUDE_DIRS} )
include_directories(${OpenCV_INCLUDE_DIRS})

set(BLAS_INCLUDE "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include")
include_directories(${BLAS_INCLUDE})

include_directories(${CUDA_INCLUDE_DIRS})

MESSAGE( STATUS "BLAS_INCLUDE: " ${BLAS_INCLUDE} )
MESSAGE( STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS} )
MESSAGE( STATUS "CUDA_CUBLAS_LIBRARIES: " ${CUDA_CUBLAS_LIBRARIES} )


# google protobuf
find_package( Protobuf REQUIRED )
include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})
MESSAGE( STATUS "PROTOBUF_LIBRARIES: " ${PROTOBUF_LIBRARIES} )

# glog
#set(GLOG_INCLUDE_DIRS ${glog_INSTALL}/include)
#set(GLOG_LIBRARIES ${GFLAGS_LIBRARIES} ${glog_INSTALL}/lib/libglog.a)
#set(GLOG_LIBRARY_DIRS ${glog_INSTALL}/lib)
#include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})

# ---[ Boost
find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
#list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

add_executable(${EXE_NAME} hand_pose_estimator.cpp run_hand_pose.cpp)
target_link_libraries(${EXE_NAME} ${CAFFE_LIBS} glog realsense2 ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES} ${Boost_LIBRARIES} python2.7)
