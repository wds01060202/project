cmake_minimum_required(VERSION 2.8.3)
project(pa_stitching)

set( CMAKE_BUILD_TYPE Debug )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
)
#multi-thread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(Boost REQUIRED)
find_package(OpenCV 2.4.13 REQUIRED)

catkin_package(
)


set(Caffe_INCLUDE_DIRS "/home/wds/catkin_ws/src/pa_stitching/caffe/build/include/caffe/proto/")
#message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(include ${catkin_INCLUDE_DIRS})

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/caffe/build/include/  )
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/caffe/include/)
include_directories(
  ${catkin_INCLUDE_DIRS}
  #${Caffe_INCLUDE_DIRS}
#  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}

# ${EIGEN3_INCLUDE_DIR}
)
INCLUDE_DIRECTORIES( /usr/local/cuda/include  )
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/caffe/build/lib)
LINK_DIRECTORIES( /usr/local/cuda-8.0/lib64)

add_executable(pa_stitching src/Muti_cpu.cpp src/ImageDeal.cpp src/segnet.cpp)
target_link_libraries(pa_stitching
	${catkin_LIBRARIES} 
	${OpenCV_LIBS} 
	
	${Caffe_LIBRARIES}
	/home/wds/catkin_ws/src/pa_stitching/caffe/build/lib/libcaffe.so 
	/usr/lib/x86_64-linux-gnu/libglog.so.0
	${Boost_LIBRARIES}
	Threads::Threads
)

add_executable(matcher_test src/test_macher.cpp src/ImageDeal.cpp src/segnet.cpp)
target_link_libraries(matcher_test 
	${catkin_LIBRARIES} 
	${OpenCV_LIBS} 
	
	${Caffe_LIBRARIES}
	/home/wds/catkin_ws/src/pa_stitching/caffe/build/lib/libcaffe.so 
	/usr/lib/x86_64-linux-gnu/libglog.so.0
	${Boost_LIBRARIES}
	Threads::Threads
)
