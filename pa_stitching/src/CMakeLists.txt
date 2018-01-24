# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io )
set(CMAKE_CXX_STANDARD 11)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04
# 增加opencv的依赖
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )


find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  sensor_msgs
roscpp_serialization 
cv_bridge  image_transport
)

#find_package(Caffe)

set(Caffe_INCLUDE_DIRS "/home/wds/catkin_ws/src/SEG-SURF/caffe/build/include/caffe/proto/")
#message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(include ${catkin_INCLUDE_DIRS})


# INCLUDE_DIRECTORIES( ${PROJECT_SOURSE_DIR}/include )

ADD_EXECUTABLE( SEG_SURF main.cpp  segnet.cpp)
TARGET_LINK_LIBRARIES( SEG_SURF
${catkin_LIBRARIES} 
${var}
${OpenCV_LIBS} 
${Caffe_LIBRARIES}
/home/wds/catkin_ws/src/SEG-SURF/caffe/build/lib/libcaffe.so 
/usr/lib/x86_64-linux-gnu/libglog.so.0
 )
