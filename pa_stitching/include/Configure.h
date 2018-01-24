#ifndef _CONFIGURE_
#define _CONFIGURE_
//std
#include <memory>
#include <iostream>
#include <cctype>
#include <time.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <math.h>
//muti-thread
#include <thread>
#include <mutex> 
#include <chrono>
#include <X11/Xlib.h>
using namespace std;
//opencv
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>

#include"opencv2/gpu/gpu.hpp"
#include<opencv2/nonfree/gpu.hpp>
//ros
#include "ros/ros.h"  
#include "std_msgs/String.h"  
//caffe
#include "segnet.h"
//define

#define PI 3.1415926

#define Th3_max(a,b)            a > b ? a : b
#define Th3_min(a,b)            a < b ? a : b

#define RecordIMG 0
#define ShowMatcher 0
#define GPU_Modle 1

//global value;

#endif