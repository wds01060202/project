#pragma once
#include "Configure.h"
using namespace cv;
using namespace std;
using namespace cv::gpu;
//thread
std::mutex mtx;           // locks access to counter
bool F_Img_start=false;
bool F_Feature_start=false;
bool F_Img__update=false;
bool F_Feature_update=false;
bool F_HomoG_start=false;
bool F_HomoG_update=false;
bool F_Segnet_start=false;
bool F_Segnet_update=false;
//ros
char msgss[50];
//project
double num; 
char Waitkey_t;

long bbc;
int Feat_fail_count;

Mat img_left, img_right;
Mat out;
Mat H_per,H;
//caffe
Mat showout;
Mat imShow;

Mat Th1_img_left, Th1_img_right;
Mat Th1_img_left_gray, Th1_img_right_gray;

Mat Th2_H_per=Mat(3,3,CV_64FC1);
Mat Th2_H=Mat(3,3,CV_64FC1);


cv::Mat Th3_xformed_proc,Th3_stitchedImage;//结果图
Mat Th3_img_left, Th3_img_right;
Mat Th3_H_per,Th3_H;
int kalman_count;

std::vector<cv::KeyPoint>keypoints1;
std::vector<cv::KeyPoint>keypoints2;
std::vector<cv::DMatch>matches,cmatches;

std::vector<cv::KeyPoint>Th1_keypoints1;
std::vector<cv::KeyPoint>Th1_keypoints2;

//GPU
 gpu::GpuMat img_left_gpu;
 gpu::GpuMat img_right_gpu;
 
 
 gpu::GpuMat Th1_img_left_gpu;
 gpu::GpuMat Th1_img_right_gpu;
 
gpu::GpuMat Th1_keypoints_gpu_1, Th1_keypoints_gpu_2;
 gpu::GpuMat Th1_descriptors_gpu_1, Th1_descriptors_gpu_2;
 gpu::SURF_GPU Th1_FeatureFinder_gpu(200);
 //caffe
 Classifier_SEG seger_left;
 Classifier_SEG seger_right;
 
string colorfile = "color.png";
cv::Mat color = cv::imread(colorfile, 1);
cv::Mat segnet_left_result;
cv::Mat segnet_right_result;
	
 //