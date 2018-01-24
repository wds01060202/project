#pragma once
#include "Configure.h"
using namespace cv;
using namespace std;
static void VID_deal(cv::Mat& img1, cv::Mat& img2, cv::Mat & out);
class ImageDeal
{
public:
	cv::Ptr<cv::FeatureDetector>detector;
	cv::Ptr<cv::DescriptorExtractor>extractor;
	cv::Point leftTop, leftBottom, rightTop, rightBottom;
    cv::VideoWriter writer;
	float ratio; //检查最近邻与次近邻的距离比率, 比值越小说明最近邻代表性越好
	double distance;
	double confidence;
	cv::Mat img1, img2,img3;
	cv::Mat img1_roi, img2_roi,img3_roi,img2_roi_mr,img2_roi_lm;
	std::string windowNameInput1;
	std::string windowNameInput2;
	std::string windowNameOutput;
	cv::Mat stitchedImage, xformed, img1_ROI, xformed_ROI;
	void(*process)(cv::Mat&, cv::Mat&, cv::Mat &);//建立回调函数
	bool callIt;
	char image_name1[30];//VideoToImage 函数用的文件名
	char image_name2[30];

	vector<cv::Point2f> inlier_points1, inlier_points2, inlier_points2_Out;
	std::vector<cv::KeyPoint>keypoints1,keypoints2;
        std::vector<cv::DMatch>matches;
	//Mat D_gray;
	bool refineH,flag_rh;
	bool  Down_flag;//控制帧标志位
	cv::Mat H, H_l,H_r,H_per; //单应矩阵
	cv::Mat H21, H23, H34, H54, H56,H61;
	cv::Mat out1;
	bool flag_c=false;
	Scalar quality_Num;
	double Quality_Num;
	double Expect_quality=0.8;

	//Mat shftMat = Mat(3, 3, CV_64FC1);;
	cv::Mat xformed_proc;//结果图
	ImageDeal();
	virtual ~ImageDeal();
	void VideoToImage(int i, Mat &mat1,Mat &mat2);
	int ratioTest(std::vector<std::vector<cv::DMatch>>&matches);
	void symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches1, const std::vector<std::vector<cv::DMatch>>& matches2, std::vector<cv::DMatch>& symMatches);	//移除非对称匹配
	cv::Mat ransacTest(const std::vector<cv::DMatch>&matches, const std::vector<cv::KeyPoint>&keypoints1, const std::vector<cv::KeyPoint>&keypoints2,int a);//
	void CalcFourCorner(cv::Mat H1,cv::Mat img22);//计算图2的四个角经矩阵H变换后的坐标
	void setFrameDeal(void(*frameDealCallback)(cv::Mat &, cv::Mat &, cv::Mat &));//初始化
	void SetInput(std::string filename1, std::string filename2);//读AVI
	void setOutput(Size videoSize, double framerate , bool isColor);//写视频格式设置
	void displayInput(std::string wn1, std::string wn2);//建立输入窗口
	void displayOutput(std::string wn);//建立输出窗口
	void dontDisplay();//销毁窗口
	bool run(cv::Mat &out, cv::Mat& image2, cv::Mat& image1);//视频输入、处理及输出部分
	bool run1(cv::Mat &out, cv::Mat& image_l, cv::Mat& image_m, cv::Mat &image_r, int image_num);//右侧为参考图
	bool run2(cv::Mat &out, cv::Mat& image_r, cv::Mat& image_m, cv::Mat &image_l, int image_num);//左侧为参考图
	void writeNextFrame(cv::Mat& frame);//写视频帧
	int GetVideoRate();//获得帧率
	bool Feature_Detecor(cv::Mat &out, cv::Mat& image_left, cv::Mat& image_right);//视频输入、处理及输出部分
	
};

