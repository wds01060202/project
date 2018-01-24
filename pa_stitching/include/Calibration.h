#pragma once
#include "Configure.h"

using namespace cv;
using namespace std;
using namespace cv::gpu;
class Calibration
{
private:

	vector<vector<Point3f>>objectPoints;//位于世界坐标的点
	vector<vector<Point2f>>imagePoints;//位于像素坐标的点
	int flag;//标定的方式
	Mat KKK1 = Mat(3, 3, CV_32FC1);//相机参数矩阵
	Mat KKK2 = Mat(3, 3, CV_32FC1);
	Mat KKK3 = Mat(3, 3, CV_32FC1);//相机参数矩阵
	Mat KKK4 = Mat(3, 3, CV_32FC1);
	Mat KKK5 = Mat(3, 3, CV_32FC1);//相机参数矩阵
	Mat KKK6 = Mat(3, 3, CV_32FC1);
	Mat KKC1 = Mat(1, 5, CV_32FC1);//相机畸变矩阵
	Mat KKC2 = Mat(1, 5, CV_32FC1);
	Mat KKC3 = Mat(1, 5, CV_32FC1);//相机畸变矩阵
	Mat KKC4 = Mat(1, 5, CV_32FC1);
	Mat KKC5 = Mat(1, 5, CV_32FC1);//相机畸变矩阵
	Mat KKC6 = Mat(1, 5, CV_32FC1);
	Mat mapx1;// x坐标映射函数
	Mat mapy1;//标映射函数
	Mat mapx3;// x坐标映射函数
	Mat mapy3;//标映射函数
	Mat mapx2;// x坐标映射函数
	Mat mapy2;//标映射函数
	Mat mapx4;// x坐标映射函数
	Mat mapy4;//标映射函数
	Mat mapx5;// x坐标映射函数
	Mat mapy5;//标映射函数
	Mat mapx6;// x坐标映射函数
	Mat mapy6;//标映射函数
	Mat ZMT_mapx1;
	Mat ZMT_mapy1;
	Mat ZMT_mapx2;
	Mat ZMT_mapy2;
	Mat ZMT_mapx3;
	Mat ZMT_mapy3;
	Mat ZMT_mapx4;
	Mat ZMT_mapy4;
	Mat ZMT_mapx5;
	Mat ZMT_mapy5;
	Mat ZMT_mapx6;
	Mat ZMT_mapy6;
	GpuMat Omapx1;// x坐标映射函数
	GpuMat Omapy1;//标映射函数
	GpuMat Omapx2;// x坐标映射函数
	GpuMat Omapy2;//标映射函数
	GpuMat Omapx3;// x坐标映射函数
	GpuMat Omapy3;//标映射函数
	GpuMat Omapx4;// x坐标映射函数
	GpuMat Omapy4;//标映射函数
	GpuMat Omapx5;// x坐标映射函数
	GpuMat Omapy5;//标映射函数
	GpuMat Omapx6;// x坐标映射函数
	GpuMat Omapy6;//标映射函数
	GpuMat OZMT_mapx1;
	GpuMat OZMT_mapy1;
	GpuMat OZMT_mapx2;
	GpuMat OZMT_mapy2;
	GpuMat OZMT_mapx3;
	GpuMat OZMT_mapy3;
	GpuMat OZMT_mapx4;
	GpuMat OZMT_mapy4;
	GpuMat OZMT_mapx5;
	GpuMat OZMT_mapy5;
	GpuMat OZMT_mapx6;
	GpuMat OZMT_mapy6;
	GpuMat Oimage1;
	GpuMat Oimage2;
	GpuMat Oimage3;
	GpuMat Oimage4;
	GpuMat Oimage5;
	GpuMat Oimage6;

	int W;
	int H;//实景图像宽和高
	int n;//图像数目
	double hfov;//照相机水平视角
	double f;//照相机像素焦距
	double u, v, w;//像素点在圆柱面上投影点的参数坐标
	double t;//参数

	bool mustInitUndistort;//map1与map2的构造标志位
	char image_name[15];
public:

	Mat cameraMatrix;//输出的矩阵
	Mat distCoeffs;

	Mat undistorted1;//映射图像
	Mat undistorted2;//映射图像
	Mat undistorted3;//映射图像
	Mat undistorted4;//映射图像
	Mat undistorted5;//映射图像
	Mat undistorted6;//映射图像
	Mat out1, out2, out3, out4, out5, out6;
	GpuMat Oundistorted1, Oundistorted11;//映射图像
	GpuMat Oundistorted2, Oundistorted22;//映射图像
	GpuMat Oundistorted3, Oundistorted33;//映射图像
	GpuMat Oundistorted4, Oundistorted44;//映射图像
	GpuMat Oundistorted5, Oundistorted55;//映射图像
	GpuMat Oundistorted6, Oundistorted66;//映射图像
	GpuMat Oout1, Oout2, Oout3, Oout4, Oout5, Oout6;
	Calibration()
	{


		flag = 0;
		mustInitUndistort = true; //旧
		KKK1 = (Mat_<float>(3, 3) <<
			374.944061, 0.000000, 379.609650,
			0.000000, 401.828400, 294.892242,
			0.000000, 0.000000, 1.000000);

		KKK2 = (Mat_<float>(3, 3) <<
			380.237732, 0.000000, 349.451752,
			0.000000, 406.140533, 326.110260,
			0.000000, 0.000000, 1.000000);
		KKC1 = (Mat_<float>(1, 5) <<
			-0.290591, 0.061375, -0.000572, -0.000461, 0.00000);
		KKC2 = (Mat_<float>(1, 5) <<
			-0.300528, 0.069788, -0.001043, 0.000197, 0);
		// mustInitUndistort = true; //新
		KKK3 = (Mat_<float>(3, 3) <<
			383.199127, 0.000000, 384.508820,
			0.000000, 409.421326, 276.038696,
			0.000000, 0.000000, 1.000000);

		KKK4 = (Mat_<float>(3, 3) <<
			368.904602, 0.000000, 350.237427,
			0.000000, 396.204407, 269.639954,
			0.000000, 0.000000, 1.000000);
		KKC3 = (Mat_<float>(1, 5) <<
			-0.312288, 0.074100, 0.002045, 0.001056, 0);
		KKC4 = (Mat_<float>(1, 5) <<
			-0.269501, 0.050341, -0.001472, -0.003330, 0);
		KKK5 = (Mat_<float>(3, 3) <<
			818.181458, 0.000000, 353.031860,
			0.000000, 855.869263, 271.299652,
			0.000000, 0.000000, 1.000000);

		KKK6 = (Mat_<float>(3, 3) <<
			378.760193, 0.000000, 358.046631,
			0.000000, 405.443726, 324.629395,
			0.000000, 0.000000, 1.000000);
		KKC5 = (Mat_<float>(1, 5) <<
			-1.216140, 1.392492, -0.027298, -0.032719, 0);
		KKC6 = (Mat_<float>(1, 5) <<
			-0.305479, 0.072930, 0.001872, -0.002252, 0);


	}
	void ZMTYInitial(int newW, int newH, float newn, Mat &Mapx, Mat &Mapy)//初始化图像宽度，高度，张数
	{
		W = newW;
		H = newH;
		n = newn;
		hfov = 2 * PI / n;
		f = W / 2 / tan(hfov / 2);
		Mapx.create(H, W, CV_32FC1);
		Mapy.create(H, W, CV_32FC1);
		//Form_ZMT.create(H, W, CV_64FC2);
		//Form_UV.create(H, W, CV_64FC4);
		for (int i = 0; i < H; i++)//i行 y //image.rows
		{
			for (int j = 0; j < W; j++)//第j列 x//image.cols
			{
				//x1 = f*(hfov / 2 + atan((x - W / 2) / f));
				//y1 = H / 2 + f*(y - H / 2) / sqrt((x - W / 2)*(x - W / 2) + f*f);
				Mapx.at<float>(i, j) = f*tan(j / f - hfov / 2) + W / 2;//
				Mapy.at<float>(i, j) = (i - H / 2)*sqrt((Mapx.at<float>(i, j) - W / 2)*(Mapx.at<float>(i, j) - W / 2) + f*f) / f + H / 2;//y
			}
		}
	}
	void Calibration_remap(cv::Mat &image1, cv::Mat &image2, cv::Mat &image3, cv::Mat &image4, cv::Mat &image5, cv::Mat &image6)
	{


		if (mustInitUndistort)
		{
			ZMTYInitial(int(0.6*image1.cols), int(0.6*image1.rows), 4.33, ZMT_mapx1, ZMT_mapy1);
			ZMTYInitial(int(0.6*image1.cols), int(0.6*image1.rows), 4.1, ZMT_mapx2, ZMT_mapy2);
			ZMTYInitial(int(0.6*image1.cols), int(0.6*image1.rows), 4, ZMT_mapx3, ZMT_mapy3);
			ZMTYInitial(int(0.6*image1.cols), int(0.6*image1.rows), 4.4, ZMT_mapx4, ZMT_mapy4);
			ZMTYInitial(int(0.6*image1.cols), int(0.6*image1.rows), 4.3, ZMT_mapx5, ZMT_mapy5);
			ZMTYInitial(int(0.6*image1.cols), int(0.6*image1.rows), 4.3, ZMT_mapx6, ZMT_mapy6);
			
			/*ZMTYInitial(int(image1.cols), int(image1.rows), 4.33, ZMT_mapx1, ZMT_mapy1);
			ZMTYInitial(int(image1.cols), int(image1.rows), 4.1, ZMT_mapx2, ZMT_mapy2);
			ZMTYInitial(int(image1.cols), int(image1.rows), 4, ZMT_mapx3, ZMT_mapy3);
			ZMTYInitial(int(image1.cols), int(image1.rows), 4.4, ZMT_mapx4, ZMT_mapy4);
			ZMTYInitial(int(image1.cols), int(image1.rows), 4.3, ZMT_mapx5, ZMT_mapy5);
			ZMTYInitial(int(image1.cols), int(image1.rows), 4.3, ZMT_mapx6, ZMT_mapy6);*/
			cv::initUndistortRectifyMap(KKK1,KKC1,cv::Mat(),cv::Mat(),Size(int(1.2*image1.cols), int(1.2*image1.rows)),CV_32FC1,mapx1,mapy1);
			cv::initUndistortRectifyMap(KKK2,KKC2,cv::Mat(),cv::Mat(),Size(int(1.2*image2.cols), int(1.2*image2.rows)),CV_32FC1,mapx2,mapy2);
			cv::initUndistortRectifyMap(KKK3,KKC3,cv::Mat(),cv::Mat(),Size(int(1.2*image3.cols), int(1.2*image3.rows)),CV_32FC1,mapx3,mapy3);
			cv::initUndistortRectifyMap(KKK4,KKC4,cv::Mat(),cv::Mat(),Size(int(1.2*image4.cols), int(1.2*image4.rows)),CV_32FC1,mapx4,mapy4);
			cv::initUndistortRectifyMap(KKK5,KKC5,cv::Mat(),cv::Mat(),Size(int(1.2*image5.cols), int(1.2*image5.rows)),CV_32FC1,mapx5,mapy5);
			cv::initUndistortRectifyMap(KKK6,KKC6,cv::Mat(),cv::Mat(),Size(int(1.2*image6.cols), int(1.2*image6.rows)),CV_32FC1,mapx6,mapy6);
			mustInitUndistort = false;
			Omapx1.upload(mapx1);
			Omapx2.upload(mapx2);
			Omapx3.upload(mapx3);
			Omapx4.upload(mapx4);
			Omapx5.upload(mapx5);
			Omapx6.upload(mapx6);
			Omapy1.upload(mapy1);
			Omapy2.upload(mapy2);
			Omapy3.upload(mapy3);
			Omapy4.upload(mapy4);
			Omapy5.upload(mapy5);
			Omapy6.upload(mapy6);

			OZMT_mapx1.upload(ZMT_mapx1);
			OZMT_mapy1.upload(ZMT_mapy1);
			OZMT_mapx2.upload(ZMT_mapx2);
			OZMT_mapy2.upload(ZMT_mapy2);
			OZMT_mapx3.upload(ZMT_mapx3);
			OZMT_mapy3.upload(ZMT_mapy3);
			OZMT_mapx4.upload(ZMT_mapx4);
			OZMT_mapy4.upload(ZMT_mapy4);
			OZMT_mapx5.upload(ZMT_mapx5);
			OZMT_mapy5.upload(ZMT_mapy5);
			OZMT_mapx6.upload(ZMT_mapx6);
			OZMT_mapy6.upload(ZMT_mapy6);

			cout << "相机标定参数初始化成功" << endl;
		}
		Mat img11, img22;
		Oimage1.upload(image1);
		Oimage2.upload(image2);
		Oimage3.upload(image3);
		Oimage4.upload(image4);
		Oimage5.upload(image5);
		Oimage6.upload(image6);
		cv::gpu::remap(Oimage1, Oundistorted1, Omapx1, Omapy1, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oimage2, Oundistorted2, Omapx2, Omapy2, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oimage3, Oundistorted3, Omapx3, Omapy3, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oimage4, Oundistorted4, Omapx4, Omapy4, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oimage5, Oundistorted5, Omapx5, Omapy5, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oimage6, Oundistorted6, Omapx6, Omapy6, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::pyrDown(Oundistorted1, Oundistorted11);
		cv::gpu::pyrDown(Oundistorted2, Oundistorted22);
		cv::gpu::pyrDown(Oundistorted3, Oundistorted33);
		cv::gpu::pyrDown(Oundistorted4, Oundistorted44);
		cv::gpu::pyrDown(Oundistorted5, Oundistorted55);
		cv::gpu::pyrDown(Oundistorted6, Oundistorted66);
		cv::gpu::remap(Oundistorted11, Oout1, OZMT_mapx1, OZMT_mapy1, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oundistorted22, Oout2, OZMT_mapx2, OZMT_mapy2, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oundistorted33, Oout3, OZMT_mapx3, OZMT_mapy3, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oundistorted44, Oout4, OZMT_mapx4, OZMT_mapy4, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oundistorted55, Oout5, OZMT_mapx5, OZMT_mapy5, INTER_LINEAR, BORDER_CONSTANT);
		cv::gpu::remap(Oundistorted66, Oout6, OZMT_mapx6, OZMT_mapy6, INTER_LINEAR, BORDER_CONSTANT);
		Oout1.download(out1);
		Oout2.download(out2);
		Oout3.download(out3);
		Oout4.download(out4);
		Oout5.download(out5);
		Oout6.download(out6);
		undistorted1 = out1(Range(0, out1.rows), Range(0, 340));
		undistorted2 = out2(Range(0, out2.rows), Range(0, 340));
		undistorted3 = out3(Range(0, out3.rows), Range(0, 340));
		undistorted4 = out4(Range(0, out4.rows), Range(0, 340));
		undistorted5 = out5(Range(0, out5.rows), Range(0, 340));
		undistorted6 = out6(Range(0, out6.rows), Range(0, 340));
	/*	undistorted1 = out1;
		undistorted2 = out2; 
		undistorted3 = out3; 
		undistorted4 = out4;
		undistorted5 = out5;
		undistorted6 = out6;*/

	}
	void L_CameraToImage(int i, Mat &mat)
	{

		printf("------------- camera to image%d ... ----------------n", i);
		sprintf(image_name, "%s%d%s", "L_Cameral", i, ".jpg");//保存的图片名
		imwrite(image_name, mat);
	}
	void R_CameraToImage(int i, Mat &mat)
	{

		printf("------------- camera to image%d... ----------------n", i);
		sprintf(image_name, "%s%d%s", "R_Cameral", i, ".jpg");//保存的图片名
		imwrite(image_name, mat);
	}
	~Calibration()
	{
		;
	}

};