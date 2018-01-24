#include "ImageDeal.h"
using namespace std;
using namespace cv;
Scalar getMSSIM_GPU(const Mat& i1, const Mat& i2);
static void VID_deal(cv::Mat& img1, cv::Mat& img2, cv::Mat & out)
{

	//旋转中心为图像中心  
	CvPoint2D32f center; 
	center.x = float(img1.cols / 2.0 + 0.5);
	center.y = float(img1.rows / 2.0 + 0.5);
	//计算二维旋转的仿射变换矩阵  
	float m[6];
	CvMat M = cvMat(2, 3, CV_32F, m);
	cv2DRotationMatrix(center, 180, 1, &M);
	//变换图像，并用黑色填充其余值  
	//cvWarpAffine(img1, img1, &M, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
}

ImageDeal::ImageDeal()
{
	callIt=true;
	ratio = 0.9f;//匹配比率
	refineH = false;
	flag_rh = true;
	confidence = 0.95;
	distance = 3.0;
	detector = new cv::SurfFeatureDetector(10);
	extractor = new cv::SurfDescriptorExtractor();
	
}
void ImageDeal::VideoToImage(int i,Mat &mat1,Mat &mat2)
{
	
	printf("------------- video to image%d ... ----------------n",i);
	sprintf(image_name1, "%s%s%d%s","stitching\\", "stitching", i, ".jpg");//保存的图片名
//	sprintf(image_name2, "%s%s%d%s","right\\", "R_image", i, ".jpg");//保存的图片名
	imwrite(image_name1, mat1);
//	imwrite(image_name2, mat2);
}

ImageDeal::~ImageDeal()
{
	dontDisplay();
}

/**********************************************************************
* 函数名:ratioTest
* 参  数:matches
* 返  回:移除的匹配数量
* 说  明:对当前匹配进行筛选，最优匹配和次优匹配响应强度大于ratio的匹配以及
*  孤立的匹配。
***********************************************************************/

int ImageDeal::ratioTest(std::vector<std::vector<cv::DMatch>>&matches)
{
	int removed = 0;

	for (std::vector<std::vector<cv::DMatch>>::iterator matcheIterator = matches.begin(); matcheIterator != matches.end(); ++matcheIterator)//遍历所有匹配
	{
		if (matcheIterator->size() > 1)//如果识别两个最近邻
		{
			if ((*matcheIterator)[0].distance / (*matcheIterator)[1].distance > ratio)//检查距离比率,比值越小说明最近邻代表性越好
			{
				matcheIterator->clear();//移除匹配
				removed++;
			}
		}
		else
		{
			matcheIterator->clear();
			removed++;
		}

	}
	return removed;
}

/**********************************************************************
* 函数名:symmetryTest
* 参  数:matches1:左匹配
*       matches2:右匹配
*       symMatche:输出的对称匹配
* 返  回:无
* 说  明:对左、右匹配进行检查，输出对称的匹配。
***********************************************************************/

void ImageDeal::symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches1, const std::vector<std::vector<cv::DMatch>>& matches2, std::vector<cv::DMatch>& symMatches)
{
	for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
	{
		if (matchIterator1->size() < 2)
			continue;
		for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator2 = matches2.begin();
			matchIterator2 != matches2.end();
			++matchIterator2)
		{
			if (matchIterator2->size() < 2)
				continue;
			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx && (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx)
			{
				symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx, (*matchIterator1)[0].distance));
				break;
			}
		}
	}

}

/**********************************************************************
* 函数名:ransacTest
* 参  数:matches:当前匹配（输入）
*       keypoints1:图像1检测到的关键点（输入）
*       keypoints2:图像2检测到的关键点（输入）
*       outMatches:完成测试的匹配（输出）
* 返  回:基础矩阵
* 说  明:对当前匹配进行RANSAC测试，计算基础矩阵，同时返回通过测试的匹配
***********************************************************************/

cv::Mat ImageDeal::ransacTest(const std::vector<cv::DMatch>&matches, const std::vector<cv::KeyPoint>&keypoints1, const std::vector<cv::KeyPoint>&keypoints2,int a)
{
	//将Keypoints转换到Point2f
	std::vector<cv::Point2f>points1, points2;
 	vector<unsigned char> match_mask;
	for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{   //左图像关键点
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));
		//右图像关键点
		x = keypoints2[it->trainIdx].pt.x+a;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}
	cv::Mat _pt1(1, matches.size(), CV_32FC2, &points1);
	cv::Mat _pt2(1, matches.size(), CV_32FC2, &points2);
	std::vector<uchar>inliers(points1.size(), 0);
	cv::Mat H1 = cv::findHomography(points2, points1, inliers, CV_RANSAC);

	vector<cv::Point2f>::const_iterator itPts = points1.begin();
	vector<uchar>::const_iterator itIn;
	for (itIn = inliers.begin(); itIn != inliers.end(); ++itIn, ++itPts)
	{
		if (*itIn)
			inlier_points1.push_back(*itPts);
	}
	itPts = points2.begin();
	for (itIn = inliers.begin(); itIn != inliers.end(); ++itIn, ++itPts)
	{
		if (*itIn)
			inlier_points2.push_back(*itPts);
	}
	return H1;
}

void ImageDeal::CalcFourCorner(cv::Mat H1,cv::Mat img22)
{
	//计算图2的四个角经矩阵H变换后的坐标
       cv::Point LeftTop, LeftBottom, RightTop, RightBottom;
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值

	cv::Mat V2 = cv::Mat(3, 1, CV_64FC1, v2);
	cv::Mat V1 = cv::Mat(3, 1, CV_64FC1, v1);
	cv::gemm(H1, V2, 1, 1, 0, V1);//矩阵乘法////the src3 can't be 0,so if you want to ignore the src3,
	//you should set the beater as 0
        
	LeftTop.x = cvRound(v1[0] / v1[2]);
	LeftTop.y = cvRound(v1[1] / v1[2]);
	//cvCircle(xformed,leftTop,7,CV_RGB(255,0,0),2);

	//将v2中数据设为左下角坐标
	v2[0] = 0;
	v2[1] = img22.rows;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);
	V1 = cv::Mat(3, 1, CV_64FC1, v1);
	cv::gemm(H1, V2, 1, 1, 0, V1);
	LeftBottom.x = cvRound(v1[0] / v1[2]);
	LeftBottom.y = cvRound(v1[1] / v1[2]);
	//cvCircle(xformed,leftBottom,7,CV_RGB(255,0,0),2);

	//将v2中数据设为右上角坐标
	v2[0] = img22.cols;
	v2[1] = 0;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);
	V1 = cv::Mat(3, 1, CV_64FC1, v1);
	cv::gemm(H1, V2, 1, 1, 0, V1);
	rightTop.x = cvRound(v1[0] / v1[2]);
	rightTop.y = cvRound(v1[1] / v1[2]);
	//cvCircle(xformed,rightTop,7,CV_RGB(255,0,0),2);

	//将v2中数据设为右下角坐标
	v2[0] = img22.cols;
	v2[1] = img22.rows;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);
	V1 = cv::Mat(3, 1, CV_64FC1, v1);
	cv::gemm(H1, V2, 1, 1, 0, V1);
	rightBottom.x = cvRound(v1[0] / v1[2]);
	rightBottom.y = cvRound(v1[1] / v1[2]);
	
	
	
	if(LeftTop.y<0)
	{
	  leftTop.y=0;
	  leftTop.x= (int)(LeftTop.x+(-LeftTop.y)/(double)(LeftBottom.y-LeftTop.y)*(LeftBottom.x-LeftTop.x));
	}
	else
	{
	  leftTop.x=LeftTop.x;
	  leftTop.y=LeftTop.y;
	}
	if(LeftBottom.y>img22.rows)
	{
	  leftBottom.y=img22.rows;
	  leftBottom.x= (int)(LeftBottom.x-(LeftBottom.y-img22.rows)/(double)(LeftBottom.y-LeftTop.y)*(LeftBottom.x-LeftTop.x));
	}
	else
	{
	  leftBottom.x=LeftBottom.x;
	  leftBottom.y=LeftBottom.y;
	}/*
	printf("LeftTop=(%d,%d)\n",LeftTop.x,LeftTop.y);
	printf("leftTop=(%d,%d)\n",leftTop.x,leftTop.y);
	printf("LeftBottom=(%d,%d)\n",LeftBottom.x,LeftBottom.y);
	printf("leftBottom=(%d,%d)\n",leftBottom.x,leftBottom.y);*/
	
	//cvCircle(xformed,rightBottom,7,CV_RGB(255,0,0),2);
	/*vector<cv::Point2f>::const_iterator itIn;
	Point2f point_out;
	for (itIn = inlier_points2.begin(); itIn != inlier_points2.end(); ++itIn)
	{
		v2[0] = (*itIn).x;
		v2[1] = (*itIn).y;
		V2 = cv::Mat(3, 1, CV_64FC1, v2);
		V1 = cv::Mat(3, 1, CV_64FC1, v1);
		cv::gemm(H1, V2, 1, 1, 0, V1);
		point_out.x = cvRound(v1[0] / v1[2]);
		point_out.y = cvRound(v1[1] / v1[2]);

		inlier_points2_Out.push_back(point_out);
	}
*/
}

void ImageDeal::setFrameDeal(void(*frameDealCallback)(cv::Mat &, cv::Mat &, cv::Mat &))//初始化
{
	process = frameDealCallback;
}
void ImageDeal::setOutput(Size videoSize = { 700, 572 }, double framerate = 10.0, bool isColor = false)
{
	writer.open("output.avi", CV_FOURCC('P', 'I', 'M', '1'), framerate, videoSize, true);
}

void ImageDeal::displayInput(std::string wn1, std::string wn2)//建立输入窗口
{
	windowNameInput1 = wn1;
	windowNameInput2 = wn2;
	cv::namedWindow(windowNameInput1);
	cv::namedWindow(windowNameInput2);
}

void ImageDeal::displayOutput(std::string wn)//建立输出窗口
{
	windowNameOutput = wn;
	cv::namedWindow(windowNameOutput);
}

void ImageDeal::dontDisplay()//销毁窗口
{
	destroyAllWindows();

}

/**********************************************************************
* 函数名:run
* 参  数:image1 :图像1（输入左侧）
*        image2:图像2（输入右侧）
*        matches:经过多重测试剩下的高质量的匹配（输出）
*        keypoints1:用于保存图像1检测到的关键点（输出）
*        keypoints2:用于保存图像2检测到的关键点（输出）
* 返  回:无
* 说  明:对输出的两幅图像进行特征检测、计算描述子，进而使用BruteForceMatcher
* 进行匹配，对初始得到的匹配关系再依次进行比率测试、对称测试最后进行Ransac
* 验证，并得到两个相机。
***********************************************************************/

bool ImageDeal::Feature_Detecor(cv::Mat &out, cv::Mat& image_left, cv::Mat& image_right)
{
  //img1 = Mat(image1.rows, image1.cols, image1.type(), Scalar(0, 0, 0));
//Mat img1_deal = image1(Range(0, image1.rows), Range(image1.cols - 320, image1.cols));
//Mat halff(img1, Range(0, img1.rows), Range(image1.cols - 320, img1.cols));
//img2 = image2(Range(0, image2.rows), Range(0, 320));
//img1_deal.copyTo(halff);		
cvtColor(image_right, img1_roi, CV_BGR2GRAY);
cvtColor(image_left, img2_roi, CV_BGR2GRAY);
//1a.测Surf特征
gpu::GpuMat img1_gpu(img1_roi);
gpu::GpuMat img2_gpu(img2_roi);
gpu::GpuMat keypoints_gpu_1, keypoints_gpu_2;
gpu::GpuMat descriptors_gpu_1, descriptors_gpu_2;
gpu::SURF_GPU FeatureFinder_gpu(200);
FeatureFinder_gpu.extended = false;
//计算特征点和特征描述子  
FeatureFinder_gpu(img1_gpu, gpu::GpuMat(), keypoints_gpu_1, descriptors_gpu_1);
FeatureFinder_gpu(img2_gpu, gpu::GpuMat(), keypoints_gpu_2, descriptors_gpu_2);

cout << "FOUND " << keypoints_gpu_1.cols << " keypoints on first image" << endl;
cout << "FOUND " << keypoints_gpu_2.cols << " keypoints on second image" << endl;
  
//将特征点下载回cpu，便于画图使用  
FeatureFinder_gpu.downloadKeypoints(keypoints_gpu_1, keypoints1);
FeatureFinder_gpu.downloadKeypoints(keypoints_gpu_2, keypoints2);
//使用gpu提供的BruteForceMatcher进行特征点匹配 
std::vector< vector<DMatch> >matches1, matches2;
//gpu::FlannBasedMatcher<L2<float>>matcher_lk;
gpu::BruteForceMatcher_GPU< L2<float> > matcher_lk;
matcher_lk.knnMatch(descriptors_gpu_1, descriptors_gpu_2, matches1, 2);
matcher_lk.knnMatch(descriptors_gpu_2, descriptors_gpu_1, matches2, 2);
int removed = ratioTest(matches1);
//cout << "图1到图2被的匹配点有" << (removed) << "个" << endl;
removed = ratioTest(matches2);
//cout << "图2到图1被的匹配点有" << (removed) << "个" << endl;
std::vector<cv::DMatch>symMatches;
symmetryTest(matches1, matches2, symMatches);
cout << "matche_best count:" << symMatches.size() << endl;
#if ShowMatcher
Mat imgMatcher;
cv::drawMatches(
image_right, keypoints1,
image_left, keypoints2,
symMatches,
imgMatcher,
cv::Scalar::all(-1),
cv::Scalar::all(-1),
vector<char>(),
DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS	
);
imshow("imgMatcher", imgMatcher);
#endif
return true;
}

bool ImageDeal::run(cv::Mat &out,cv::Mat& image1, cv::Mat& image2)//视频输入、处理及输出部分
{
  
   	if (H.empty()&&(!image1.empty()) && (!image2.empty()) || flag_rh)
	{
		
		Down_flag = true;
		//img1 = Mat(image1.rows, image1.cols, image1.type(), Scalar(0, 0, 0));
		//Mat img1_deal = image1(Range(0, image1.rows), Range(image1.cols - 320, image1.cols));
		//Mat halff(img1, Range(0, img1.rows), Range(image1.cols - 320, img1.cols));
		//img2 = image2(Range(0, image2.rows), Range(0, 320));
		//img1_deal.copyTo(halff);		
		cvtColor(image1, img1_roi, CV_BGR2GRAY);
		cvtColor(image2, img2_roi, CV_BGR2GRAY);
		//1a.测Surf特征
		gpu::GpuMat img1_gpu(img1_roi);
		gpu::GpuMat img2_gpu(img2_roi);
		gpu::GpuMat keypoints_gpu_1, keypoints_gpu_2;
		gpu::GpuMat descriptors_gpu_1, descriptors_gpu_2;
		gpu::SURF_GPU FeatureFinder_gpu(200);
		FeatureFinder_gpu.extended = false;
		//计算特征点和特征描述子  
		FeatureFinder_gpu(img1_gpu, gpu::GpuMat(), keypoints_gpu_1, descriptors_gpu_1);
		FeatureFinder_gpu(img2_gpu, gpu::GpuMat(), keypoints_gpu_2, descriptors_gpu_2);
		//将特征点下载回cpu，便于画图使用  
		FeatureFinder_gpu.downloadKeypoints(keypoints_gpu_1, keypoints1);
		FeatureFinder_gpu.downloadKeypoints(keypoints_gpu_2, keypoints2);
		//使用gpu提供的BruteForceMatcher进行特征点匹配 
		std::vector< vector<DMatch> >matches1, matches2;
		//gpu::FlannBasedMatcher<L2<float>>matcher_lk;
		gpu::BruteForceMatcher_GPU< L2<float> > matcher_lk;
		matcher_lk.knnMatch(descriptors_gpu_1, descriptors_gpu_2, matches1, 2);
		matcher_lk.knnMatch(descriptors_gpu_2, descriptors_gpu_1, matches2, 2);
		int removed = ratioTest(matches1);
		//cout << "图1到图2被的匹配点有" << (removed) << "个" << endl;
		removed = ratioTest(matches2);
		//cout << "图2到图1被的匹配点有" << (removed) << "个" << endl;
		std::vector<cv::DMatch>symMatches;
		symmetryTest(matches1, matches2, symMatches);
		cout << "matche_best count:" << symMatches.size() << endl;
#if ShowMatcher
		Mat imgMatcher;
		cv::drawMatches(
		image1, keypoints1,
		image2, keypoints2,
		symMatches,
		imgMatcher,
		cv::Scalar::all(-1),
		cv::Scalar::all(-1),
		vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS	
		);
		imshow("imgMatcher", imgMatcher);
#endif
		if (symMatches.size() >= 10)
			{ 
				H = ransacTest(symMatches, keypoints1, keypoints2, 0);
				H_per = H;
				//FileStorage fs("H.xml", FileStorage::WRITE);
				//fs << "H" << H;
				//fs.release();
				//imwrite("1.jpg", image2);
				//imwrite("2.jpg", image1);
			}
		else
			{
				cout <<"全景图匹配对过少，提高图像质量或拼接角度重新匹配" << endl;
				if (H_per.empty())
				{
					H.release();
					return false;
					out.release();
				}
				else
					H = H_per;				
			}
	
	}
	cout<<"H="<<!H.empty()<<endl;
	if (!H.empty()) //0.07s
	{
		
		CalcFourCorner(H,image2);
		int mRows = image2.rows;
		if (image1.rows > image2.rows)
		{
			mRows = image1.rows;
		}
		gpu::GpuMat G_image2(image2);
		gpu::GpuMat G_stitchedImage;
		
		cv::gpu::warpPerspective(G_image2, G_stitchedImage, H, cv::Size(image1.cols + image2.cols, mRows));
		G_stitchedImage.download(stitchedImage);
		imshow("stitchedImage", stitchedImage);
		xformed_proc = stitchedImage.clone();
        xformed = stitchedImage.clone();
		int Lcols = cv::min(leftTop.x, leftBottom.x);
		if (cv::min(leftTop.x, leftBottom.x) > image1.cols)
		{
			Lcols = image1.cols;
		}
		else if(cv::min(leftTop.x, leftBottom.x) < 0)
		{
			H.release();
			cout << "全景H提取失败，请重新尝试" << endl;	
			return false;
		}
		img1_ROI = image1(cv::Rect(0, 0, Lcols, xformed_proc.rows));//img1上感兴趣的区域，为一个矩形 Rect（左上点x,y，x方向的长，y方向的宽）
		xformed_ROI = xformed_proc(cv::Rect(0, 0, Lcols, xformed_proc.rows));
		cv::addWeighted(img1_ROI, 1, xformed_ROI, 0, 0, xformed_proc(cv::Rect(0, 0, Lcols, xformed_proc.rows)));//img1_ROI*1+xformed_ROI*0->xformed_proc的矩形区域内。
	//	imshow("xformed_proc", xformed_proc);
		int nl = xformed_proc.rows;
		int nc1 = image1.cols;
		int start = Lcols;
		//    double processWidth=img1.cols*img1.channels()-start;
		double processWidth = img1.cols - start;
		nc1 = nc1 - processWidth / 2;
		double alpha = 1;
		for (int j = 0; j < nl; j++)
		{
			uchar *data = xformed_proc.ptr<uchar>(j);
			uchar *data_img1 = image1.ptr<uchar>(j);
			uchar *data_xformed = xformed_proc.ptr<uchar>(j);
			for (int i = start; i < nc1; i++)
			{
				if (i < start + 50)
					alpha = 1;
				else
				{
					alpha = (nc1 - start - (i - start)) / (double)(nc1 - start - 50);
					alpha = alpha*alpha*alpha;
				}
				data[i * 3] = data_img1[i * 3] * alpha + data_xformed[i * 3] * (1 - alpha);
				data[i * 3 + 1] = data_img1[i * 3 + 1] * alpha + data_xformed[i * 3 + 1] * (1 - alpha);
				data[i * 3 + 2] = data_img1[i * 3 + 2] * alpha + data_xformed[i * 3 + 2] * (1 - alpha);
			}
		}
		/************************************quality of image stitching*****************************************/
		/*if (flag_c == true)
		{
	         Mat image1_ROI, image_stitch_ROI;
		image1_ROI = image1(Range(0.3*image1.rows, 0.7*image1.rows), Range(0.80*image1.cols, image1.cols));
		image_stitch_ROI = xformed_proc(Range(0.3*image1.rows, 0.7*image1.rows), Range(0.80*image1.cols, image1.cols));
		//cvNamedWindow("image_stitch_ROI", 0);
		//cvNamedWindow("image1_ROI", 0);
		//imshow("image_stitch_ROI", image_stitch_ROI);
		//imshow("image1_ROI", image1_ROI);
		quality_Num = getMSSIM_GPU(image_stitch_ROI, image1_ROI);
		Quality_Num = (quality_Num.val[0] + quality_Num.val[1] + quality_Num.val[2]) / 3.0;
		cout << "MSSIM GPU "
			<< " With result of " << Quality_Num << endl;
		if (Quality_Num < Expect_quality)
		{
			H.release();
			return false;
		}
		}
	
		/************************************quality of image stitching*****************************************/
		out = xformed_proc;
	}
	return true;
}

bool ImageDeal::run1(cv::Mat &out, cv::Mat& image_r, cv::Mat& image_m,cv::Mat &image_l,int image_num)//视频输入、处理及输出部分
{
	

	if (image_num == 612)
	{
		if (!H21.empty()&&!H61.empty())
		{
			H_r = H21;
			H_l = H61;
		}
		else
		{
			refineH = true;
		}
	}
	else if (image_num == 345)
	{
		if (!H34.empty()&&!H54.empty())
		{
			H_r = H54;
			H_l = H34;
		}
		else
		{
			refineH = true;
		}
	}
	else
	{
		cout << "未指定3个匹配图像" << endl;
		return false;
	}
	
	if ((refineH &&!image_l.empty()) && (!image_m.empty()) && (!image_r.empty()) || flag_rh)
	{ 
		cvtColor(image_l, img1, CV_BGR2GRAY);
	    cvtColor(image_m, img2, CV_BGR2GRAY);
		cvtColor(image_r, img3, CV_BGR2GRAY);

		std::vector<cv::DMatch> matches_mr;
		std::vector<cv::DMatch> matches_lm;
		std::vector<cv::KeyPoint> keypoints_r;
		std::vector<cv::KeyPoint> keypoints_lm;
		std::vector<cv::KeyPoint> keypoints_mr;
		std::vector<cv::KeyPoint> keypoints_l;

		img1_roi = Mat(img1.rows, img1.cols, img1.type(), Scalar(0, 0, 0));
		Mat img1_deal = img1(Range(0, img1.rows), Range(0.5*img1.cols, img1.cols));
		Mat halff(img1_roi, Range(0, img1.rows), Range(0.5*img1.cols, img1.cols));
	    img1_deal.copyTo(halff);

		img2_roi_mr = img2;// Mat(img2.rows, img2.cols, img2.type(), Scalar(0, 0, 0));
	//	Mat img2_deal = img2(Range(0, img2.rows), Range(0.5*img2.cols,img2.cols));
		//Mat halff1(img2_roi_mr, Range(0, img2.rows), Range(0.5*img2.cols,img2.cols));
		//img2_deal.copyTo(halff1);

		img2_roi_lm = img2;// (Range(0, img2.rows), Range(0, 0.5*img2.cols));
		img3_roi = img3;// (Range(0, img3.rows), Range(0, 0.5*img3.cols));
	
		/*imshow("l", img1_roi);
		imshow("lm", img2_roi_lm);
		imshow("mr", img2_roi_mr);
		imshow("r", img3_roi);
		waitKey(0);*/
		//1a.测Surf特征

		gpu::GpuMat img1_gpu(img1_roi);
		gpu::GpuMat img2_lm_gpu(img2_roi_lm);
		gpu::GpuMat img2_mr_gpu(img2_roi_mr);
		gpu::GpuMat img3_gpu(img3_roi);
		gpu::GpuMat keypoints_gpu_l, keypoints_gpu_lm, keypoints_gpu_mr, keypoints_gpu_r;
		gpu::GpuMat descriptors_gpu_1, descriptors_gpu_2, descriptors_gpu_3, descriptors_gpu_4;
		gpu::SURF_GPU FeatureFinder_gpu(100);
		FeatureFinder_gpu.extended = false;
		//计算特征点和特征描述子  
		FeatureFinder_gpu(img1_gpu, gpu::GpuMat(), keypoints_gpu_l, descriptors_gpu_1);
		FeatureFinder_gpu(img2_lm_gpu, gpu::GpuMat(), keypoints_gpu_lm, descriptors_gpu_2);
		FeatureFinder_gpu(img2_mr_gpu, gpu::GpuMat(), keypoints_gpu_mr, descriptors_gpu_3);
		FeatureFinder_gpu(img3_gpu, gpu::GpuMat(), keypoints_gpu_r, descriptors_gpu_4);
		//将特征点下载回cpu，便于画图使用  
		FeatureFinder_gpu.downloadKeypoints(keypoints_gpu_l, keypoints_l);
		FeatureFinder_gpu.downloadKeypoints(keypoints_gpu_lm, keypoints_lm);
		FeatureFinder_gpu.downloadKeypoints(keypoints_gpu_mr, keypoints_mr);
		FeatureFinder_gpu.downloadKeypoints(keypoints_gpu_r, keypoints_r);
		//使用gpu提供的BruteForceMatcher进行特征点匹配 
		gpu::BruteForceMatcher_GPU< L2<float> > matcher_lk;
		std::vector<std::vector<cv::DMatch>>matches1;
		matcher_lk.knnMatch(descriptors_gpu_1, descriptors_gpu_2, matches1, 2);
		//从图2到图1的k最近邻
		std::vector<std::vector<cv::DMatch>>matches2;
		matcher_lk.knnMatch(descriptors_gpu_2, descriptors_gpu_1, matches2, 2);
		//3.移除NN比率大于阈值的匹配
		//从图一到图二的k最近邻
		std::vector<std::vector<cv::DMatch>>matches3;
		matcher_lk.knnMatch(descriptors_gpu_3, descriptors_gpu_4, matches3, 2);
		//从图2到图1的k最近邻
		std::vector<std::vector<cv::DMatch>>matches4;
		matcher_lk.knnMatch(descriptors_gpu_4, descriptors_gpu_3, matches4, 2);


		//detector->detect(img1_roi, keypoints_l);
		//detector->detect(img2_roi_lm, keypoints_lm);
		//detector->detect(img2_roi_mr, keypoints_mr);
		//detector->detect(img3_roi, keypoints_r);
		////1b.取描述子
		//cv::Mat descriptors1, descriptors2,descriptors3,descriptors4;
		//extractor->compute(img1_roi, keypoints_l, descriptors1);
		//extractor->compute(img2_roi_lm, keypoints_lm, descriptors2);
		//extractor->compute(img2_roi_mr, keypoints_mr, descriptors3);
		//extractor->compute(img3_roi, keypoints_r, descriptors4);
		//2.配两幅图的描述子
		//创建匹配器
		//cv::BruteForceMatcher<cv::L2<float>>matcher;
		//FlannBasedMatcher matcher;  
		//从图一到图二的k最近邻
		//std::vector<std::vector<cv::DMatch>>matches1;
		//matcher.knnMatch(descriptors1, descriptors2, matches1, 2);
		////从图2到图1的k最近邻
		//std::vector<std::vector<cv::DMatch>>matches2;
		//matcher.knnMatch(descriptors2, descriptors1, matches2, 2);
		////3.移除NN比率大于阈值的匹配
		////从图一到图二的k最近邻
		//std::vector<std::vector<cv::DMatch>>matches3;
		//matcher.knnMatch(descriptors3, descriptors4, matches3, 2);
		////从图2到图1的k最近邻
		//std::vector<std::vector<cv::DMatch>>matches4;
		//matcher.knnMatch(descriptors4, descriptors3, matches4, 2);



		//清理图1到图2的匹配
		int removed = ratioTest(matches1);
		//cout << "图1到图2被的匹配点有" << (matches1.size() - removed) << "个" << endl;
		removed = ratioTest(matches2);
		//cout << "图2到图1被的匹配点有" << (matches2.size() - removed) << "个" << endl;
		removed = ratioTest(matches3);
		removed = ratioTest(matches4);
		//4.移除非对称匹配
		std::vector<cv::DMatch>symMatches1,symMatches2;
		cv::Mat out1,out2;
		symmetryTest(matches2, matches1, symMatches1);
		cout << "run1 l-m有效的匹配点有" << symMatches1.size() << "个" << endl;
	/*	cv::drawMatches(
			image_m, keypoints_lm,
			image_l, keypoints_l,
			symMatches1,
			out1,
			cv::Scalar(255, 255, 255)
			);
		imshow("out1", out1);*/
		symmetryTest(matches4, matches3, symMatches2);
		cout << "run1 m-r有效的匹配点有" << symMatches2.size() << "个" << endl;
		/*cv::drawMatches(
			image_r, keypoints_r,
			image_m, keypoints_mr,
			symMatches2,
			out2,
			cv::Scalar(255, 255, 255)
			);
		imshow("out2", out2);
		waitKey(0);*/
		//5.得到H矩阵
		if ((symMatches1.size()>10)&&(symMatches2.size()>10))
		{	
			H_l = ransacTest(symMatches1, keypoints_lm, keypoints_l,0);
			H_r = ransacTest(symMatches2, keypoints_r, keypoints_mr,img1.cols);
			if (image_num == 612)
			{
            FileStorage fs1("H21.xml", FileStorage::WRITE);
			FileStorage fs2("H61.xml", FileStorage::WRITE);
			H61 = H_l;
			H21 = H_r;	
			fs2 << "H6_1" << H61;
		    fs1 << "H2_1" << H21;
			fs1.release();
			fs2.release();
			}
			else
			{
            FileStorage fs3("H34.xml", FileStorage::WRITE);
			FileStorage fs4("H54.xml", FileStorage::WRITE);
			H34 = H_l;
			H54 = H_r;
			fs3 << "H3_4" << H34;
			fs4 << "H5_4" << H54;
			fs3.release();
			fs4.release();
			}		
			refineH = false;	
		}
		else
		{
			//Down_flag = false;
			cout << "图  " << image_num << " 匹配对过少，提高图像质量或拼接角度重新匹配" << endl;
			refineH = false;
			//cout << "refineH" << refineH << endl;
			H_r.release();
			H_l.release();
			return false;
			
		}

		flag_rh = false;
	}
	if ((!H_r.empty())&&(!H_l.empty()))
	{
		double colsss = image_l.cols;
		double shftm[3][3] = { 1.0, 0, colsss, 0, 1.0, 0, 0, 0, 1.0 };
		Mat shftMat(3, 3, CV_64FC1, shftm);
		CalcFourCorner(shftMat*H_l, image_l);
		int mRows = image_m.rows;
		Mat tiledImg;
		int Lcols = cv::max(rightTop.x, rightBottom.x);
		if (cv::max(rightTop.x, rightBottom.x) <image_l.cols)
		{
			Lcols = image_l.cols;
		}
		gpu::GpuMat image_gpu_l(image_l);
		gpu::GpuMat tiledImag_gpu;
		cv::gpu::warpPerspective(image_gpu_l, tiledImag_gpu, shftMat*H_l, Size(image_l.cols + image_m.cols, mRows));
		tiledImag_gpu.download(tiledImg);
		shftMat.release();
	//	imshow("tiledImg11", tiledImg);
	//	waitKey(0);
		Mat xformed = tiledImg.clone();
		int large1 = Lcols - image_l.cols;
		int large2 = image_m.cols - Lcols + image_l.cols;
		if ((large1 <= 0)||(large2 <= 0))
			return false;
		Mat img1_ROI = image_m(cv::Rect(large1, 0,large2, mRows));//img1上感兴趣的区域，为一个矩形 Rect（左上点x,y，x方向的长，y方向的宽）
		Mat xformed_ROI = tiledImg(cv::Rect(Lcols, 0, image_m.cols - Lcols + image_l.cols, mRows));
		cv::addWeighted(img1_ROI, 1, xformed_ROI, 0, 0, tiledImg(cv::Rect(Lcols, 0, image_m.cols - Lcols + image_l.cols, mRows)));//img1_ROI*1+xformed_ROI*0->xformed_proc的矩形区域内。
		int nl = tiledImg.rows;
		int nc1 = Lcols;
		int start = min(rightTop.x, rightBottom.x);;
		float alpha = 1;
		for (int j = 0; j < nl; j++)
		{
			uchar *data = tiledImg.ptr<uchar>(j);
			uchar *data_img1 = image_m.ptr<uchar>(j);
			uchar *data_xformed = xformed.ptr<uchar>(j);
			for (int i = start; i < nc1; i++)
			{	
				alpha = (nc1 - i) / (float)(nc1 - start);
				alpha = alpha*alpha*alpha;
				data[i * 3] = data_img1[(i - image_l.cols) * 3] * (1 - alpha) + data_xformed[i * 3] * alpha;
				data[i * 3 + 1] = data_img1[(i - image_l.cols) * 3 + 1] * (1 - alpha) + data_xformed[i * 3 + 1] * alpha;
				data[i * 3 + 2] = data_img1[(i - image_l.cols) * 3 + 2] * (1 - alpha) + data_xformed[i * 3 + 2] * alpha;
			}
		}
	//	imshow("tiledImg", tiledImg);
//		waitKey(0);

    Mat TiledImg = tiledImg.clone();
	     double shftm1[3][3] = { 1.0, 0, (double)TiledImg.cols, 0, 1.0, 0, 0, 0, 1.0 };
		Mat shftMat1(3, 3, CV_64FC1, shftm1);
		CalcFourCorner(shftMat1*H_r, TiledImg);
		mRows = image_r.rows;
		Lcols = cv::max(rightTop.x, rightBottom.x);
		if (cv::max(rightTop.x, rightBottom.x) <TiledImg.cols)
		{
			Lcols = TiledImg.cols;
		}
		gpu::GpuMat TiledImg_gpu;
		TiledImg_gpu.upload(TiledImg);
		gpu::GpuMat tiledImg_gpu;
		cv::gpu::warpPerspective(TiledImg_gpu, tiledImg_gpu, shftMat1*H_r, Size(TiledImg.cols + image_r.cols, mRows));
		tiledImg_gpu.download(tiledImg);
		shftMat1.release();
		xformed = tiledImg.clone();
		int R_roi;
		R_roi = Lcols - TiledImg.cols;
		if (R_roi > image_r.cols)
		{
            //return false;
			R_roi = 0.3*image_r.cols;
		}
	//	imshow("tiledImg111", xformed);
	//	waitKey(0);

		img1_ROI = image_r(cv::Rect(R_roi, 0, image_r.cols - R_roi, mRows));//img1上感兴趣的区域，为一个矩形 Rect（左上点x,y，x方向的长，y方向的宽）
		xformed_ROI = tiledImg(cv::Rect(TiledImg.cols+R_roi, 0, image_r.cols - R_roi, mRows));
		cv::addWeighted(img1_ROI, 1, xformed_ROI, 0, 0, tiledImg(cv::Rect(TiledImg.cols + R_roi, 0, image_r.cols - R_roi, mRows)));//img1_ROI*1+xformed_ROI*0->xformed_proc的矩形区域内。
	//	imshow("tiledImg222", xformed);
	//	waitKey(0);
		nl = tiledImg.rows;
		nc1 = Lcols;
		start = min(rightTop.x, rightBottom.x);;
		alpha = 1;
		for (int j = 0; j < nl; j++)
		{
			uchar *data = tiledImg.ptr<uchar>(j);
			uchar *data_img1 = image_r.ptr<uchar>(j);
			uchar *data_xformed = xformed.ptr<uchar>(j);
			for (int i = start; i < nc1; i++)
			{
				alpha = (nc1 - i) / (float)(nc1 - start);
				alpha = alpha*alpha*alpha;
				data[i * 3] = data_img1[(i - TiledImg.cols) * 3] * (1 - alpha) + data_xformed[i * 3] * alpha;
				data[i * 3 + 1] = data_img1[(i - TiledImg.cols) * 3 + 1] * (1 - alpha) + data_xformed[i * 3 + 1] * alpha;
				data[i * 3 + 2] = data_img1[(i - TiledImg.cols) * 3 + 2] * (1 - alpha) + data_xformed[i * 3 + 2] * alpha;
			}
		}
	//	imshow("tiledImg1", tiledImg);
	//	waitKey(0);

		out = tiledImg;
	}
	return true;
}

bool ImageDeal::run2(cv::Mat &out, cv::Mat& image_r, cv::Mat& image_m, cv::Mat &image_l, int image_num)//视频输入、处理及输出部分
{


	if (image_num == 612)
	{
		if (!H21.empty() && !H61.empty())
		{
			H_r = H21;
			H_l = H61;
		}
		else
		{
			refineH = true;
		}
	}
	else if (image_num == 345)
	{
		if (!H34.empty() && !H54.empty())
		{
			H_r = H54;
			H_l = H34;
		}
		else
		{
			refineH = true;
		}
	}
	else
	{
		cout << "未指定3个匹配图像" << endl;
		return false;
	}

	if (refineH &&(!image_l.empty()) && (!image_m.empty()) && (!image_r.empty()) || flag_rh)
	{
		cvtColor(image_l, img1, CV_BGR2GRAY);
		cvtColor(image_m, img2, CV_BGR2GRAY);
		cvtColor(image_r, img3, CV_BGR2GRAY);

		std::vector<cv::DMatch> matches_mr;
		std::vector<cv::DMatch> matches_lm;
		std::vector<cv::KeyPoint> keypoints_r;
		std::vector<cv::KeyPoint> keypoints_lm;
		std::vector<cv::KeyPoint> keypoints_mr;
		std::vector<cv::KeyPoint> keypoints_l;

		img1_roi = Mat(img1.rows, img1.cols, img1.type(), Scalar(0, 0, 0));
		Mat img1_deal = img1(Range(0, img1.rows), Range(0.5*img1.cols, img1.cols));
		Mat halff(img1_roi, Range(0, img1.rows), Range(0.5*img1.cols, img1.cols));
		img1_deal.copyTo(halff);

		img2_roi_mr = img2;//Mat(img2.rows, img2.cols, img2.type(), Scalar(0, 0, 0));
		//Mat img2_deal = img2(Range(0, img2.rows), Range(0.5*img2.cols, img2.cols));
		//Mat halff1(img2_roi_mr, Range(0, img2.rows), Range(0.5*img2.cols, img2.cols));
		//img2_deal.copyTo(halff1);

		img2_roi_lm = img2;// (Range(0, img2.rows), Range(0, 0.5*img2.cols));
		img3_roi = img3;// (Range(0, img3.rows), Range(0, 0.5*img3.cols));

		/*imshow("l", img1_roi);
		imshow("lm", img2_roi_lm);
		imshow("mr", img2_roi_mr);
		imshow("r", img3_roi);
		waitKey(0);*/
		//1a.测Surf特征

		gpu::GpuMat img1_gpu(img1_roi);
		gpu::GpuMat img2_lm_gpu(img2_roi_lm);
		gpu::GpuMat img2_mr_gpu(img2_roi_mr);
		gpu::GpuMat img3_gpu(img3_roi);
		gpu::GpuMat keypoints_gpu_l, keypoints_gpu_lm, keypoints_gpu_mr, keypoints_gpu_r;
		gpu::GpuMat descriptors_gpu_1, descriptors_gpu_2, descriptors_gpu_3, descriptors_gpu_4;
	    gpu::SURF_GPU FeatureFinder_gpu2(300);
		FeatureFinder_gpu2.extended = false;
		////计算特征点和特征描述子  
		FeatureFinder_gpu2(img1_gpu, gpu::GpuMat(), keypoints_gpu_l, descriptors_gpu_1);
		FeatureFinder_gpu2(img2_lm_gpu, gpu::GpuMat(), keypoints_gpu_lm, descriptors_gpu_2);
		FeatureFinder_gpu2(img2_mr_gpu, gpu::GpuMat(), keypoints_gpu_mr, descriptors_gpu_3);
		FeatureFinder_gpu2(img3_gpu, gpu::GpuMat(), keypoints_gpu_r, descriptors_gpu_4);
		//将特征点下载回cpu，便于画图使用  
		FeatureFinder_gpu2.downloadKeypoints(keypoints_gpu_l, keypoints_l);
		FeatureFinder_gpu2.downloadKeypoints(keypoints_gpu_lm, keypoints_lm);
		FeatureFinder_gpu2.downloadKeypoints(keypoints_gpu_mr, keypoints_mr);
		FeatureFinder_gpu2.downloadKeypoints(keypoints_gpu_r, keypoints_r);
		//使用gpu提供的BruteForceMatcher进行特征点匹配 
		gpu::BruteForceMatcher_GPU< L2<float> > matcher_lk;
		std::vector<std::vector<cv::DMatch>>matches1;
		matcher_lk.knnMatch(descriptors_gpu_1, descriptors_gpu_2, matches1, 2);
		//从图2到图1的k最近邻
		std::vector<std::vector<cv::DMatch>>matches2;
		matcher_lk.knnMatch(descriptors_gpu_2, descriptors_gpu_1, matches2, 2);
		//3.移除NN比率大于阈值的匹配
		//从图一到图二的k最近邻
		std::vector<std::vector<cv::DMatch>>matches3;
		matcher_lk.knnMatch(descriptors_gpu_3, descriptors_gpu_4, matches3, 2);
		//从图2到图1的k最近邻
		std::vector<std::vector<cv::DMatch>>matches4;
		matcher_lk.knnMatch(descriptors_gpu_4, descriptors_gpu_3, matches4, 2);


		//detector->detect(img1_roi, keypoints_l);
		//detector->detect(img2_roi_lm, keypoints_lm);
		//detector->detect(img2_roi_mr, keypoints_mr);
		//detector->detect(img3_roi, keypoints_r);
		////1b.取描述子
		//cv::Mat descriptors1, descriptors2, descriptors3, descriptors4;
		//extractor->compute(img1_roi, keypoints_l, descriptors1);
		//extractor->compute(img2_roi_lm, keypoints_lm, descriptors2);
		//extractor->compute(img2_roi_mr, keypoints_mr, descriptors3);
		//extractor->compute(img3_roi, keypoints_r, descriptors4);
		////2.配两幅图的描述子
		////创建匹配器
		//cv::BruteForceMatcher<cv::L2<float>>matcher;
		////FlannBasedMatcher matcher;  
		////从图一到图二的k最近邻
		//std::vector<std::vector<cv::DMatch>>matches1;
		//matcher.knnMatch(descriptors1, descriptors2, matches1, 2);
		////从图2到图1的k最近邻
		//std::vector<std::vector<cv::DMatch>>matches2;
		//matcher.knnMatch(descriptors2, descriptors1, matches2, 2);
		////3.移除NN比率大于阈值的匹配
		////从图一到图二的k最近邻
		//std::vector<std::vector<cv::DMatch>>matches3;
		//matcher.knnMatch(descriptors3, descriptors4, matches3, 2);
		////从图2到图1的k最近邻
		//std::vector<std::vector<cv::DMatch>>matches4;
		//matcher.knnMatch(descriptors4, descriptors3, matches4, 2);




		//清理图1到图2的匹配
		int removed = ratioTest(matches1);
		//cout << "图1到图2被的匹配点有" << (matches1.size() - removed) << "个" << endl;
		removed = ratioTest(matches2);
		//cout << "图2到图1被的匹配点有" << (matches2.size() - removed) << "个" << endl;
		removed = ratioTest(matches3);
		removed = ratioTest(matches4);
		//4.移除非对称匹配
		std::vector<cv::DMatch>symMatches1, symMatches2;
	//	cv::Mat out1, out2;
		symmetryTest(matches1, matches2, symMatches1);
		cout << "run2 l-m有效的匹配点有" << symMatches1.size() << "个" << endl;
	/*	cv::drawMatches(
		image_l, keypoints_l,
		image_m, keypoints_lm,
		symMatches1,
		out1,
		cv::Scalar(255, 255, 255)
		);
		imshow("out1", out1);*/
		symmetryTest(matches3, matches4, symMatches2);
		cout << "run2 m-r有效的匹配点有" << symMatches2.size() << "个" << endl;
		/*cv::drawMatches(
		image_m, keypoints_mr,
		image_r, keypoints_r,
		symMatches2,
		out2,
		cv::Scalar(255, 255, 255)
		);
		imshow("out2", out2);
 		waitKey(0);*/
		//5.得到H矩阵
		if ((symMatches1.size()>10) && (symMatches2.size()>10))
		{
			H_l = ransacTest(symMatches1, keypoints_l, keypoints_lm,0);
			H_r = ransacTest(symMatches2, keypoints_mr, keypoints_r,0);
			if (image_num == 612)
			{
				FileStorage fs1("H21.xml", FileStorage::WRITE);
				FileStorage fs2("H61.xml", FileStorage::WRITE);
				H61 = H_l;
				H21 = H_r;
				fs2 << "H6_1" << H61;
				fs1 << "H2_1" << H21;
				fs1.release();
				fs2.release();
			}
			else
			{
				FileStorage fs3("H34.xml", FileStorage::WRITE);
				FileStorage fs4("H54.xml", FileStorage::WRITE);
				H34 = H_l;
				H54 = H_r;
				fs3 << "H3_4" << H34;
				fs4 << "H5_4" << H54;
				fs3.release();
				fs4.release();
			}
			refineH = false;
		}
		else
		{
			//Down_flag = false;
			cout << "图  " << image_num << "  匹配对过少，提高图像质量或拼接角度重新匹配" << endl;
			refineH = false;
		//	cout << "refineH" << refineH << endl;
			H_r.release();
			H_l.release();
			return false;

		}

		flag_rh = false;
	}
	//	cout << "H_r= " << H_r << endl;
	//	cout << "H_l= " << H_l << endl;

	//if (Down_flag&&(!H_r.empty()))
	if ((!H_r.empty()) && (!H_l.empty()))
	{
		CalcFourCorner(H_r,image_r);
		int mRows = image_r.rows;
		if (image_m.rows > image_r.rows)
		{
			mRows = image_m.rows;
		}
		if (min(rightTop.x, rightBottom.x) > 800)
		{
			cout << "min(rightTop.x, rightBottom.x)超过800像素" << endl;
			if (image_num == 612)
			{
				H21.release();
				H61.release();
			}
			else
			{
				H34.release();
				H54.release();
			}
			return false;
		}
		//stitchedImage = cv::Mat::zeros(cv::min(rightTop.x, rightBottom.x), mRows, CV_8UC3);
		gpu::GpuMat image_gpu_r(image_r);
		gpu::GpuMat stitchedImage_gpu;
		cv::gpu::warpPerspective(image_gpu_r, stitchedImage_gpu, H_r, cv::Size(image_r.cols + image_m.cols, mRows));//stichedImage行列颠倒 将img2透视变换放到stitchedImage左侧
		stitchedImage_gpu.download(stitchedImage);
		xformed_proc = stitchedImage.clone();
		//	imshow("stitche", stitchedImage);

		int Lcols = cv::min(leftTop.x, leftBottom.x);
		if (cv::min(leftTop.x, leftBottom.x)>image_m.cols)
		{
			Lcols = image_m.cols;
		}

		img1_ROI = image_m(cv::Rect(0, 0, Lcols, xformed_proc.rows));//img1上感兴趣的区域，为一个矩形 Rect（左上点x,y，x方向的长，y方向的宽）
		xformed_ROI = xformed_proc(cv::Rect(0, 0, Lcols, xformed_proc.rows));
		cv::addWeighted(img1_ROI, 1, xformed_ROI, 0, 0, xformed_proc(cv::Rect(0, 0, Lcols, xformed_proc.rows)));//img1_ROI*1+xformed_ROI*0->xformed_proc的矩形区域内。
		//	imshow("1111111", xformed_proc);
		//	waitKey(0);
		int nl = xformed_proc.rows;
		int nc1 = image_m.cols;
		int start = Lcols;
		//    double processWidth=img1.cols*img1.channels()-start;
		double processWidth = image_m.cols - start;
		nc1 = nc1 - processWidth / 2;
		double alpha = 1;
		//    if(xformed_proc.isContinuous()){
		//        nc*=nl;
		//        nl=1;
		//    }
		for (int j = 0; j < nl; j++)
		{
			uchar *data = xformed_proc.ptr<uchar>(j);
			uchar *data_img1 = image_m.ptr<uchar>(j);
			uchar *data_xformed = xformed_proc.ptr<uchar>(j);
			for (int i = start; i < nc1; i++)
			{
				//如果遇到图像xformed中无像素的黑点，则完全拷贝图1中的数据 
				//if ((data_xformed[i] <150) && (data_xformed[i + 1] <150) && (data_xformed[i + 2] <150))
				if (i < start + 50)
					alpha = 1;
				else
				{
					alpha = (nc1 - start - (i - start)) / (double)(nc1 - start - 50);
					alpha = alpha*alpha*alpha;
				}
				data[i * 3] = data_img1[i * 3] * alpha + data_xformed[i * 3] * (1 - alpha);
				data[i * 3 + 1] = data_img1[i * 3 + 1] * alpha + data_xformed[i * 3 + 1] * (1 - alpha);
				data[i * 3 + 2] = data_img1[i * 3 + 2] * alpha + data_xformed[i * 3 + 2] * (1 - alpha);
			}
		}
		stitchedImage.release();
			img2 = xformed_proc;
			CalcFourCorner(H_l, xformed_proc);
			mRows = xformed_proc.rows;
			if (image_l.rows > xformed_proc.rows)
			{
				mRows = image_l.rows;
			}
		/*	if (min(rightTop.x, rightBottom.x) > 2600)
			{
				cout << "min(rightTop.x, rightBottom.x)超过2600像素" << endl;
				if (image_num == 612)
				{
					H21.release();
					H61.release();
				}
				else
				{
					H34.release();
					H54.release();
				}
				return false;
			}*/
			//stitchedImage = cv::Mat::zeros(cv::min(rightTop.x, rightBottom.x), mRows, CV_8UC3);
			gpu::GpuMat xformed_proc_gpu(xformed_proc);
			gpu::GpuMat StitchedImage_gpu;
			cv::gpu::warpPerspective(xformed_proc_gpu, StitchedImage_gpu, H_l, cv::Size(xformed_proc.cols + image_l.cols, mRows));//stichedImage行列颠倒 将img2透视变换放到stitchedImage左侧
			StitchedImage_gpu.download(stitchedImage);
			xformed_proc = stitchedImage.clone();
		    //	imshow("stitche", stitchedImage);
			//	imshow("imagel", image_l);
			//	waitKey(0);
			 Lcols = cv::min(leftTop.x, leftBottom.x);
			if (cv::min(leftTop.x, leftBottom.x)>image_l.cols)
			{
				Lcols = image_l.cols;
			}

			img1_ROI = image_l(cv::Rect(0, 0, Lcols, xformed_proc.rows));//img1上感兴趣的区域，为一个矩形 Rect（左上点x,y，x方向的长，y方向的宽）
			xformed_ROI = xformed_proc(cv::Rect(0, 0, Lcols, xformed_proc.rows));
			cv::addWeighted(img1_ROI, 1, xformed_ROI, 0, 0, xformed_proc(cv::Rect(0, 0, Lcols, xformed_proc.rows)));//img1_ROI*1+xformed_ROI*0->xformed_proc的矩形区域内。
			//	imshow("1111111", xformed_proc);
			//	imshow("img1_ROI", img1_ROI);
			//	waitKey(0);
			 nl = xformed_proc.rows;
		//	 waitKey(0);
			 nc1 = image_l.cols;
			 start = Lcols;
			//    double processWidth=img1.cols*img1.channels()-start;
			 processWidth = image_l.cols - start;
			nc1 = nc1 - processWidth / 2;
			 alpha = 1;
			//    if(xformed_proc.isContinuous()){
			//        nc*=nl;
			//        nl=1;
			//    }
			for (int j = 0; j < nl; j++)
			{
				uchar *data = xformed_proc.ptr<uchar>(j);
				uchar *data_img1 = image_l.ptr<uchar>(j);
				uchar *data_xformed = xformed_proc.ptr<uchar>(j);
				for (int i = start; i < nc1; i++)
				{
					//如果遇到图像xformed中无像素的黑点，则完全拷贝图1中的数据 
					//if ((data_xformed[i] <150) && (data_xformed[i + 1] <150) && (data_xformed[i + 2] <150))
					if (i < start + 50)
						alpha = 1;
					else
					{
						alpha = (nc1 - start - (i - start)) / (double)(nc1 - start - 50);
						alpha = alpha*alpha*alpha;
					}
					data[i * 3] = data_img1[i * 3] * alpha + data_xformed[i * 3] * (1 - alpha);
					data[i * 3 + 1] = data_img1[i * 3 + 1] * alpha + data_xformed[i * 3 + 1] * (1 - alpha);
					data[i * 3 + 2] = data_img1[i * 3 + 2] * alpha + data_xformed[i * 3 + 2] * (1 - alpha);
				}
			}
		//	cv::imshow("stitching2", xformed_proc);
		//	waitKey(0);
			out = xformed_proc;
			xformed_proc.release();
	}

	return true;
}

void ImageDeal::writeNextFrame(cv::Mat& frame)
{
	writer << frame;
}

int ImageDeal::GetVideoRate()
{
	//double rate = capture1.get(CV_CAP_PROP_FPS);//获取帧率
	//cout << "The inputrate is" << rate << endl;
	return 0;//int)rate;
}
Scalar getMSSIM_GPU(const Mat& i1, const Mat& i2)
{
	const float C1 = 6.5025f, C2 = 58.5225f;
	/***************************** INITS **********************************/
	gpu::GpuMat gI1, gI2, gs1, t1, t2;

	gI1.upload(i1);
	gI2.upload(i2);

	gI1.convertTo(t1, CV_MAKE_TYPE(CV_32F, gI1.channels()));
	gI2.convertTo(t2, CV_MAKE_TYPE(CV_32F, gI2.channels()));

	vector<gpu::GpuMat> vI1, vI2;
	gpu::split(t1, vI1);
	gpu::split(t2, vI2);
	Scalar mssim;

	for (int i = 0; i < gI1.channels(); ++i)
	{
		gpu::GpuMat I2_2, I1_2, I1_I2;

		gpu::multiply(vI2[i], vI2[i], I2_2);        // I2^2
		gpu::multiply(vI1[i], vI1[i], I1_2);        // I1^2
		gpu::multiply(vI1[i], vI2[i], I1_I2);       // I1 * I2

		/*************************** END INITS **********************************/
		gpu::GpuMat mu1, mu2;   // PRELIMINARY COMPUTING
		gpu::GaussianBlur(vI1[i], mu1, Size(11, 11), 1.5);
		gpu::GaussianBlur(vI2[i], mu2, Size(11, 11), 1.5);

		gpu::GpuMat mu1_2, mu2_2, mu1_mu2;
		gpu::multiply(mu1, mu1, mu1_2);
		gpu::multiply(mu2, mu2, mu2_2);
		gpu::multiply(mu1, mu2, mu1_mu2);

		gpu::GpuMat sigma1_2, sigma2_2, sigma12;

		gpu::GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
		gpu::subtract(sigma1_2, mu1_2, sigma1_2);
		//sigma1_2 = sigma1_2 - mu1_2;

		gpu::GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
		//sigma2_2 = sigma2_2 - mu2_2;
		gpu::subtract(sigma2_2, mu2_2, sigma2_2);
		gpu::GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
		//(Mat)sigma12 = (Mat)sigma12 - (Mat)mu1_mu2;
		gpu::subtract(sigma12, mu1_mu2, sigma12);
		//sigma12 = sigma12 - mu1_mu2
		Mat Mu1_mu2, Sigma12, Mu1_2, Mu2_2, Sigma1_2, Sigma2_2;
		mu1_mu2.download(Mu1_mu2);
		sigma12.download(Sigma12);
		mu1_2.download(Mu1_2);
		mu2_2.download(Mu2_2);
		sigma1_2.download(Sigma1_2);
		sigma2_2.download(Sigma2_2);
		///////////////////////////////// FORMULA ////////////////////////////////
		Mat t1, t2, t3;

		t1 = 2 * Mu1_mu2 + C1;
		t2 = 2 * Sigma12 + C2;
		multiply(t1, t2, t3);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

		t1 = Mu1_2 + Mu2_2 + C1;
		t2 = Sigma1_2 + Sigma2_2 + C2;
		multiply(t1, t2, t1);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

		//	gpu::GpuMat ssim_map_gpu;
		cv::Mat ssim_map;
		divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
		//	ssim_map_gpu.download(ssim_map);
		Scalar s = sum(ssim_map);
		mssim.val[i] = s.val[0] / (ssim_map.rows * ssim_map.cols);

	}
	return mssim;
}