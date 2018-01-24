 #include "Calibration.h"
#include "ImageDeal.h"
#include "gloabal_value.h"

ImageDeal processor;
#define Gettime 1


class Mat_kalman
{
  private:
    cv::Mat Data_mid;
  public:
    cv::Mat R;
    cv::Mat Q;
    cv::Mat x_last;
    cv::Mat x_mid;
    cv::Mat x_now;
    cv::Mat p_last;
    cv::Mat p_mid;
    cv::Mat p_now;
    cv::Mat kg;
    cv::Mat ADData;
    void Kalman_init(int row, int col);
    cv::Mat Kalman_Filter(cv::Mat Data );
};

//void Img_predeal();
/*void Feature_Detect(); 
void Compute_Homography(); 
void ThreadINit();*/
//void Img_Fusion();
//void Segnet_detection();
Mat_kalman kalman;
int main(int argc, char** argv) 
{ /********Ros*******/
  ros::init(argc, argv, "Pa_stitching");
  ros::start();
  ros::NodeHandle node;
  ros::Publisher chatter_pub = node.advertise<std_msgs::String>("chatter", 1000); 
  /************************Project**************************/
  img_left=imread("/media/wds/26EED644EED60BC5/Users/59760/Desktop/1111论文准备/拼接素材/APAP-garden/campus_01.jpg");
  img_right=imread("/media/wds/26EED644EED60BC5/Users/59760/Desktop/1111论文准备/拼接素材/APAP-garden/campus_02.jpg");
#if Gettime
  clock_t Mainstart = clock();
#endif
  resize(img_left,img_left,Size(640,480));
  resize(img_right,img_right,Size(640,480));
 // kalman.Kalman_init(3,3);
  //     bbc=500;
       //caffe
//    showout.create(img_left.rows, img_left.cols * 2, CV_8UC3);   
//    cout << "---------caffe_init-----------" << endl;
//     
//   /*******************GPU******************************/
//   cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
//   std::cout << "Input The Quality Threshould of Image with 0~1!:";
//   cin >> num ;
//   if ((0 < num) && (num < 1.0))
//     {
// 	    std::cout << " Stitching system was initially successful!" << endl;
// 	    processor.Expect_quality = num;
//     }
//   else
//     {
// 	    std::cout << "The Quality Threshould of Image ：0.8" << endl;
// 	    processor.Expect_quality = 0.8;
//     }
//   std::cout << "Loading parameter..." << endl; 
//       
//          
// //          img_left_gpu.upload(img_left);
// // 	 CV_Assert(!img_left_gpu.empty());
// // 	 img_right_gpu.upload(img_right);
// // 	 CV_Assert(!img_right_gpu.empty());
//  std::cout << "Gpu loaded success!" << endl; 
//   /*********Thread********/
//     ThreadINit();
//     waitKey(20);
//     unsigned int procs = thread::hardware_concurrency();
   // cout<<"procs number is "<<procs<<endl;
   // std::thread Img_Fusion_thread(Img_Fusion);
   // Img_Fusion_thread.detach();
   // std::thread Feature_Detect_thread(Feature_Detect);
   // Feature_Detect_thread.detach();
    //std::thread  Homography_thread(Compute_Homography);
   // Homography_thread.detach();
   // std::thread  Segnet_thread(Segnet_detection);
  //  Segnet_thread.detach();
    cout<<"Thread open success"<<endl;  
   /*******************main******************************/ 
 #if Gettime
   clock_t MainendTime = clock();
      cout<<"main start is "<<static_cast<double>(MainendTime - Mainstart) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
#endif
 while (ros::ok())
  {  
//     if(!img_left.empty()&&!img_right.empty())
//    {
//      F_Feature_start=true;
//     // F_Segnet_start=true;
//    }
//     else
//     {
//       cout<<"frame read fail"<<endl;
//        F_Feature_start=false;
//       // F_Segnet_start=false;
//     }
       clock_t startTime = clock();
       cv::Mat frame_left,frame_right;	
       cv::resize(img_left, frame_left, cv::Size(480,360));
       cv::resize(img_right, frame_right, cv::Size(480,360));
// 	
     //   cv::imshow("frame", frame);
        //cv::waitKey(0);
        std::vector<Prediction_seg> predictions_seg_left = seger_left.Classify(frame_left);
	std::vector<Prediction_seg> predictions_seg_right = seger_right.Classify(frame_right);
	
        cv::Mat segnet_left(frame_left.size(), CV_8UC3, cv::Scalar(0,0,0));
        cv::Mat segnet_right(frame_right.size(), CV_8UC3, cv::Scalar(0,0,0));
	
	
	
	for (int i = 0; i < 360; ++i)
        {
            uchar* segnet_ptr_left = segnet_left.ptr<uchar>(i);
            uchar* segnet_ptr_right = segnet_right.ptr<uchar>(i);
	    for (int j = 0; j < 480; ++j)
            {
                segnet_ptr_left[j*3+0] = predictions_seg_left[i*480+j].second;
                segnet_ptr_left[j*3+1] = predictions_seg_left[i*480+j].second;
                segnet_ptr_left[j*3+2] = predictions_seg_left[i*480+j].second;
                
		segnet_ptr_right[j*3+0] = predictions_seg_right[i*480+j].second;
                segnet_ptr_right[j*3+1] = predictions_seg_right[i*480+j].second;
                segnet_ptr_right[j*3+2] = predictions_seg_right[i*480+j].second;
	      
	    }
        }
        // recover
   //  cv::LUT(segnet_left, color, segnet_left);
   //  cv::LUT(segnet_right, color, segnet_right);
     
     
     imwrite("/media/wds/26EED644EED60BC5/Users/59760/Desktop/1111论文准备/拼接素材/APAP-garden/segnet_left.jpg",segnet_left);
     imwrite("/media/wds/26EED644EED60BC5/Users/59760/Desktop/1111论文准备/拼接素材/APAP-garden/segnet_right.jpg",segnet_right);
     imshow("segnet",segnet_left);
     waitKey(0);
     break;
     
#if Gettime
   clock_t segnetTime1 = clock();
    cout<<"pure segnetTime is "<<static_cast<double>(segnetTime1 - startTime) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
#endif
    cv::resize(segnet_left, segnet_left_result, img_left.size());
    cv::resize(segnet_right, segnet_right_result, img_right.size());
 
#if Gettime
   clock_t segnetTime = clock();
    cout<<"segnetTime is "<<static_cast<double>(segnetTime - startTime) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
#endif
//     
//        Mat imgrawROI;
//         imgrawROI = showout(Rect(0, 0, img_left.cols,img_left.rows));
//         segnet_left_result.copyTo(imgrawROI);
//         imgrawROI = showout(Rect(img_right.cols, 0, img_right.cols,img_right.rows));
//         segnet_right_result.copyTo(imgrawROI);
//         imshow("IMG_show",showout);
// 	
// 	imwrite("segnet_left.jpg",segnet_left_result);
// 	imwrite("segnet_right.jpg",segnet_right_result);
	
//        waitKey(5);
    if(F_Feature_update==true)
    {
      F_HomoG_start=true;
      F_Feature_update=false;
   //  cout<<"The matches number is "<<matches.size()<<endl;
     Feat_fail_count=0;
    }
    else
    {
      if(Feat_fail_count>3)
      {
	F_HomoG_start=false;
	F_Img_start=false;
	cout<<"Feature is not enough!"<<endl;
      }
    }
   if( F_HomoG_update==true)
   {
     F_HomoG_update=false;
     F_Img_start=true;
     
     //cout<<"The Homography="<<H<<endl;
   }
   if(F_Img__update==true)
   {
     F_Img__update=false;
     imshow("result",out);  
     imwrite("reslt.jpg",out);
   //  waitKey(5);
   }
      clock_t endTime = clock();
      sprintf(msgss,"System time %f ms.\r\n", static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC * 1000.);
      std_msgs::String msg;  
      std::stringstream ss;  
      ss << msgss;  
      msg.data = ss.str();  
      ROS_INFO("%s", msg.data.c_str());  
      chatter_pub.publish(msg);  
      ros::spinOnce();  
  }
  
 // Img_predeal_thread.join(); 
  // Feature_Detect_thread.join();
   destroyAllWindows();
   
}
void ThreadINit()
{
  //  namedWindow("change",1);
#if ShowMatcher
   namedWindow("Th1_imgMatcher",1); //不加windows多线程使用imshow会出错;
   namedWindow("Th3_out",1);
   ;
   
#endif
  // namedWindow("Th3_xformed_proc",1);
   
   namedWindow("Th1_imgMatcher",1); //不加windows多线程使用imshow会出错;
   namedWindow("result",1);
   namedWindow("IMG_show",1);
     F_Feature_start=false;
     F_Img_start=false;
     F_Segnet_start=false;
     F_HomoG_start=false;
     F_Feature_update=false;
     F_Img__update=false;
     F_HomoG_update=false;
     F_Segnet_update=false;
     
     kalman_count=1000;
     Feat_fail_count=0;
}
void Feature_Detect()
{
 cout<<"creat the feature_Detect thread !"<<endl;
 
 
 
 bool matcher_flage=0 ;
 
 
  while(ros::ok())
  { 
    
     if(F_Feature_start)
    {
      if(mtx.try_lock())
      {
       img_left.copyTo(Th1_img_left);
       img_right.copyTo(Th1_img_right);
       mtx.unlock();
      }
      else
      {
//	cout<<"F_Feature_start mtx.try_lock()"<<endl;
	continue;
	
      }
      clock_t Th1_startTime = clock();
      cvtColor(Th1_img_left, Th1_img_left_gray, CV_BGR2GRAY);
      cvtColor(Th1_img_right, Th1_img_right_gray, CV_BGR2GRAY);
      //1a.测Surf特征
     Th1_img_left_gpu.upload(Th1_img_left_gray);
     Th1_img_right_gpu.upload(Th1_img_right_gray);
     
     Th1_FeatureFinder_gpu.extended = false;
      //计算特征点和特征描述子  
     Th1_FeatureFinder_gpu(Th1_img_left_gpu, gpu::GpuMat(), Th1_keypoints_gpu_1, Th1_descriptors_gpu_1);
     Th1_FeatureFinder_gpu(Th1_img_right_gpu, gpu::GpuMat(), Th1_keypoints_gpu_2, Th1_descriptors_gpu_2);

   //   cout << "FOUND " << Th1_keypoints_gpu_1.cols << " keypoints on first image" << endl;
   //   cout << "FOUND " << Th1_keypoints_gpu_2.cols << " keypoints on second image" << endl;
      
      //将特征点下载回cpu，便于画图使用  
      Th1_FeatureFinder_gpu.downloadKeypoints(Th1_keypoints_gpu_1, Th1_keypoints1);
      Th1_FeatureFinder_gpu.downloadKeypoints(Th1_keypoints_gpu_2, Th1_keypoints2);
      
      clock_t Th2_endTime3 = clock();
      cout<<"featuredetect time "<<static_cast<double>(Th2_endTime3- Th1_startTime) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
      
      
      
      //使用gpu提供的BruteForceMatcher进行特征点匹配 
      std::vector< vector<DMatch> >matches1, matches2;
      //gpu::FlannBasedMatcher<L2<float>>matcher_lk;
      gpu::BruteForceMatcher_GPU< L2<float> > matcher_lk;
      matcher_lk.knnMatch(Th1_descriptors_gpu_1, Th1_descriptors_gpu_2, matches1, 2);
      matcher_lk.knnMatch(Th1_descriptors_gpu_2, Th1_descriptors_gpu_1, matches2, 2);
      int removed = processor.ratioTest(matches1);
      
       cout << "matches1 count:" << matches1.size() << endl;
     // cout << "图1到图2被的匹配点有" << (removed) << "个" << endl;
      removed = processor.ratioTest(matches2);
      
    //  cout << "matches2 count:" << matches2.size() << endl;
      
     // cout << "图2到图1被的匹配点有" << (removed) << "个" << endl;
      std::vector<cv::DMatch>Th1_matches,Th1_cmatches;
      
      processor.symmetryTest(matches1, matches2, Th1_cmatches);
      
      
      
      
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
			else 
			{
				Th1_matches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx, (*matchIterator1)[0].distance));
				break;
			}
		}
	}
      
      
      
    //  cout << "matche_best count:" << Th1_matches.size() << endl;
#if ShowMatcher
      Mat imgMatcher;
      cv::drawMatches(
      Th1_img_left, Th1_keypoints1,
      Th1_img_right, Th1_keypoints2,
      Th1_matches,
      imgMatcher,
      cv::Scalar::all(-1),
      cv::Scalar::all(-1),
      vector<char>(),
      DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS	
      );
      imshow("Th1_imgMatcher", imgMatcher);
      if(matcher_flage==0)
      {
      imwrite("/home/wds/catkin_ws/src/pa_stitching/data/SPHP-campus/ori-Matcher.jpg",imgMatcher);
        matcher_flage=1;
      }
#endif
      
       
   
       if((char)waitKey(3)=='s') 
       break;
      if(Th1_matches.size()>20)
      {
	mtx.lock();
	matches=Th1_matches;
	cmatches=Th1_cmatches;
	keypoints1=Th1_keypoints1;
	keypoints2=Th1_keypoints2;
	mtx.unlock();
	
	F_Feature_start=false;
	F_Feature_update=true;
	Feat_fail_count=0;
      }
      else
      {
	cout<<"特征过少只有"<<Th1_matches.size()<<"组"<<endl;
	F_Feature_update=false;
	Feat_fail_count++;
      }
      clock_t Th1_endTime = clock();
      cout<<"Feature match time "<<static_cast<double>(Th1_endTime - Th2_endTime3) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
    }
    else
    { 
      std::chrono::milliseconds dura(50);
     //Feature_Detect_thread.sleep_for(dura);
      std::this_thread::sleep_for( dura );
    //  std::cout << "Feature_Detect_thread sleep 50 ms\n";
    }
  }
  cout<<"The feature_Detect thread is destroyed!"<<endl;
}

void Compute_Homography()
{
   cout<<"creat the Homography thread !"<<endl;
#if Gettime
       int numm=0;
       
#endif
  while(ros::ok())
  {
    if(F_HomoG_start)
    {
      clock_t Th2_startTime = clock();
      std::vector<cv::KeyPoint>Th2_keypoints1;
      std::vector<cv::KeyPoint>Th2_keypoints2;
      std::vector<cv::DMatch>Th2_matches;

      if(mtx.try_lock())
      {
	Th2_keypoints1 = keypoints1;
	Th2_keypoints2 = keypoints2;
	Th2_matches = matches;
	mtx.unlock();
      }
      else 
	continue;
      
      Th2_H = processor.ransacTest(Th2_matches, Th2_keypoints1, Th2_keypoints2, 0);
      Th2_H_per = Th2_H;
      Th2_H = kalman.Kalman_Filter(Th2_H);
      if(kalman_count>0)
      {
	kalman_count--;
	//if((kalman_count%100)==0)
	cout<<"kalman_count="<<kalman_count<<endl;
	F_HomoG_update=false;
	continue;
	
      }
      else
      {
	mtx.lock();
	Th2_H.copyTo(H);
	Th2_H_per.copyTo(H_per);
	mtx.unlock();
	F_HomoG_update=true;
      }
       clock_t Th2_endTime = clock();
#if Gettime
        numm++;
	if(num==20)
	  numm=0;
       if(numm==2)
#endif
       cout<<"computer homograhy time "<<static_cast<double>(Th2_endTime - Th2_startTime) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
    }
    else
    {
      std::chrono::milliseconds Th2_dura( 20 );
      std::this_thread::sleep_for( Th2_dura );
   //   std::cout << "Homography thread sleep 20 ms\n";
    }
  }
  cout<<"The Homography thread is destroyed!"<<endl;
}

void Img_Fusion()
{
  cout<<"creat the Img_Fusion thread !"<<endl;
  
  
  bool writeflag=0;
  
  int iiiii=0;
  double numma=0,numse=0;
  
  while(ros::ok())
  {
     if(F_HomoG_start)
    {
      clock_t Th2_startTime = clock();
      std::vector<cv::KeyPoint>Th2_keypoints1;
      std::vector<cv::KeyPoint>Th2_keypoints2;
      std::vector<cv::DMatch>Th2_matches;
      Mat Th2_segnet_left,Th2_segnet_right;
      Mat Th2_image_left,Th2_image_right;
      if(mtx.try_lock())
      {
	segnet_left_result.copyTo(Th2_segnet_left);
	segnet_right_result.copyTo(Th2_segnet_right);
	mtx.unlock();
      }
      else 
	continue;
       if(mtx.try_lock())
      {
	Th2_image_left=img_left.clone();
	Th2_image_right=img_right.clone();
	Th2_keypoints1 = keypoints1;
	Th2_keypoints2 = keypoints2;
	Th2_matches = matches;
	mtx.unlock();
      }
       else 
	continue;
        cout<<"orig-matches is "<<Th2_matches.size()<<endl;
     //保存匹配对序号  
    vector<int> queryIdxs( Th2_matches.size() ), trainIdxs( Th2_matches.size() );  
    for( size_t i = 0; i < Th2_matches.size(); i++ )  
    {  
        queryIdxs[i] = Th2_matches[i].queryIdx;  
        trainIdxs[i] = Th2_matches[i].trainIdx;  
    }     
  
   
    vector<Point2f> points1; 
    KeyPoint::convert(Th2_keypoints1, points1, queryIdxs);  
    vector<Point2f> points2; 
    KeyPoint::convert(Th2_keypoints2, points2, trainIdxs);  
    int ransacReprojThreshold =5;  //拒绝阈值  
     
    
        std::vector<Point2f> obj_global;//1

	std::vector<Point2f> scene_global;//2



	std::vector<Point2f> obj_top;//1

	std::vector<Point2f> scene_top;//2



	std::vector<Point2f> obj_down;//1

	std::vector<Point2f> scene_down;//2



	for (int i = 0; i < Th2_matches.size(); i++)

	{

		//-- Get the keypoints from the good matches

		Point2f kPoint0 = Th2_keypoints1[Th2_matches[i].queryIdx].pt;

		Point2f kPoint1 = Th2_keypoints2[Th2_matches[i].trainIdx].pt;



		//obj_global.push_back(kPoint0);

		//scene_global.push_back(kPoint1);



		Point2f midPoint;

		midPoint.y = (kPoint0.y + kPoint1.y) / 2;



		if (midPoint.y < (Th2_image_left.rows) / 2)

		{

			obj_top.push_back(kPoint0);

			scene_top.push_back(kPoint1);

		}

		else

		{

			obj_down.push_back(kPoint0);

			scene_down.push_back(kPoint1);

		}





	}   
	 Mat Htop,Hdown;   //变换矩阵 
    Htop = findHomography(obj_top, scene_top, CV_RANSAC, ransacReprojThreshold ); 
    Hdown = findHomography(obj_down, scene_down, CV_RANSAC, ransacReprojThreshold ); 
    
    vector<char> matchesMask( Th2_matches.size(), 0 );    
    Mat points1t,points2t;  
    
    perspectiveTransform(Mat(points1), points1t, Htop);
    perspectiveTransform(Mat(points1), points1t, Htop);
    
    int inline_num=0;
    for( size_t i1 = 0; i1 < points1.size(); i1++ )  //保存‘内点’  
    {  
        if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= ransacReprojThreshold ) //给内点做标记  
        {  
            matchesMask[i1] = 1;  
	    inline_num++;
        }     
    }  
//     Mat match_img2;   //滤除‘外点’后  
//       
//     drawMatches(Th2_image_left,Th2_keypoints1,
// 		Th2_image_right,Th2_keypoints2,
// 		Th2_matches,match_img2,
// 		cv::Scalar::all(-1),
// 		Scalar::all(-1),
//       matchesMask,DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  
//   
//     //画出目标位置  
//     std::vector<Point2f> obj_corners(4);  
//     obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( Th2_image_left.cols, 0 );  
//     obj_corners[2] = cvPoint( Th2_image_left.cols, Th2_image_left.rows ); obj_corners[3] = cvPoint( 0, Th2_image_left.rows );  
// //     std::vector<Point2f> scene_corners(4);  
// //     perspectiveTransform( obj_corners, scene_corners, H12);  
// //     line( match_img2, scene_corners[0] + Point2f(static_cast<float>(Th2_image_left.cols), 0),   
// //         scene_corners[1] + Point2f(static_cast<float>(Th2_image_left.cols), 0),Scalar(0,0,255),2);  
// //     line( match_img2, scene_corners[1] + Point2f(static_cast<float>(Th2_image_left.cols), 0),   
// //         scene_corners[2] + Point2f(static_cast<float>(Th2_image_left.cols), 0),Scalar(0,0,255),2);  
// //     line( match_img2, scene_corners[2] + Point2f(static_cast<float>(Th2_image_left.cols), 0),   
// //         scene_corners[3] + Point2f(static_cast<float>(Th2_image_left.cols), 0),Scalar(0,0,255),2);  
// //     line( match_img2, scene_corners[3] + Point2f(static_cast<float>(Th2_image_left.cols), 0),  
// //         scene_corners[0] + Point2f(static_cast<float>(Th2_image_left.cols), 0),Scalar(0,0,255),2);  
//   
//    // imshow("滤除误匹配后",match_img2);  
//    // waitKey(10); 
    if(iiiii<10)
    {
      numma=numma+inline_num/double(Th2_matches.size()*9);
    }
   
     cout<<"inline-matches is "<<inline_num<<endl;
    
       clock_t Th2_endTime3 = clock();
      cout<<"inline-matches time "<<static_cast<double>(Th2_endTime3- Th2_startTime) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
      
//       if(writeflag==0)
//       {
// 	imwrite("/home/wds/catkin_ws/src/pa_stitching/data/SPHP-campus/matcher_RANSAC.jpg",match_img2);
// 	
//       }
	
       
       
       
       
      //add segnet 
      int removed=0;
   //  cout<<"Matcher size ="<<Th2_matches.size()<<endl;
      
       std::vector<cv::DMatch> symMatches;
//    /*   for (std::vector<cv::DMatch>::const_iterator it = Th2_matches.begin(); it != Th2_matches.end(); ++it)
// 	{   //左图像关键点
//            cv::Point2f points1, points2;
// 		int x1 = Th2_keypoints1[it->queryIdx].pt.x;
// 		int y1 = Th2_keypoints1[it->queryIdx].pt.y;
// 		points1=cv::Point2f(x1, y1);
// 		//右图像关键点
// 		int x2 = Th2_keypoints2[it->trainIdx].pt.x;
// 		int y2 = Th2_keypoints2[it->trainIdx].pt.y;
// 		points2=cv::Point2f(x2, y2);
// 		if(segnet_left_result.at<Vec3d>(y1,x1)!=segnet_right_result.at<Vec3d>(y2,x2))
// 		{
// 		  Th2_matches.erase(it);
// 	         removed++;
// 		}
// 		else
// 	        symMatches.push_back(cv::DMatch((*it).queryIdx, (*it).trainIdx, (*it).distance));
// 	}
//	*/
      vector <cv::DMatch>::iterator Iter; 
        if(mtx.try_lock())
      {
      for(Iter = cmatches.begin(); Iter != cmatches.end(); Iter++) 
      {
	int i1=(*Iter).queryIdx;
	int i2=(*Iter).trainIdx;
	const Point2f kp1 = Th2_keypoints1[i1].pt, kp2 = Th2_keypoints2[i2].pt;
	if(segnet_left_result.at<Vec3d>(kp1.y,kp1.x)[0]==segnet_right_result.at<Vec3d>(kp2.y,kp2.x)[0]&&segnet_left_result.at<Vec3d>(kp1.y,kp1.x)[1]==segnet_right_result.at<Vec3d>(kp2.y,kp2.x)[1]&&segnet_left_result.at<Vec3d>(kp1.y,kp1.x)[2]==segnet_right_result.at<Vec3d>(kp2.y,kp2.x)[2])
	{
	// Th2_matches.erase(Iter);
	  symMatches.push_back(cv::DMatch((*Iter).queryIdx, (*Iter).trainIdx, (*Iter).distance));
	  
	}
	else
	  removed++;
      }
      mtx.unlock();
      }
       else 
	continue;
       
//       Mat imgMatcher;
//       cv::drawMatches(
//       Th2_image_left, Th1_keypoints1,
//       Th2_image_right, Th1_keypoints2,
//       symMatches,
//       imgMatcher,
//       cv::Scalar::all(-1),
//       cv::Scalar::all(-1),
//       vector<char>(),
//       DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS	
//       );
//       imshow("ori-segnet-matches", imgMatcher);
       
        clock_t Th2_endTime4 = clock();
      cout<<"segnet-matches time "<<static_cast<double>(Th2_endTime4- Th2_endTime3) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
   /*   
      if(writeflag==0)
      {
	imwrite("/home/wds/catkin_ws/src/pa_stitching/data/SPHP-campus/ori_segnet_matcher.jpg",match_img2);
	
      }
      waitKey(5);*/
      
    //   cout<<"segnet remove num="<<removed<<endl;
    //   cout<<"segnet symMatches num="<<symMatches.size()<<endl;
       
    cout<<"segnet-matches is "<<symMatches.size()<<endl; 
       
         //保存匹配对序号  
    vector<int> queryIdxs1(symMatches.size() ), trainIdxs1( symMatches.size() );  
    for( size_t i = 0; i < symMatches.size(); i++ )  
    {  
        queryIdxs1[i] = symMatches[i].queryIdx;  
        trainIdxs1[i] = symMatches[i].trainIdx;  
    }     
  
    vector<Point2f> points11; KeyPoint::convert(Th2_keypoints1, points11, queryIdxs1);  
    vector<Point2f> points21; KeyPoint::convert(Th2_keypoints2, points21, trainIdxs1);  
  
   Mat H12;
    H12 = findHomography( Mat(points11), Mat(points21), CV_RANSAC, ransacReprojThreshold );  
    vector<char> matchesMask1( symMatches.size(), 0 );    
    perspectiveTransform(Mat(points11), points1t, H12);
    inline_num=0;
    for( size_t i1 = 0; i1 < points11.size(); i1++ )  //保存‘内点’  
    {  
        if( norm(points21[i1] - points1t.at<Point2f>((int)i1,0)) <= ransacReprojThreshold ) //给内点做标记  
        {  
            matchesMask1[i1] = 1;  
	    inline_num++;
        }     
    }  
   
//     drawMatches(Th2_image_left,Th2_keypoints1,
// 		Th2_image_right,Th2_keypoints2,
// 		symMatches,match_img2,
// 		cv::Scalar::all(-1),
// 		Scalar::all(-1),
//       matchesMask1,
//       DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); 
//     //画出目标位置
//     /*  
//     std::vector<Point2f> obj_corners(4);  
//     obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( Th2_image_left.cols, 0 );  
//     obj_corners[2] = cvPoint( Th2_image_left.cols, Th2_image_left.rows ); obj_corners[3] = cvPoint( 0, Th2_image_left.rows );  */
// //     std::vector<Point2f> scene_corners(4);  
// //     perspectiveTransform( obj_corners, scene_corners, H12);  
// //     line( match_img2, scene_corners[0] + Point2f(static_cast<float>(Th2_image_left.cols), 0),   
// //         scene_corners[1] + Point2f(static_cast<float>(Th2_image_left.cols), 0),Scalar(0,0,255),2);  
// //     line( match_img2, scene_corners[1] + Point2f(static_cast<float>(Th2_image_left.cols), 0),   
// //         scene_corners[2] + Point2f(static_cast<float>(Th2_image_left.cols), 0),Scalar(0,0,255),2);  
// //     line( match_img2, scene_corners[2] + Point2f(static_cast<float>(Th2_image_left.cols), 0),   
// //         scene_corners[3] + Point2f(static_cast<float>(Th2_image_left.cols), 0),Scalar(0,0,255),2);  
// //     line( match_img2, scene_corners[3] + Point2f(static_cast<float>(Th2_image_left.cols), 0),  
// //         scene_corners[0] + Point2f(static_cast<float>(Th2_image_left.cols), 0),Scalar(0,0,255),2);  
//   
//     imshow("segnet滤除误匹配后",match_img2);  
//      if(writeflag==0)
//       {
// 	imwrite("/home/wds/catkin_ws/src/pa_stitching/data/SPHP-campus/segnet_matcher_RANSAC.jpg",match_img2);
// 	writeflag=1;
// 	
//       }
//     
//     waitKey(10); 
    
     cout<<"inline-segnet-matches is "<<inline_num<<endl;
     
     
     if(iiiii<10)
    {
      numse=numse+inline_num/double(symMatches.size()*9);
    }
    else if(iiiii==10)
    {
      iiiii=0;
      cout<< "matche rate is                                               "<<numma<<endl;
      numma=0;
     cout<< "seget matche rate is                                               "<<numse<<endl;
      numse=0;
    }
    iiiii++;
     
       clock_t Th2_endTime5 = clock();
      cout<<"inline-segnet-matches time "<<static_cast<double>(Th2_endTime5- Th2_endTime4) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
      
     
#if Gettime
    clock_t Th2_endTime1 = clock();
      cout<<"feature matcher time "<<static_cast<double>(Th2_endTime1 - Th2_startTime) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
#endif
    if(symMatches.size()<20)
      continue;
      Th2_H = processor.ransacTest(symMatches, Th2_keypoints1, Th2_keypoints2, 0);
      Th2_H_per = Th2_H;
      Th2_H = kalman.Kalman_Filter(Th2_H);
      
       cout<<"segnet-matches is "<<symMatches.size()<<endl;
       
      if(kalman_count>0)
      {
	kalman_count--;
	//if((kalman_count%100)==0)
	cout<<"kalman_count="<<kalman_count<<endl;
	F_HomoG_update=false;
	continue;
	
      }
      else
      {
	mtx.lock();
	Th2_H.copyTo(H);
	Th2_H_per.copyTo(H_per);
	mtx.unlock();
	F_HomoG_update=true;
      }
       clock_t Th2_endTime = clock();
#if Gettime
    clock_t Th2_endTime2 = clock();
      cout<<"computer homograhy time "<<static_cast<double>(Th2_endTime2 - Th2_endTime1) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
#endif
   //   cout<<"computer homograhy time "<<static_cast<double>(Th2_endTime - Th2_startTime) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
    }
    
    if(F_Img_start)
    {
      clock_t Th3_startTime = clock();
      if(mtx.try_lock())
      {
      Th3_img_right=img_right.clone();
      Th3_img_left=img_left.clone();
      Th3_H=H.clone();  
       mtx.unlock();
      }
      else continue;
      processor.CalcFourCorner(Th3_H,Th3_img_right);
      int mRows = Th3_img_right.rows;
      if (Th3_img_left.rows > Th3_img_right.rows)
      {
	      mRows = Th3_img_left.rows;
      }
      gpu::GpuMat G_image2(Th3_img_right);
      gpu::GpuMat G_stitchedImage;
      
      cv::gpu::warpPerspective(G_image2, G_stitchedImage, Th3_H, cv::Size(Th3_img_left.cols + Th3_img_right.cols, mRows));
      G_stitchedImage.download(Th3_stitchedImage);
    //  imshow("Th3_stitchedImage", Th3_stitchedImage);
       Th3_xformed_proc = Th3_stitchedImage.clone();
       int Lcols = Th3_min( processor.leftTop.x,  processor.leftBottom.x);
       int Rcols =Th3_max( processor.leftTop.x,  processor.leftBottom.x);
       int RRcols=Th3_max( processor.rightTop.x,  processor.rightBottom.x);
       
  //    cout<<"leftTopx= "<<processor.leftTop.x<<endl;
   //    cout<<"leftTopy= "<<processor.leftBottom.x<<endl;
    //   cout<<"Lcols= "<<Lcols<<endl;
     //  cout<<"RRcols= "<<RRcols<<endl;
       if (Lcols > Th3_img_left.cols)
       {
	      Lcols = Th3_img_left.cols-1;
       }
      else if( Lcols < 0||RRcols<Th3_img_left.cols)
      {      F_Img__update=false;
	     cout << "超出拼接范围" << endl;
	     Mat Th3_failimg=Mat(mRows,Th3_img_left.cols + Th3_img_right.cols,Th3_img_left.type());
	     Mat Th3_Roi_img1;
	     Th3_Roi_img1=Th3_failimg(cv::Rect(0,0,Th3_img_left.cols,mRows));
	     Th3_img_left.copyTo(Th3_Roi_img1);
	     Mat Th3_Roi_img2;
	     Th3_Roi_img2=Th3_failimg(cv::Rect(Th3_img_left.cols,0,Th3_img_right.cols,mRows));
	     Th3_img_right.copyTo(Th3_Roi_img2);  
	     mtx.lock();
	     imshow("result",Th3_failimg);
             mtx.unlock();	     
	     continue;
      }
      
   //    cout<<"Lcols2= "<<Lcols<<endl;
      Mat img1_ROI;
      img1_ROI = Th3_img_left(cv::Rect(0, 0, Lcols, Th3_xformed_proc.rows));//img1上感兴趣的区域，为一个矩形 Rect（左上点x,y，x方向的长，y方向的宽）
      Mat xformed_ROI;
      xformed_ROI = Th3_xformed_proc(cv::Rect(0, 0, Lcols, Th3_xformed_proc.rows));
      cv::addWeighted(img1_ROI, 1, xformed_ROI, 0, 0, Th3_xformed_proc(cv::Rect(0, 0, Lcols, Th3_xformed_proc.rows)));//img1_ROI*1+xformed_ROI*0->xformed_proc的矩形区域内。
    //  imshow("Th3_xformed_proc", Th3_xformed_proc);
     Mat Th3_out=Th3_xformed_proc.clone();
      int nl = Th3_xformed_proc.rows;
      int nc1 = Th3_img_left.cols;
      int start = Lcols;
      int start_cols=0;
      if(nc1>=Rcols)
      {
	start_cols=Rcols-start;
      }   
 //     cout<<"start= "<<start<<endl;
  //     cout<<"Th3_img_left.cols= "<<nc1<<endl;
      //    double processWidth=Th3_img_left.cols*Th3_img_left.channels()-start;
     // double processWidth = Th3_img_left.cols - start;
   //   nc1 = nc1 - processWidth / 4;
      double alpha = 1;
      for (int j = 0; j < nl; j++)
      {
	      uchar *data = Th3_out.ptr<uchar>(j);
	      uchar *data_img1 = Th3_img_left.ptr<uchar>(j);
	      uchar *data_xformed = Th3_xformed_proc.ptr<uchar>(j);
	      for (int i = start; i < nc1; i++)
	      {
 		      if (i < start + start_cols)
 			      alpha = 1;
 		      else
		      {
		      
			alpha = (nc1 -i ) / (double)(nc1 - start - start_cols);
			alpha = alpha;//*alpha*alpha;
		      }
		      data[i * 3] = data_img1[i * 3] * alpha + data_xformed[i * 3] * (1 - alpha);
		      data[i * 3 + 1] = data_img1[i * 3 + 1] * alpha + data_xformed[i * 3 + 1] * (1 - alpha);
		      data[i * 3 + 2] = data_img1[i * 3 + 2] * alpha + data_xformed[i * 3 + 2] * (1 - alpha);
	      }
      }
     //  if((char)waitKey(20)=='s')
#if ShowMatcher
     cv::imshow("Th3_out",Th3_out); 
#endif
     mtx.lock(); 
     Th3_out.copyTo(out);
     mtx.unlock();
     clock_t Th3_endTime = clock();
     cout<<"Img_Fusion time "<<static_cast<double>(Th3_endTime - Th3_startTime) / CLOCKS_PER_SEC * 1000.<<"ms.\r\n"<<endl;
     
     if(static_cast<double>(Th3_endTime - Th3_startTime)<50)
     waitKey(50-static_cast<double>(Th3_endTime - Th3_startTime));
     F_Img__update=true;
    }
    else
    {
    //  std::chrono::milliseconds Th3_dura( 40 );
    //  std::this_thread::sleep_for( Th3_dura );
    //  std::cout << "Img_Fusion thread sleep 40 ms\n";
    }
  }
  cout<<"The Img_Fusion thread is destroyed!"<<endl;
}/*
void Segnet_detection()
{
  cout<<"creat the Segnet_Detect thread !"<<endl;
  while(ros::ok())
  {
   
    if(F_Segnet_start)
    {
      
    }
     else
    {
      std::chrono::milliseconds Th4_dura( 50 );
      std::this_thread::sleep_for( Th4_dura );
      std::cout << "Segnet_Detect thread sleep 20 ms\n";
    }
    
  }
  cout<<"The Segnet_Detect thread is destroyed!"<<endl;
}*/
void Mat_kalman::Kalman_init(int row, int col)
{
    R=Mat::ones(row,col,CV_64FC1);
    Q=Mat::ones(row,col,CV_64FC1);
    x_last=Mat::eye(row,col,CV_64FC1);
    x_mid.create(row,col,CV_64FC1);
    x_now.create(row,col,CV_64FC1);
    p_last=Mat::eye(row,col,CV_64FC1);
    p_mid.create(row,col,CV_64FC1);
    p_now.create(row,col,CV_64FC1);
    kg.create(row,col,CV_64FC1);
    ADData.create(row,col,CV_64FC1);
   
    R = 50000*R;
    Q = 1*Q;
    x_last=1*x_last;
    p_last=1*p_last;
    cout<<"The Kalman inited successfully!"<<endl;
    cout<<"R="<<R<<endl;
    cout<<"Q="<<Q<<endl;
}
cv::Mat Mat_kalman::Kalman_Filter(cv::Mat Data)
{
    x_mid=x_last.clone(); //x_last=x(k-1|k-1),x_mid=x(k|k-1)
    p_mid=p_last+Q; //p_mid=p(k|k-1),p_last=p(k-1|k-1),Q=噪声
    kg=p_mid/(p_mid+R); //kg为kalman filter，R为噪声
    x_now=x_mid+kg.mul(Data-x_mid);//估计出的最优值
    p_now=(1-kg).mul(p_mid);//最优值对应的covariance
    p_last = p_now; //更新covariance值
    x_last = x_now; //更新系统状态值
    return x_now;		
}