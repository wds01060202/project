#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//ROS
#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point32.h>

#include "segnet.h"
#include "classification.h"

#define CAM_INPUT 0 //相机输入 
#define IMG_INPUT 1
#define Write_IMG 0

using namespace cv;
using namespace std;

Mat imgraw;
Mat imgrawShow;
Mat imShow;
Mat imIPMraw;
Mat houghspace_show1;
Mat save_Img;
int counter = 50;//起始帧

Classifier_SEG seger;
string colorfile = "color.png";
cv::Mat color = cv::imread(colorfile, 1);

////IPM设置/////
Mat src;
char name[50];

int main(int argc, char ** argv)
{
    ros::init(argc, argv, "slam");
     ros::start();


        ros::NodeHandle node;

        // Set up publishers
        image_transport::ImageTransport it(node);
        image_transport::Publisher im_publisher;
        im_publisher = it.advertise("drivingimgs",1);

    while (ros::ok())
    {
       clock_t startTime = clock();
       
      
      
        imShow.create(360 * 2, 480 * 2, CV_8UC3);   
	
        cout << "--------------------" << endl;
        sprintf(name, "RawData/R_%ld.jpg", counter);
        src = imread(name);
        namedWindow("IMG_show",0);
	
        Mat imshowsrc,colormat;
        //transpose(src, imshowsrc);
        //transpose(imshowsrc, imshowsrc);
        flip(src, imshowsrc, 0);
        flip(imshowsrc, imshowsrc, 1);
	Mat Rawsrc;
	cv::resize(imshowsrc, Rawsrc, cv::Size(480,360));
      //  imshow("src", imshowsrc);

        cv::Mat frame;
        //segnet
        cv::resize(imshowsrc, frame, cv::Size(480,360));
        if(frame.size().width<=0)continue;
        cv::Mat copy_frame = frame.clone();



        // Prediction
        cv::resize(frame, frame, cv::Size(480,360));
     //   cv::imshow("frame", frame);
        //cv::waitKey(0);
        std::vector<Prediction_seg> predictions_seg = seger.Classify(frame);
        cv::Mat segnet(frame.size(), CV_8UC3, cv::Scalar(0,0,0));
        for (int i = 0; i < 360; ++i)
        {
            uchar* segnet_ptr = segnet.ptr<uchar>(i);
            for (int j = 0; j < 480; ++j)
            {
                segnet_ptr[j*3+0] = predictions_seg[i*480+j].second;
                segnet_ptr[j*3+1] = predictions_seg[i*480+j].second;
                segnet_ptr[j*3+2] = predictions_seg[i*480+j].second;
            }
        }

        // recover
        cv::resize(segnet, segnet, copy_frame.size());
        cv::LUT(segnet, color, segnet);
     //   cv::imshow("segnet", segnet);
        cv::Mat result;
        cv::addWeighted(segnet, 0.7, copy_frame, 1.0, 0, result);
       // cv::imshow("result", result);
	
	//im_publisher.publish(result);
	
	Mat imgrawROI;
	 imgrawROI = imShow(Rect(0, 0, 480,360));
	 Rawsrc.copyTo(imgrawROI);
	  imgrawROI = imShow(Rect(480, 0, 480,360));
	 frame.copyTo(imgrawROI);
	 imgrawROI = imShow(Rect(0, 360, 480,360));
	 segnet.copyTo(imgrawROI);
	  imgrawROI = imShow(Rect(480, 360, 480,360));
	 result.copyTo(imgrawROI);
	 imshow("IMG_show",imShow);
	 
        clock_t endTime = clock();
        printf("System time %f ms.\r\n", static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC * 1000.);
        printf("Counter: %ld\r\n", counter);
	
#if Write_IMG
        /*sprintf(name,"Result//d_%ld.jpg",counter);
        imwrite(name,imShow);

        sprintf(name,"RawData//R_%ld.jpg",counter);
        imwrite(name,save_Img);*/

#endif
        printf("\r\n\r\n");
        counter++;
	
        if (cv::waitKey(10) == 27) break;
        ros::spinOnce();
    }
    cv::destroyAllWindows();


}
