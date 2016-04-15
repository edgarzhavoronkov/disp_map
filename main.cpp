#include <stdio.h>
#include <cmath>
#include <vector>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::StereoSGBM sgbm;

std::vector<double> xs;

void callback(const sensor_msgs::ImageConstPtr& left
    , const sensor_msgs::ImageConstPtr& right
    , const sensor_msgs::ImageConstPtr& depth)
{
    printf("Caught left image\n");
    printf("Caught right image \n");
    printf("Caught depth map\n");

    cv::Mat leftImage = cv_bridge::toCvShare(left, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat rightImage = cv_bridge::toCvShare(right, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat depthImage = cv_bridge::toCvShare(depth)->image;
    cv::Mat greyLeftImage, greyRightImage;
    cv::Mat disp, disp8;

    cv::cvtColor(leftImage, greyLeftImage, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightImage, greyRightImage, cv::COLOR_BGR2GRAY);

    sgbm(greyLeftImage, greyRightImage, disp);
    cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);
    printf("Computed disp_map\n\n");

    cv::Mat diff = disp8 - depthImage;
    printf("Difference is %f\n", cv::sum(diff)[0]);
    xs.push_back(cv::sum(diff)[0]);
}

int main(int argc, char** argv )
{
    ros::init(argc, argv, "disp_map");
    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> leftImageSub(nh, "/wide_stereo/left/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> rightImageSub(nh, "/wide_stereo/right/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depthMapSub(nh, "/camera/depth/image_raw", 1);


    sgbm.SADWindowSize = 5;
    sgbm.numberOfDisparities = 192;
    sgbm.preFilterCap = 4;
    sgbm.minDisparity = -64;
    sgbm.uniquenessRatio = 1;
    sgbm.speckleWindowSize = 150;
    sgbm.speckleRange = 2;
    sgbm.disp12MaxDiff = 10;
    sgbm.fullDP = false;
    sgbm.P1 = 600;
    sgbm.P2 = 2400;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> syncPolicy;

    message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10), leftImageSub, rightImageSub, depthMapSub);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3));

    ros::spin();

    //should execute after we read all messages
    double mean = 0;
    for (size_t i = 0; i < xs.size(); ++i)
    {
        mean += xs[i];
    }
    mean /= xs.size();

    double sd = 0;
    for (size_t i = 0; i < xs.size(); ++i)
    {
        sd += ((xs[i] - mean) * (xs[i] - mean));
    }
    sd = sqrt(sd / xs.size());

    printf("Standard deviation is: %f\n", sd);

    return 0;
}
