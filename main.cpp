#include <stdio.h>
#include <cmath>
#include <vector>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float64.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::StereoSGBM sgbm;

std::vector<long double> xs;

inline long double sqr(long double x)
{
    return x * x;
}

void callback(const sensor_msgs::ImageConstPtr& left
    , const sensor_msgs::ImageConstPtr& right
    , const sensor_msgs::ImageConstPtr& depth
    , const sensor_msgs::CameraInfoConstPtr& leftCameraInfo
    , const sensor_msgs::CameraInfoConstPtr& rightCameraInfo)
{
    printf("Caught left image\n");
    printf("Caught right image \n");
    printf("Caught depth map\n");
    printf("Left camera dimensions are %d x %d\n", leftCameraInfo->width, leftCameraInfo->height);
    printf("Right camera dimensions are %d x %d\n", rightCameraInfo->width, rightCameraInfo->height);

    cv::Mat leftImage = cv_bridge::toCvShare(left, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat rightImage = cv_bridge::toCvShare(right, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat depthImage = cv_bridge::toCvShare(depth)->image;
    cv::Mat greyLeftImage, greyRightImage;
    cv::Mat disp, disp8;

    cv::cvtColor(leftImage, greyLeftImage, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightImage, greyRightImage, cv::COLOR_BGR2GRAY);

    //TODO: calibarate cameras - assume cameras are calibrated

    //rectification

    std::vector<std::vector<double> > leftCamMat = {
        {leftCameraInfo->K[0], leftCameraInfo->K[1], leftCameraInfo->K[2]},
        {leftCameraInfo->K[3], leftCameraInfo->K[4], leftCameraInfo->K[5]},
        {leftCameraInfo->K[6], leftCameraInfo->K[7], leftCameraInfo->K[8]}
    };

    std::vector<std::vector<double> > rightCamMat = {
        {rightCameraInfo->K[0], rightCameraInfo->K[1], rightCameraInfo->K[2]},
        {rightCameraInfo->K[3], rightCameraInfo->K[4], rightCameraInfo->K[5]},
        {rightCameraInfo->K[6], rightCameraInfo->K[7], rightCameraInfo->K[8]}
    };

    std::vector<std::vector<double> > R = {
        {leftCameraInfo->R[0], leftCameraInfo->R[1], leftCameraInfo->R[2]},
        {leftCameraInfo->R[3], leftCameraInfo->R[4], leftCameraInfo->R[5]},
        {leftCameraInfo->R[6], leftCameraInfo->R[7], leftCameraInfo->R[8]}
    };

    std::vector<double> leftCamD = leftCameraInfo->D;
    std::vector<double> rightCamD = rightCameraInfo->D;

    std::vector<double> T = {
        leftCameraInfo->P[3],
        leftCameraInfo->P[7],
        leftCameraInfo->P[11]
    };

    std::vector<std::vector<double>> Q(4, std::vector<double>(4));

    cv::stereoRectify(
        leftCamMat,
        leftCamD,
        rightCamMat,
        rightCamD,
        leftImage.size(),
        R,
        T,
        cv::noArray(),
        cv::noArray(),
        cv::noArray(),
        cv::noArray(),
        cv::noArray()
    );

    //computing disparity

    sgbm(greyLeftImage, greyRightImage, disp);
    cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_16UC1);
    printf("Computed disparity map\n");

    //computing depth from disparity

    cv::Mat depthMap = cv::Mat::zeros(disp8.size(), CV_16UC1);
    cv::reprojectImageTo3D(disp8, depthMap, Q);

    printf("Computed depth map\n");

    cv::Mat diff = depthMap - depthImage;
    printf("Difference is %f\n\n", cv::sum(diff)[0]);
    xs.push_back(cv::sum(diff)[0]);
}

int main(int argc, char** argv )
{
    ros::init(argc, argv, "disp_map");
    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> leftImageSub(nh, "/wide_stereo/left/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo> leftCameraInfoSub(nh, "/wide_stereo/left/camera_info", 1);

    message_filters::Subscriber<sensor_msgs::Image> rightImageSub(nh, "/wide_stereo/right/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo> rightCameraInfoSub(nh, "/wide_stereo/right/camera_info", 1);

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

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> syncPolicy;

    message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10), leftImageSub, rightImageSub, depthMapSub, leftCameraInfoSub, rightCameraInfoSub);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5));

    ros::spin();

    //should execute after we read all messages
    //theese formulaes are recursive so we can avoid overflow
    long double mean = 0;
    for (size_t i = 0; i < xs.size(); ++i)
    {
        mean = ((mean * i) + xs[i])/(i + 1);
    }
    long double sd = 0;
    for (size_t i = 0; i < xs.size(); ++i)
    {
        sd = sqrt( ((sqr(sd) * i) + sqr(xs[i] - mean))/(i + 1) );
    }

    printf("Mean is: %Lf\n", mean);
    printf("Standard deviation is: %Lf\n", sd);

    return 0;
}
