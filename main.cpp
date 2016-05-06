#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>

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

cv::StereoBM sgbm(cv::StereoBM::BASIC_PRESET, 80,5);
// cv::StereoSGBM sgbm;

const int MMS_IN_METER = 1000;

cv::Mat leftCamMap1, leftCamMap2;
cv::Mat rightCamMap1, rightCamMap2;
cv::Mat Q;

std::vector<long double> xs;

inline long double sqr(long double x)
{
    return x * x;
}

void onMouseMove(int event, int x, int y, int flags, void* userdata)
{
    //hope no memory leaks are in here
    cv::Mat pic = cv::Mat::zeros(250, 250, CV_8UC3);
    cv::Mat* depthPicPtr = (cv::Mat*)userdata;
    std :: string text = "depth = " + std :: to_string(depthPicPtr->at<float>(x, y));
    cv::putText(pic, text, cv::Point(50,50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, 8, false);
    cv::imshow("Text", pic);
    //std :: cout << "x = " << x << " y = " << y  << std :: endl;
}

inline static void initSGBM()
{
    // sgbm.SADWindowSize = 5;
    // sgbm.numberOfDisparities = 192;
    // sgbm.preFilterCap = 4;
    // sgbm.minDisparity = -64;
    // sgbm.uniquenessRatio = 1;
    // sgbm.speckleWindowSize = 150;
    // sgbm.speckleRange = 2;
    // sgbm.disp12MaxDiff = 10;
    // sgbm.fullDP = false;
    // sgbm.P1 = 600;
    // sgbm.P2 = 2400;

    sgbm.state->SADWindowSize = 9;
    sgbm.state->numberOfDisparities = 112;
    sgbm.state->preFilterSize = 5;
    sgbm.state->preFilterCap = 31;
    sgbm.state->minDisparity = 0;
    sgbm.state->textureThreshold = 10;
    sgbm.state->uniquenessRatio = 15;
    sgbm.state->speckleWindowSize = 100;
    sgbm.state->speckleRange = 32;
    sgbm.state->disp12MaxDiff = 1;
}

inline cv::Mat getMatrix(std::vector<std::vector<double>>& vec)
{
    cv::Mat res(vec.size(), vec[0].size(), CV_64FC1);
    for (size_t i = 0; i < vec.size(); i++)
    {
        for (size_t j = 0; j < vec[i].size(); j++)
        {
            res.at<double>(i, j) = vec[i][j];
        }
    }
    return res;
}

void cameraInfoHandler(const sensor_msgs::ImageConstPtr& left
        , const sensor_msgs::ImageConstPtr& right
        , const sensor_msgs::ImageConstPtr& depth
        , const sensor_msgs::CameraInfoConstPtr& leftCameraInfo
        , const sensor_msgs::CameraInfoConstPtr& rightCameraInfo)
{
    //TODO: calibarate cameras - assume cameras are calibrated

    //rectification

    static std::vector<std::vector<double> > leftCamMatData = {
        {leftCameraInfo->K[0], leftCameraInfo->K[1], leftCameraInfo->K[2]},
        {leftCameraInfo->K[3], leftCameraInfo->K[4], leftCameraInfo->K[5]},
        {leftCameraInfo->K[6], leftCameraInfo->K[7], leftCameraInfo->K[8]}
    };

    static std::vector<std::vector<double> > rightCamMatData = {
        {rightCameraInfo->K[0], rightCameraInfo->K[1], rightCameraInfo->K[2]},
        {rightCameraInfo->K[3], rightCameraInfo->K[4], rightCameraInfo->K[5]},
        {rightCameraInfo->K[6], rightCameraInfo->K[7], rightCameraInfo->K[8]}
    };

    static std::vector<std::vector<double> > rData = {
        {leftCameraInfo->R[0], leftCameraInfo->R[1], leftCameraInfo->R[2]},
        {leftCameraInfo->R[3], leftCameraInfo->R[4], leftCameraInfo->R[5]},
        {leftCameraInfo->R[6], leftCameraInfo->R[7], leftCameraInfo->R[8]}
    };

    static std::vector<std::vector<double> > leftCamDData = {
        { leftCameraInfo->D[0] },
        { leftCameraInfo->D[1] },
        { leftCameraInfo->D[2] },
        { leftCameraInfo->D[3] },
        { leftCameraInfo->D[4] }
    };

    static std::vector<std::vector<double> > rightCamDData = {
        { rightCameraInfo->D[0] },
        { rightCameraInfo->D[1] },
        { rightCameraInfo->D[2] },
        { rightCameraInfo->D[3] },
        { rightCameraInfo->D[4] }
    };

    static std::vector<std::vector<double> > tData = {
        { rightCameraInfo->P[3] },
        { rightCameraInfo->P[7] },
        { rightCameraInfo->P[11] }
    };

    static cv::Mat leftCamMat = getMatrix(leftCamMatData);
    static cv::Mat rightCamMat = getMatrix(rightCamMatData);
    static cv::Mat leftCamD = getMatrix(leftCamDData);
    static cv::Mat rightCamD = getMatrix(rightCamDData);
    static cv::Mat R = getMatrix(rData);
    static cv::Mat T = getMatrix(tData);
    static cv::Mat R1, R2, P1, P2;

    if (Q.empty())
    {
        cv::stereoRectify(
            leftCamMat,
            leftCamD,
            rightCamMat,
            rightCamD,
            cv::Size(leftCameraInfo->width, leftCameraInfo->height),
            R,
            T,
            R1,
            R2,
            P1,
            P2,
            Q
        );
        Q.convertTo(Q, CV_32FC1);
        Q.at<float>(3, 3) = -Q.at<float>(3, 3);
        std::cout << "Computed rectification matrices!" << std::endl;
    }

    if (leftCamMap1.empty() && leftCamMap2.empty())
    {
        cv::initUndistortRectifyMap(
            leftCamMat,
            leftCamD,
            R1,
            leftCamMat,
            cv::Size(leftCameraInfo->width, leftCameraInfo->height),
            CV_32FC1,
            leftCamMap1,
            leftCamMap2
        );
        std::cout << "Computed rectification map for left camera!" << std::endl;
    }


    if (rightCamMap1.empty() && rightCamMap2.empty())
    {
        cv::initUndistortRectifyMap(
            rightCamMat,
            rightCamD,
            R2,
            rightCamMat,
            cv::Size(rightCameraInfo->width, rightCameraInfo->height),
            CV_32FC1,
            rightCamMap1,
            rightCamMap2
        );
        std::cout << "Computed rectification map for right camera!" << std::endl;
    }

}


void picHandler(const sensor_msgs::ImageConstPtr& left
    , const sensor_msgs::ImageConstPtr& right
    , const sensor_msgs::ImageConstPtr& depth
    , const sensor_msgs::CameraInfoConstPtr& leftCameraInfo
    , const sensor_msgs::CameraInfoConstPtr& rightCameraInfo)
{
    printf("Caught left image\n");
    printf("Caught right image \n");
    printf("Caught depth map\n");

    cv::Mat leftImage = cv_bridge::toCvShare(left, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat rightImage = cv_bridge::toCvShare(right, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat depthImage = cv_bridge::toCvShare(depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;

    cv::Mat mappedLeftImage, mappedRightImage;
    cv::remap(
        leftImage,
        mappedLeftImage,
        leftCamMap1,
        leftCamMap2,
        cv::INTER_LINEAR
    );

    std::cout << "Remapped left image!" << std::endl;

    cv::remap(
        rightImage,
        mappedRightImage,
        rightCamMap1,
        rightCamMap2,
        cv::INTER_LINEAR
    );

    std::cout << "Remapped right image!" << std::endl;

    cv::Mat greyLeftImage, greyRightImage;

    cv::cvtColor(mappedLeftImage, greyLeftImage, cv::COLOR_BGR2GRAY);
    std::cout << "Converted to grey left image!" << std::endl;

    cv::cvtColor(mappedRightImage, greyRightImage, cv::COLOR_BGR2GRAY);
    std::cout << "Converted to grey right image!" << std::endl;

    //computing disparity
    cv::Mat disp;
    sgbm(greyLeftImage, greyRightImage, disp, CV_32F);

    //normalization - not needed. It needs to be performed only for rendering

    // cv::Mat disp8 = cv::Mat::zeros(disp.size(), CV_8U);
    //cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);

    printf("Computed disparity map\n");

    //computing depth from disparity

    cv::Mat depthMap3D = cv::Mat::zeros(depthImage.size(), CV_32FC1);;
    cv::reprojectImageTo3D(disp, depthMap3D, Q, true);


    cv::Point2f leftCamPrincipalPoint(leftCameraInfo->K[2], leftCameraInfo->K[5]);
    cv::Point2f rightCamPrincipalPoint(rightCameraInfo->K[2], rightCameraInfo->K[5]);

    cv::Point2f midPoint = (leftCamPrincipalPoint + rightCamPrincipalPoint) * .5;
    cv::Point3f midPoint3D(midPoint.x, midPoint.y, 0);

    cv::Mat depthMap = cv::Mat::zeros(depthMap3D.size(), CV_32FC1);

    for (size_t i = 0; i < depthMap.rows; ++i)
    {
        for (size_t j = 0; j < depthMap.cols; ++j)
        {
            cv::Point3f pixel =  depthMap3D.at<cv::Point3f>(i, j);
            depthMap.at<float>(i, j) = cv::norm(pixel - midPoint3D);
            // depthMap.at<float>(i, j) = depthMap3D.at<cv::Point3f>(i, j).z;
        }
    }

    printf("Computed depth map\n");

    // cv::imshow("left", leftImage);
    // cv::imshow("right", rightImage);
    // cv::imshow("disp", disp);
    cv::imshow("expected", depthImage);
    cv::imshow("actual", depthMap);
    cv::setMouseCallback("actual", onMouseMove, (void*)&depthMap);
    cv::setMouseCallback("expected", onMouseMove, (void*)&depthImage);

    cv::waitKey(30);

    cv::Mat diff = depthMap - depthImage;

    long double sum = 0;

    for (size_t i = 0; i < diff.rows; ++i)
    {
        for (size_t j = 0; j < diff.cols; ++j)
        {
            if (std::isfinite(diff.at<float>(i, j)))
            {
                sum += diff.at<float>(i, j);
            }
        }
    }

    printf("Total difference is %Lf\n\n", sum);
    xs.push_back(sum);
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

    initSGBM();

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                            sensor_msgs::Image,
                                                            sensor_msgs::Image,
                                                            sensor_msgs::CameraInfo,
                                                            sensor_msgs::CameraInfo> syncPolicy;
    message_filters::Synchronizer<syncPolicy> synchronizer(syncPolicy(10), leftImageSub, rightImageSub, depthMapSub, leftCameraInfoSub, rightCameraInfoSub);
    //first callback must execute before second - it is very important
    synchronizer.registerCallback(boost::bind(&cameraInfoHandler, _1, _2, _3, _4, _5));
    synchronizer.registerCallback(boost::bind(&picHandler, _1, _2, _3, _4, _5));


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
