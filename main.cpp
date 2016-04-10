#include <stdio.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


cv::Mat leftImage;
cv::Mat rightImage;
cv::Mat depthMap;

const int MSG_COUNT = 33152;

void handleLeftPicReceived(const sensor_msgs::ImageConstPtr& msg)
{
    printf("Caught left image\n");
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
        leftImage = cv_ptr->image;
        printf("Converted left image to CV image!\n");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s\n", e.what());
        return;
    }
}

void handleRightPicReceived(const sensor_msgs::ImageConstPtr& msg)
{
    printf("Caught right image \n");
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
        rightImage = cv_ptr->image;
        printf("Converted right image to CV image!\n");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s\n", e.what());
        return;
    }
}

void handleEtalonPicReceived(const sensor_msgs::ImageConstPtr& msg)
{
    printf("Caught depth map\n");
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
        depthMap = cv_ptr->image;
        printf("Converted etalon image to CV image!\n");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s\n", e.what());
        return;
    }
}


int main(int argc, char** argv )
{
    ros::init(argc, argv, "disp_map");
    ros::NodeHandle nh;

    //subscribe and get images from publisher
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber lPicSubscriber = it.subscribe("/wide_stereo/left/image_raw", 1000, handleLeftPicReceived);
    image_transport::Subscriber rPicSubscriber = it.subscribe("/wide_stereo/right/image_raw", 1000, handleRightPicReceived);
    image_transport::Subscriber etalonPicSubsriber = it.subscribe("/camera/depth/image_raw", 1000, handleEtalonPicReceived);

    cv::Mat greyLeftImage;
    cv::Mat greyRightImage;
    cv::Mat disp, disp8;

    if (leftImage.data && rightImage.data)
    {
        cv::cvtColor(leftImage, greyLeftImage, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightImage, greyRightImage, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::StereoSGBM> sgbm = cv::createStereoSGBM(-64, 192, 15, 600, 2400, 10, 4, 1, 150, 2);
        sgbm->setDisp12MaxDiff(10);

        sgbm->compute(greyLeftImage, greyRightImage, disp);
        cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    else
    {
        printf("No data in any of images!\n");
        return -1;
    }

    ros::spin();

    return 0;
}
