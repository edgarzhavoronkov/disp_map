#include <stdio.h>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main(int argc, char** argv )
{
    ros::init(argc, argv, "disp_map");
    ros::NodeHandle nh;
    if ( argc != 3 )
    {
        printf("usage: DispMap <LeftImagePath> <RightImagePath>\n");
        return -1;
    }

    Mat leftImage, greyLeftImage;
    Mat rightImage, greyRightImage;
    Mat disp, disp8;
    leftImage = imread(argv[1], 1);
    rightImage = imread(argv[2], 1);

    if (leftImage.data && rightImage.data)
    {
        cvtColor(leftImage, greyLeftImage, COLOR_BGR2GRAY);
        cvtColor(rightImage, greyRightImage, COLOR_BGR2GRAY);

        Ptr<StereoSGBM> sgbm = createStereoSGBM(-64, 192, 15, 600, 2400, 10, 4, 1, 150, 2);
        sgbm->setDisp12MaxDiff(10);

        sgbm->compute(greyLeftImage, greyRightImage, disp);
        normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8U);

        namedWindow("Display Image1", WINDOW_AUTOSIZE );
        namedWindow("Display Image2", WINDOW_AUTOSIZE );
        namedWindow("Display Image3", WINDOW_AUTOSIZE );

        imshow("Display Image1", greyLeftImage);
        imshow("Display Image2", greyRightImage);
        imshow("Display Image3", disp8);
        waitKey(0);
    }
    else
    {
        printf("No data in any of images!\n");
        return -1;
    }

    return 0;
}