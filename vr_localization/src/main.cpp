/*
 * Created on Tue Jul 21 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */
#include <iostream>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ar_tag_detector.h"

using namespace std;
using namespace cv;

int main(int argc, char**argv)
{
    ros::init(argc, argv, "sim_demo");
    Mat img = imread("/home/cm/Workspaces/VR_localization/src/vr_localization/example_imgs/img1.png");
    imshow("src", img);
    moveWindow("src", 200, 700);

    ARTagDetector atd;
    atd.DetectTags(img);
    waitKey(0);
}
