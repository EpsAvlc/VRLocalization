/*
 * Created on Tue Jul 21 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "ar_tag_detector.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/

using namespace std;
using namespace cv;

vector<ARTag> ARTagDetector::DetectTags(const cv::Mat& img)
{
    Mat origin = img.clone();
    vector<Mat> bgr;
    split(origin, bgr); 
    imshow("r")
}