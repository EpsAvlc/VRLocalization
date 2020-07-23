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
#include <opencv2/highgui/highgui.hpp>

#include <stack>

using namespace std;
using namespace cv;

vector<ARTag> ARTagDetector::DetectTags(const cv::Mat& img)
{
    Mat grad_dir, grad_mag;
    boundarySegmentation(img);

    return vector<ARTag>();
}

Mat ARTagDetector::unionFind(const cv::Mat& bin_img)
{
    Mat res(bin_img.size(), CV_16UC1, Scalar(0));
    unsigned short label = 1;
    for(int r = 0; r < bin_img.rows; r++)
        for(int c = 0; c < bin_img.cols; c++)
        {
            if(res.at<unsigned short>(r, c) == 0)
            {
                res.at<unsigned short>(r, c) = label;
                stack<Point2i> s;
                s.push(Point2i(c, r));
                while(!s.empty())
                {
                    Point2i cur = s.top();
                    s.pop();
                    for(int i = -1; i <= 1; i++) 
                        for(int j = -1; j <= 1; j++)
                        {
                            Point2i neigh(cur.x + i, cur.y +j);
                            if(neigh.x >= 0 && neigh.x < bin_img.cols && 
                                neigh.y >= 0 && neigh.y < bin_img.rows)
                            {
                                if(res.at<unsigned short>(neigh) == 0)
                                {
                                    if(bin_img.at<unsigned short>(neigh) == 
                                        bin_img.at<unsigned short>(cur))
                                        {
                                            res.at<unsigned short>(neigh) = label;
                                            s.push(neigh);
                                        }
                                }
                            }
                        }
                }
                if(label < 65535)
                    label++;
                else
                {
                    cerr << "label over flow. " << endl;
                    throw runtime_error("label over flow.");
                } 
            }
        }
}

void ARTagDetector::boundarySegmentation(const cv::Mat& img)
{
    Mat gray_img;
    if(img.channels() == 3)
    {
        Mat img_hsv;
        cvtColor(img, img_hsv, COLOR_BGR2HSV);
        vector<Mat> hsv;
        split(img_hsv, hsv);
        gray_img = hsv[1];
    }
    else
    {
        gray_img = img.clone();
    }
    Mat bin_img;
    threshold(gray_img, bin_img, -1, 255, THRESH_BINARY | THRESH_OTSU);

    Mat uf = unionFind(bin_img);
}
