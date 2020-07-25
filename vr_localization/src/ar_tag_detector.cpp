/*
 * Created on Tue Jul 21 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "ar_tag_detector.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <queue>

using namespace std;
using namespace cv;

vector<ARTag> ARTagDetector::DetectTags(const cv::Mat& img)
{
    Mat grad_dir, grad_mag;
    unordered_map<uint32_t, vector<Point2f>> segments = boundarySegmentation(img);
    fittingQuads(segments);
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
                                    if(bin_img.at<uchar>(neigh) == 
                                        bin_img.at<uchar>(cur))
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
    return std::move(res);
}

unordered_map<uint32_t, vector<Point2f>> ARTagDetector::boundarySegmentation(const cv::Mat& img)
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
    imshow("bin_img", bin_img);

    Mat uf = unionFind(bin_img);

    unordered_map<uint32_t, vector<Point2f>> segments;
    vector<vector<bool>> visited(uf.rows, vector<bool>(uf.cols, false));
    for(int r = 0; r < uf.rows; r++)
    {
        for(int c = 0; c < uf.cols; c++)
        {
            Point2i cur(c, r);
            visited[r][c] = true;
            for(int i = -1; i <= 1; i++)
            {
                for(int j = -1; j <= 1; j++)
                {
                    Point2i neigh(cur.x + i, cur.y +j);
                    if(neigh.x >= 0 && neigh.x < bin_img.cols && 
                        neigh.y >= 0 && neigh.y < bin_img.rows)
                    {
                        if(visited[neigh.y][neigh.x])
                        {
                            continue;
                        }
                        else
                        {
                            if(bin_img.at<uchar>(cur) != bin_img.at<uchar>(neigh))
                            {
                                uint32_t id = 0;
                                uint16_t r0 = uf.at<uint16_t>(cur);
                                uint16_t r1 = uf.at<uint16_t>(neigh);
                                if(r0 < r1)
                                {
                                    id = ((uint32_t)r0 << 16) + r1;
                                }
                                else
                                {
                                    id = ((uint32_t)r1 << 16) + r0;
                                }
                                Point2f pt;
                                pt.x = static_cast<float>(cur.x + neigh.x) / 2;
                                pt.y = static_cast<float>(cur.y + neigh.y) / 2;
                                segments[id].push_back(pt);
                            }
                        }
                    }
                }
            }
        }
    }
    return std::move(segments);
}

vector<vector<Point2f>> ARTagDetector::fittingQuads(unordered_map<uint32_t, vector<Point2f>>& segments)
{
    /* Sort points by angle in a consistent winding order around their centroid. */
    Mat disp(Size(640, 480), CV_8UC1, Scalar(0));
    for(auto & seg : segments)
    {
        Point2f centreOfGravity(0, 0);
        for(auto pt : seg.second)
        {
            centreOfGravity.x += pt.x;
            centreOfGravity.y += pt.y;
        }
        centreOfGravity.x /= seg.second.size();
        centreOfGravity.y /= seg.second.size();

        sort(seg.second.begin(), seg.second.end(), [&centreOfGravity](Point2f& lhs, Point2f& rhs)
        {
            float lhs_angle = 0, rhs_angle = 0;
            Point2f diff_lhs(lhs.x - centreOfGravity.x, lhs.y - centreOfGravity.y);
            Point2f diff_rhs(rhs.x - centreOfGravity.x, rhs.y - centreOfGravity.y);
            lhs_angle = acos(diff_lhs.y / sqrt(diff_lhs.x * diff_lhs.x + diff_lhs.y *diff_lhs.y + 0.0000001));
            if(diff_lhs.x < 0)
            {
                lhs_angle = 2 * M_PI - lhs_angle;
            }
            rhs_angle = acos(diff_rhs.y / sqrt(diff_rhs.x * diff_rhs.x + diff_rhs.y *diff_rhs.y + 0.0000001));
            if(diff_rhs.x < 0)
            {
                rhs_angle = 2 * M_PI - rhs_angle;
            }
            return lhs_angle < rhs_angle;
            // return false;
        }
        );
    }  
    // Mat disp_seg(480, 640, CV_8UC1, Scalar(0));
    // for(auto & seg: segments)
    // {
    //     for(int i = 0; i < seg.second.size(); i++)
    //     {
    //         // disp_seg.at<uchar>(seg.second[i]) = 255;
    //         circle(disp_seg, seg.second[i], 2, 255, -1);
    //         imshow("disp_seg", disp_seg);
    //         moveWindow("disp_seg", 200, 700);
    //         waitKey(10);
    //         cout << seg.first << endl;
    //     }
    // }
    
    Mat cor_disp(480, 640, CV_8UC1, Scalar(0));
    vector<vector<Point2f>> res;
    for(auto &seg : segments)
    {
        if(seg.second.size() < 200)
        {
            continue;
        }
        int win_size = seg.second.size() / 6;
        if(win_size % 2 == 0)
        {
            win_size -= 1;
        }
        vector<Point2f>& seg_pts = seg.second;
        
        vector<int> errs(seg_pts.size(), 0);
        for(int i = 0; i < seg_pts.size(); i++)
        {
            Mat win_pts_mat(2, win_size, CV_32FC1, Scalar(0));
            Mat disp_seg(480, 640, CV_8UC1, Scalar(0));
            for(int j = -win_size / 2; j <= win_size / 2; j++)
            {
                int cur_index = i+j;
                if(cur_index < 0)
                {
                    cur_index += seg_pts.size();
                }
                else if(cur_index >= seg_pts.size())
                {
                    cur_index -= seg_pts.size();
                }
                assert(cur_index >= 0 && cur_index < seg_pts.size());
                win_pts_mat.at<float>(0, j + win_size / 2) = seg_pts[cur_index].x;
                win_pts_mat.at<float>(1, j + win_size / 2) = seg_pts[cur_index].y;
                if(seg.first == 65561)
                {
                    disp_seg.at<uchar>(seg_pts[cur_index]) = 255;
                }
            }
            // if(seg.first == 65561)
            // {
            //     imshow("disp_seg", disp_seg);
            //     moveWindow("disp_seg", 200, 700);
            //     waitKey(0);
            // }
            Mat cov_mat, mean_mat;
            calcCovarMatrix(win_pts_mat, cov_mat, mean_mat, CV_COVAR_NORMAL | CV_COVAR_COLS, CV_32F);

            Mat eigen_vecs, eigen_values;
            eigen(cov_mat, eigen_values, eigen_vecs);
            Vec2f dir(eigen_vecs.at<float>(0, 0), eigen_vecs.at<float>(1, 0));
            // Ax + By + C = 0;
            float A = 0, B = 0, C = 0;
            if(dir[0] == 0)
            {
                A = 1;
                B = 0;
                C = -mean_mat.at<float>(0, 0); // C = - mean.x
            }
            else
            {
                float k = dir[1] / dir[0];
                A = k;
                B = -1;
                C = -k * mean_mat.at<float>(0, 0) + mean_mat.at<float>(1, 0);
            }

            float err = 0;
            float sqrt_A2_B2 = sqrt(A*A + B*B);
            for(int j = 0; j < win_size; j++)
            {
                err += abs(A * win_pts_mat.at<float>(0, j) + B * win_pts_mat.at<float>(1, j) + C) / sqrt_A2_B2;
            }
            errs[i] = err;

            cor_disp.at<uchar>(seg_pts[i]) = 255;
        }
        vector<Point2f> corners;
        for(int i = 0; i < seg_pts.size(); i++)
        {
            int prev = i - 1 < 0 ? seg_pts.size() - 1 : i - 1;
            int next = i + 1 >= seg_pts.size() ? 0 : i + 1;
            if(errs[prev] < errs[i] && errs[i] > errs[next])
            {
                corners.push_back(seg_pts[i]);
            }
        }
        if(corners.size() <= 5)
        {
            res.push_back(corners);
            for(int i = 0; i < corners.size(); i++)
            {
                circle(cor_disp, corners[i], 3, Scalar(255), -1);
            }
        }
    }
    
    imshow("disp_cor", cor_disp);
    moveWindow("disp_cor", 200, 700);
    return std::move(res);
}