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
    Mat bin_img = autoThreshold(img);
    unordered_map<uint32_t, vector<Point2f>> segments = boundarySegmentation(bin_img);
    auto quads = fittingQuads(segments);
    for(int i = 0; i < quads.size(); i ++)
    {
        uint16_t code = decoding(bin_img, quads[i]);
        
    }
    return vector<ARTag>();
}

Mat ARTagDetector::autoThreshold(const cv::Mat& img)
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
    return move(bin_img);
}

unordered_map<uint32_t, vector<Point2f>> ARTagDetector::boundarySegmentation(const cv::Mat& bin_img)
{

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
    Mat disp_seg(480, 640, CV_8UC1, Scalar(0));
    
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
            }

            errs[i] = fitLineAndComputeMSE(win_pts_mat);

            cor_disp.at<uchar>(seg_pts[i]) = 255;
        }
        vector<Point2f> corners;
        vector<int> corner_indices;
        for(int i = 0; i < seg_pts.size(); i++)
        {
            int prev = i - 1 < 0 ? seg_pts.size() - 1 : i - 1;
            int next = i + 1 >= seg_pts.size() ? 0 : i + 1;
            if(errs[prev] < errs[i] && errs[i] > errs[next])
            {
                corners.push_back(seg_pts[i]);
                corner_indices.push_back(i);
            }
        }
        if(corners.size() < 4)
        {
            continue;
        }
        /* iterate through all permutations of four candidate corners, 
         fitting lines to each side of the candidate quad. */
        if(corners.size() > 4)
        {
            vector<vector<int>> permutations = getPermutations(corner_indices);
            int best_permu_index = -1;
            float min_err = numeric_limits<float>::max();
            for(int i = 0; i < permutations.size(); i++)
            {
                vector<int>& permu = permutations[i];
                /* filter whose corner angles deviate too far from 90 degree */
                vector<Vec2f> dirs;
                for(int j = 0; j < 4; j++)
                {
                    int prev = j - 1 < 0 ? j + 3 : j - 1; 
                    float diff_x = corners[permu[j]].x - corners[permu[prev]].x;
                    float diff_y = corners[permu[j]].y - corners[permu[prev]].y;
                    float norm = sqrt(diff_x * diff_x + diff_y * diff_y);
                    diff_x /= norm;
                    diff_y /= norm;
                    dirs.push_back(Vec2f(diff_x, diff_y));
                }
                bool is_corner_angles_misfit = false;
                for(int j = 0; j < 4; j++)
                {
                    int prev = j - 1 < 0 ? j + 3 : j - 1;
                    if(fabs(dirs[j].dot(dirs[prev])) > 0.5)
                    {
                        is_corner_angles_misfit = true;
                        break;
                    }
                }
                if(is_corner_angles_misfit)
                {
                    continue;
                }

                float err_sum = 0;
                for(int j = 0; j < 4; j++)
                {
                    int prev = j - 1 < 0 ? 3 : j - 1; 
                    int start = corner_indices[permu[prev]];
                    int end = corner_indices[permu[j]];
                    if(start > end)
                    {
                        start -= seg.second.size();
                    }
                    Mat pts_mat(2, end - start + 1, CV_32FC1, Scalar(0));
                    for(int k = start; k < end; k++)
                    {
                        int cur_index = k < 0 ? k + seg.second.size() : k;
                        pts_mat.at<float>(0, k - start) = seg.second[cur_index].x;
                        pts_mat.at<float>(1, k - start) = seg.second[cur_index].y;
                    }
                    float err = fitLineAndComputeMSE(pts_mat);
                    err /= start - end + 1;
                    err_sum += err;
                }
                if(err_sum / 4 >= 2 )
                    continue;
                if(err_sum < min_err)
                {
                    min_err = err_sum;
                    best_permu_index = i;
                }
            }
            if(best_permu_index != -1)
            {
                vector<Point2f> best_corners;
                for(int i = 0; i < 4; i++)
                {
                    best_corners.push_back(corners[permutations[best_permu_index][i]]);
                }
                corners.swap(best_corners);
            }
            else
            {
                continue;
            }
            
        }
        res.push_back(corners);
    }
    
    for(int i = 0; i < res.size(); i++)
    {
        for(int j = 0; j < res[i].size(); j++)
            circle(cor_disp, res[i][j], 2, 255, -1);
    }
    imshow("disp_cor", cor_disp);
    moveWindow("disp_cor", 200, 700);
    return std::move(res);
}

uint16_t ARTagDetector::decoding(const cv::Mat& img, const vector<Point2f>& corners)
{
    vector<Point2f> target_pts(4);
    target_pts[0] = Point2f(0, 0);
    target_pts[1] = Point2f(100, 0);
    target_pts[2] = Point2f(100, 100);
    target_pts[3] = Point2f(0, 100);

    Mat affineTransform = getPerspectiveTransform(corners, target_pts);
    Mat marker;
    warpPerspective(img, marker, affineTransform, Size(100, 100));
    marker = marker(Range(14, 86), Range(14, 86));
    resize(marker, marker, Size(100, 100));
    
    uint16_t code = 0;
    vector<pair<Point2f, Point2f>> lines;
    Point2i start_point, end_point;

    // ||
    start_point.x = 0;
    start_point.y = 50;
    end_point.x = 100;
    end_point.y = 50;
    lines.push_back(make_pair(start_point, end_point));

    // --
    start_point.x = 50;
    start_point.y = 0;
    end_point.x = 50;
    end_point.y = 100;
    lines.push_back(make_pair(start_point, end_point));

    // left up -> right down 
    start_point.x = 0;
    start_point.y = 0;
    end_point.x = 100;
    end_point.y = 100;
    lines.push_back(make_pair(start_point, end_point));

    // left down -> right up
    start_point.x = 100;
    start_point.y = 0;
    end_point.x = 0;
    end_point.y = 100;
    lines.push_back(make_pair(start_point, end_point));

    // | up
    start_point.x = 50;
    start_point.y = 0;
    end_point.x = 50;
    end_point.y = 50;
    lines.push_back(make_pair(start_point, end_point));

    // - right
    start_point.x = 100;
    start_point.y = 50;
    end_point.x = 50;
    end_point.y = 50;
    lines.push_back(make_pair(start_point, end_point));
    
    // | down
    start_point.x = 50;
    start_point.y = 100;
    end_point.x = 50;
    end_point.y = 50;
    lines.push_back(make_pair(start_point, end_point));

    // - left
    start_point.x = 0;
    start_point.y = 50;
    end_point.x = 50;
    end_point.y = 50;
    lines.push_back(make_pair(start_point, end_point));

    /* \ up */
    start_point.x = 0;
    start_point.y = 0;
    end_point.x = 50;
    end_point.y = 50;
    lines.push_back(make_pair(start_point, end_point));

    /* / up */
    start_point.x = 100;
    start_point.y = 0;
    end_point.x = 50;
    end_point.y = 50;
    lines.push_back(make_pair(start_point, end_point));

    /* \ down */
    start_point.x = 100;
    start_point.y = 100;
    end_point.x = 50;
    end_point.y = 50;
    lines.push_back(make_pair(start_point, end_point));

    /* / down */
    start_point.x = 0;
    start_point.y = 100;
    end_point.x = 50;
    end_point.y = 50;
    lines.push_back(make_pair(start_point, end_point));

    for(int i = 0; i < lines.size(); i++)
    {
        Mat line_mat(Size(100, 100), CV_8UC1, Scalar(0));
        line(line_mat, lines[i].first, lines[i].second, Scalar(255), 1);
        imshow("line_mat", line_mat);
        Mat and_mat;
        bitwise_and(marker, line_mat, and_mat);

        int non_zero_count = countNonZero(and_mat);
        Point2f& start_pt = lines[i].first;
        Point2f& end_pt = lines[i].second;
        int length = i < 4 ? 100 : 50;
        // if(non_zero_count >= length / 4 * 3)
        // {
        //     code += 1 << i;
        //     cout << "response" << endl;
        //     cout << code << endl;
        // }

        // imshow("and_mat", and_mat);
        // imshow("marker", marker);
        // moveWindow("marker", 200, 900);
        // moveWindow("line_mat", 300, 700);
        // moveWindow("and_mat", 400, 1100);
        // cout << "Non zero count: " << non_zero_count << endl;
        // cout << "length: " << length << endl;
        // waitKey(0);
    }

    return code;
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

vector<vector<int>> ARTagDetector::getPermutations(const vector<int>& indices)
{
    vector<vector<int>> res;
    for(int i = 0; i < indices.size() - 4 + 1; i++)
    {
        vector<int> permutation(4, 0);
        permutation[0] = i;
        for(int j = i+1; j < indices.size() - 4 + 2; j++)
        {
            permutation[1] = j;
            for(int k = j + 1; k < indices.size() - 4 + 3; k++)
            {
                permutation[2] = k;
                for(int l = k + 1; l < indices.size(); l++)
                {
                    permutation[3] = l;
                    res.push_back(permutation);
                }
            }
        }
    }
    return res;
}

float ARTagDetector::fitLineAndComputeMSE(const cv::Mat& pts_mat)
{
    Mat cov_mat, mean_mat;
    calcCovarMatrix(pts_mat, cov_mat, mean_mat, CV_COVAR_NORMAL | CV_COVAR_COLS, CV_32F);

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
    for(int j = 0; j < pts_mat.cols; j++)
    {
        err += abs(A * pts_mat.at<float>(0, j) + B * pts_mat.at<float>(1, j) + C) / sqrt_A2_B2;
    }
    return err;
}