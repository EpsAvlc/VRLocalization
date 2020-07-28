/*
 * Created on Tue Jul 21 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */
#ifndef AR_TAG_DETECTOR__
#define AR_TAG_DETECTOR__

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>

#include <unordered_map>
#include <stack>
#include <list>

#include "ar_tag.h"

class ARTagDetector
{
public:
    ARTagDetector() {};
    std::vector<ARTag> DetectTags(const cv::Mat& img);
private:
    /* big function */
    cv::Mat autoThreshold(const cv::Mat& img);
    std::unordered_map<uint32_t, std::vector<cv::Point2f>>  boundarySegmentation(const cv::Mat& img);
    std::vector<std::vector<cv::Point2f>> fittingQuads(std::unordered_map<uint32_t, std::vector<cv::Point2f>>& segments);
    uint16_t decoding(const cv::Mat& img, const std::vector<cv::Point2f>& corners);

    /* tool function */
    cv::Mat unionFind(const cv::Mat& bin_img);
    std::vector<std::vector<int>> getPermutations(const std::vector<int>& indices);
    float fitLineAndComputeMSE(const cv::Mat& pts_mat);
};

#endif // !AR_TAG_DETECTOR__
