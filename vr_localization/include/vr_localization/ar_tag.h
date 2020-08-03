/*
 * Created on Tue Jul 21 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */
#ifndef AR_TAG_H__
#define AR_TAG_H__
#include <opencv2/core/core.hpp>
#include <vector>
#include <bitset>

class ARTag
{
public:
    ARTag(uint16_t code) : code_(code){};
    void UpdateLocation(uint16_t code, std::vector<cv::Point2f>& corners);
private:
    void rotate90Clockwise(uint16_t& code, std::vector<cv::Point2f>& corners);
    std::bitset<16> code_;
    std::vector<cv::Point2f> corners_;
    bool updated_ = false;
};

#endif