/*
 * Created on Mon Aug 03 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "ar_tag.h"

using namespace std;
using namespace cv;

void ARTag::UpdateLocation(uint16_t code, vector<cv::Point2f>& corners)
{
    uint16_t this_code = static_cast<uint16_t>(code_.to_ulong());
    for(int i = 0; i < 4; i++)
    {
        rotate90Clockwise(code, corners);
    }
}

void ARTag::rotate90Clockwise(uint16_t& code, vector<cv::Point2f>& corners)
{
    bitset<16> cur_code(code);
    
    // swap 0 and 1
    bool tmp = cur_code[0];
    cur_code[0] = cur_code[1];
    cur_code[1] = tmp;
    // swap 2 and 3
    tmp = cur_code[2];
    cur_code[2] = cur_code[3];
    cur_code[3] = tmp;

    tmp = cur_code[7];
    for(int i = 7; i >= 5; i--)
    {
        cur_code[i] = cur_code[i-1];
    }
    cur_code[4] = tmp;

    tmp = cur_code[11];
    for(int i = 11; i >= 9; i--)
    {
        cur_code[i] = cur_code[i-1];
    }
    cur_code[9] = tmp;

    Point2f tmp_point = corners[3];
    for(int i = 3; i >= 1; i++)
    {
        corners[i] = corners[i-1];
    }
    corners[0] = tmp_point;
}