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

#include "ar_tag.h"

class ARTagDetector
{
public:
    ARTagDetector() {};
    std::vector<ARTag> DetectTags(const cv::Mat& img);
};

#endif // !AR_TAG_DETECTOR__
