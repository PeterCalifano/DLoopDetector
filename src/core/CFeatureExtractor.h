/**
 * @file CFeatureExtractor.h
 * @author PeterC (petercalifano.gs@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-09
 */
#pragma once

#include <iostream>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// DLoopDetector and DBoW2
#include "DLoopDetector.h"
#include <DBoW2/DBoW2.h>
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>

using namespace DLoopDetector;
using namespace DBoW2;

using std::cout, std::string, std::vector, std::fstream, std::ios;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// Generic class to create functors to extract features
template <class TDescriptor>
class CFeatureExtractor
{
  public:
    /**
     * Extracts features
     * @param im image
     * @param keys keypoints extracted
     * @param descriptors descriptors extracted
     */
    virtual void operator()(const cv::Mat &im,
                            vector<cv::KeyPoint> &keys, 
                            vector<TDescriptor> &descriptors) const = 0;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
