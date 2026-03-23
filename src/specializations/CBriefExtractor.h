/**
 * @file CBriefExtractor.h
 * @author PeterC (petercalifano.gs@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-09
 */
#pragma once

#include <iostream>
#include <string>
#include <vector>

// DLoopDetector and DBoW2
#include <DLoopDetector/DLoopDetector.h> // defines BriefLoopDetector
#include <DBoW2/DBoW2.h>     // defines BriefVocabulary
#include <DVision/DVision.h> // Brief

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <core/CDefaultLoopDetector.h>

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using std::vector, std::string, std::cout, std::fstream, std::ios;

/// This functor extracts BRIEF descriptors in the required format
class BriefExtractor : public CFeatureExtractor<BRIEF256::bitset>
{
  public:
    /**
     * Extracts features from an image
     * @param im image
     * @param keys keypoints extracted
     * @param descriptors descriptors extracted
     */
    void operator()(const cv::Mat &im,
                    vector<cv::KeyPoint> &keys, vector<BRIEF256::bitset> &descriptors) const override;

    /**
     * Creates the brief extractor with the given pattern file
     * @param pattern_file
     */
    BriefExtractor(const std::string &pattern_file);

  private:
    /// BRIEF descriptor extractor
    BRIEF256 m_brief;
};

// ----------------------------------------------------------------------------

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
    // The DVision::BRIEF extractor computes a random pattern by default when
    // the object is created.
    // We load the pattern that we used to build the vocabulary, to make
    // the descriptors compatible with the predefined vocabulary

    // loads the pattern
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if (!fs.isOpened())
        throw string("Could not open file ") + pattern_file;

    vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    m_brief.importPairs(x1, y1, x2, y2);
}

// ----------------------------------------------------------------------------

void BriefExtractor::operator()(const cv::Mat &im,
                                vector<cv::KeyPoint> &keys, vector<BRIEF256::bitset> &descriptors) const
{
    // extract FAST keypoints with opencv
    const int fast_th = 20; // corner detector response threshold
    cv::FAST(im, keys, fast_th, true);

    // compute their BRIEF descriptor
    m_brief.compute(im, keys, descriptors);
}

// ----------------------------------------------------------------------------
