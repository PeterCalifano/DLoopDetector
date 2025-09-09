/**
 * File: demo_brief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <string>
#include <vector>

// DLoopDetector and DBoW2
#include "DLoopDetector.h"   // defines BriefLoopDetector
#include <DBoW2/DBoW2.h>     // defines BriefVocabulary
#include <DVision/DVision.h> // Brief

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <core/CDefaultLoopDetector.h>
#include <specializations/CBriefExtractor.h>

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;

using std::cout, std::string, std::vector, std::fstream, std::ios;

// ----------------------------------------------------------------------------
// TODO replace this with command line arguments
static const std::string ROOT_PATH = "/home/peterc/devDir/SLAM-repos/loopClosures_for_SpaceNav/lib/DLoopDetector_PeterCdev"; // path to resources

// Concat paths
static const std::string VOC_FILE = ROOT_PATH + "/resources/brief_k10L6.voc.gz"; // .voc.gz
static const std::string IMAGE_DIR = ROOT_PATH + "/resources/images";
static const std::string POSE_FILE = ROOT_PATH + "/resources/pose.txt";
static const int IMAGE_W = 640; // image size
static const int IMAGE_H = 480;
static const std::string BRIEF_PATTERN_FILE = ROOT_PATH + "/resources/brief_pattern.yml";
// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    bool show = true;
    if (argc > 1 && std::string(argv[1]) == "-noshow")
    {
        show = false;
    }

    // prepares the demo
    CDefaultLoopDetector<BriefVocabulary, BriefLoopDetector, FBrief::TDescriptor> demo(
        VOC_FILE.c_str(),
        IMAGE_DIR.c_str(),
        POSE_FILE.c_str(),
        IMAGE_W,
        IMAGE_H,
        show);

    try
    {
        // run the demo with the given functor to extract features
        BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
        demo.run("BRIEF", extractor);
    }
    catch (const std::string &ex)
    {
        cout << "Error: " << ex << "\n";
    }

    return 0;
}
