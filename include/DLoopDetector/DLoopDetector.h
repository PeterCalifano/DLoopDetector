/**
 * @file DLoopDetector.h
 * @brief Public include and built-in policy aliases for DLoopDetector.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See the DLoopDetector LICENSE.txt file.
 */

#pragma once

#include "CBowCandidateRetriever.h"
#include "CKnnFundamentalMatrixVerifier.h"
#include "LoopDetectionContracts.h"
#include "TemplatedLoopDetector.h"

#include <DBoW2/DBoW2.h>

/** @brief ORB compatibility loop detector. */
using OrbLoopDetector = DLoopDetector::TemplatedLoopDetector<DBoW2::FORB>;

/** @brief BRIEF compatibility loop detector. */
using BriefLoopDetector = DLoopDetector::TemplatedLoopDetector<DBoW2::FBrief>;

/** @brief SURF64 compatibility loop detector. */
using Surf64LoopDetector = DLoopDetector::TemplatedLoopDetector<DBoW2::FSurf64>;

/** @brief SIFT compatibility loop detector. */
using SiftLoopDetector = DLoopDetector::TemplatedLoopDetector<DBoW2::FSIFT>;
