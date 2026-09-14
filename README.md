DLoopDetector
=============

## Overview

DLoopDetector is an open source C++ library for detecting loops in a sequence of local-feature
frames. It implements the approach presented in GalvezTRO12 using a DBoW2 database, temporal
islands, descriptor matching, and geometric verification.

This fork uses a C++20 descriptor-policy contract. Its built-in aliases support ORB, BRIEF, SURF64,
and SIFT; fixed-float policies also support learned descriptor widths without a runtime feature-name
registry. The default implementation separates `CBowCandidateRetriever<TPolicy>` from
`CKnnFundamentalVerifier<TPolicy>` and composes them through `TemplatedLoopDetector<TPolicy>`.

The historical `TemplatedLoopDetector` source adapter remains available for migration, but its
legacy DI, FLANN, and exhaustive geometry names intentionally select the same deterministic
policy-distance verifier. It does not reproduce the old NSS score normalization or bit-for-bit
historical temporal/island tuning. ORB, BRIEF, SURF64, and SIFT themselves remain fully supported
through the new vocabulary, retrieval, and geometric-verification contracts.

DLoopDetector requires C++20, OpenCV 4.2 or newer, and DBoW2. It no longer depends on DLib or Boost.

## Citing

If you use this software in an academic work, please cite:

    @ARTICLE{GalvezTRO12,
      author={G\'alvez-L\'opez, Dorian and Tard\'os, J. D.},
      journal={IEEE Transactions on Robotics},
      title={Bags of Binary Words for Fast Place Recognition in Image Sequences},
      year={2012},
      month={October},
      volume={28},
      number={5},
      pages={1188--1197},
      doi={10.1109/TRO.2012.2197158},
      ISSN={1552-3098}
    }

## Install and usage notes

Install DBoW2 first or expose its build/install prefix through `CMAKE_PREFIX_PATH`, then configure
this project normally. `DLoopDetector` is an interface target exported as
`DLoopDetector::DLoopDetector`.

Applications select the descriptor policy at compile time. They provide original-image
`cv::KeyPoint` coordinates plus an aligned descriptor batch. DLoopDetector owns retrieval and
geometry only; inference, model loading, tensor layouts, thresholding, coordinate restoration, and
transport remain application responsibilities.
