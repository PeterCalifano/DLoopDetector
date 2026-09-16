/**
 * @file CKnnFundamentalMatrixVerifier.h
 * @brief Policy-based two-nearest matching and fundamental-matrix verification.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See the DLoopDetector LICENSE.txt file.
 */

#pragma once

#include "LoopDetectionContracts.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace DLoopDetector
{
    /** @brief Matching, RANSAC, and acceptance gates for fundamental-matrix verification. */
    struct SGeometricVerificationParameters
    {
        double maximum_neighbor_ratio = 0.8; ///< Strict best/second-best distance ratio.
        std::size_t minimum_correspondences = 8; ///< Matches required before RANSAC.
        std::size_t minimum_inliers = 12; ///< Inliers required to accept a model.
        double maximum_reprojection_error = 1.0; ///< RANSAC pixel-error threshold.
        double ransac_confidence = 0.99; ///< Requested probability of a valid model.
        int maximum_ransac_iterations = 2000; ///< Positive OpenCV RANSAC iteration cap.
    };

    /**
     * @brief Verify fundamental-matrix geometry with unique two-nearest matches and RANSAC.
     *
     * Matching is brute-force through the descriptor policy so binary and float descriptors share
     * one implementation. The class owns no frame data and is safe to reuse for independent pairs.
     *
     * @tparam TPolicy Descriptor policy supplying validation and distance operations.
     */
    template <DBoW2::DescriptorPolicy TPolicy>
    class CKnnFundamentalMatrixVerifier
    {
      public:
        using Policy = TPolicy;
        using Descriptor = typename TPolicy::Descriptor;

        /**
         * @brief Construct a verifier with explicit matching and RANSAC gates.
         * @param parameters Verification parameters.
         * @throws std::invalid_argument If any gate is outside its valid domain.
         */
        explicit CKnnFundamentalMatrixVerifier(
            const SGeometricVerificationParameters &parameters = {})
            : parameters_(parameters)
        {
            ValidateParameters();
        }

        /**
         * @brief Match and verify a reference/query frame pair.
         * @param reference_keypoints Reference keypoints in original-image pixel coordinates.
         * @param reference_descriptors Descriptors aligned one-to-one with reference keypoints.
         * @param query_keypoints Query keypoints in original-image pixel coordinates.
         * @param query_descriptors Descriptors aligned one-to-one with query keypoints.
         * @return Explicit expected outcome plus correspondence and inlier counts.
         * @throws std::invalid_argument If keypoint/descriptor counts differ or data is nonfinite.
         */
        [[nodiscard]] SGeometricVerificationResult Verify(
            const std::span<const cv::KeyPoint> reference_keypoints,
            const std::span<const Descriptor> reference_descriptors,
            const std::span<const cv::KeyPoint> query_keypoints,
            const std::span<const Descriptor> query_descriptors) const
        {
            // Validate both frame contracts once before entering quadratic matching.
            ValidateFrame(reference_keypoints, reference_descriptors, "reference");
            ValidateFrame(query_keypoints, query_descriptors, "query");

            // Reject frames that cannot provide two neighbors or a solvable query set.
            SGeometricVerificationResult result;
            if (reference_descriptors.size() < 2 ||
                query_descriptors.size() < parameters_.minimum_correspondences)
            {
                result.status = EGeometricVerificationStatus::insufficient_features;
                return result;
            }

            // Retain ratio-tested matches with at most one query per reference descriptor.
            std::vector<SDescriptorMatch> matches =
                MatchUnique(reference_descriptors, query_descriptors);
            result.correspondence_count = matches.size();

            if (matches.size() < parameters_.minimum_correspondences)
            {
                result.status = EGeometricVerificationStatus::insufficient_matches;
                return result;
            }

            // Project descriptor indices into aligned image coordinates for RANSAC.
            std::vector<cv::Point2f> reference_points;
            std::vector<cv::Point2f> query_points;
            reference_points.reserve(matches.size());
            query_points.reserve(matches.size());

            for (const SDescriptorMatch &match : matches)
            {
                reference_points.push_back(reference_keypoints[match.reference_index].pt);
                query_points.push_back(query_keypoints[match.query_index].pt);
            }

            // Estimate one fundamental model and reject malformed or rank-deficient results.
            cv::Mat inlier_mask;
            const cv::Mat fundamental_matrix = cv::findFundamentalMat(
                reference_points, query_points, cv::FM_RANSAC,
                parameters_.maximum_reprojection_error, parameters_.ransac_confidence,
                parameters_.maximum_ransac_iterations, inlier_mask);

            if (IsDegenerate(fundamental_matrix))
            {
                result.status = EGeometricVerificationStatus::degenerate_configuration;
                return result;
            }

            // Accept the valid model only after it reaches the configured inlier gate.
            result.inlier_count = static_cast<std::size_t>(cv::countNonZero(inlier_mask));
            result.status = result.inlier_count >= parameters_.minimum_inliers
                                ? EGeometricVerificationStatus::accepted
                                : EGeometricVerificationStatus::ransac_rejected;
            return result;
        }

      private:
        struct SDescriptorMatch
        {
            std::size_t reference_index = 0;
            std::size_t query_index = 0;
            double distance = 0.0;
        };

        void ValidateParameters() const
        {
            if (!std::isfinite(parameters_.maximum_neighbor_ratio) ||
                !(parameters_.maximum_neighbor_ratio > 0.0 &&
                  parameters_.maximum_neighbor_ratio < 1.0))
            {
                throw std::invalid_argument("maximum_neighbor_ratio must be in (0, 1).");
            }

            if (parameters_.minimum_correspondences < 8 ||
                parameters_.minimum_inliers == 0)
            {
                throw std::invalid_argument(
                    "Fundamental verification requires at least eight "
                    "correspondences and one inlier.");
            }

            if (!std::isfinite(parameters_.maximum_reprojection_error) ||
                !(parameters_.maximum_reprojection_error > 0.0) ||
                !std::isfinite(parameters_.ransac_confidence) ||
                !(parameters_.ransac_confidence > 0.0 &&
                  parameters_.ransac_confidence < 1.0) ||
                parameters_.maximum_ransac_iterations <= 0)
            {
                throw std::invalid_argument("Invalid fundamental-matrix RANSAC parameters.");
            }
        }

        static void ValidateFrame(
            const std::span<const cv::KeyPoint> keypoints,
            const std::span<const Descriptor> descriptors,
            const char *frame_name)
        {
            if (keypoints.size() != descriptors.size())
            {
                throw std::invalid_argument(std::string(frame_name) +
                                            " keypoint and descriptor counts differ.");
            }

            // Reject nonfinite image coordinates before invoking OpenCV geometry.
            for (const cv::KeyPoint &keypoint : keypoints)
            {
                if (!std::isfinite(keypoint.pt.x) || !std::isfinite(keypoint.pt.y))
                {
                    throw std::invalid_argument(std::string(frame_name) +
                                                " keypoints contain nonfinite coordinates.");
                }
            }

            // Validate once at the public boundary before entering the quadratic matching pass.
            for (const Descriptor &descriptor : descriptors)
            {
                TPolicy::Validate(descriptor);
            }
        }

        /**
         * @brief Apply two-nearest ratio matching with unique reference assignments.
         * @param reference_descriptors Candidate descriptors searched for every query.
         * @param query_descriptors Query descriptors matched against the reference set.
         * @return Deterministic matches containing at most one query per reference descriptor.
         */
        [[nodiscard]] std::vector<SDescriptorMatch> MatchUnique(
            const std::span<const Descriptor> reference_descriptors,
            const std::span<const Descriptor> query_descriptors) const
        {
            constexpr std::size_t no_query = std::numeric_limits<std::size_t>::max();

            // Initialize one deterministic winner slot for every reference descriptor.
            std::vector<SDescriptorMatch> best_matches(reference_descriptors.size());
            for (std::size_t reference_index = 0;
                 reference_index < best_matches.size(); ++reference_index)
            {
                best_matches[reference_index] = {
                    reference_index, no_query, std::numeric_limits<double>::infinity()};
            }

            // Find the two nearest references for every query descriptor.
            for (std::size_t query_index = 0;
                 query_index < query_descriptors.size(); ++query_index)
            {
                std::size_t best_index = 0;
                double best_distance = std::numeric_limits<double>::infinity();
                double second_distance = std::numeric_limits<double>::infinity();

                for (std::size_t reference_index = 0;
                     reference_index < reference_descriptors.size(); ++reference_index)
                {
                    const double distance = TPolicy::Distance(
                        query_descriptors[query_index], reference_descriptors[reference_index]);
                    if (distance < best_distance)
                    {
                        second_distance = best_distance;
                        best_distance = distance;
                        best_index = reference_index;
                    }
                    else if (distance < second_distance)
                    {
                        second_distance = distance;
                    }
                }

                // Apply the ratio gate and resolve reference collisions by minimum distance.
                if (std::isfinite(second_distance) &&
                    best_distance < parameters_.maximum_neighbor_ratio * second_distance)
                {
                    SDescriptorMatch &selected = best_matches[best_index];
                    if (best_distance < selected.distance)
                    {
                        selected = {best_index, query_index, best_distance};
                    }
                }
            }

            // Discard unmatched slots after retaining one deterministic query per reference.
            std::erase_if(best_matches, [no_query](const SDescriptorMatch &match) {
                return match.query_index == no_query;
            });

            return best_matches;
        }

        /**
         * @brief Reject malformed or numerically rank-deficient fundamental matrices.
         * @param fundamental_matrix OpenCV model returned by RANSAC.
         * @return True when the matrix cannot support geometric verification.
         */
        [[nodiscard]] static bool IsDegenerate(const cv::Mat &fundamental_matrix)
        {
            if (fundamental_matrix.empty() || fundamental_matrix.rows != 3 ||
                fundamental_matrix.cols != 3 || !cv::checkRange(fundamental_matrix))
            {
                return true;
            }

            // Require two significant singular values for a usable fundamental model.
            cv::SVD decomposition(fundamental_matrix, cv::SVD::NO_UV);
            const double largest = decomposition.w.at<double>(0);
            const double second = decomposition.w.at<double>(1);

            return largest <= std::numeric_limits<double>::epsilon() ||
                   second <= largest * 1.0e-9;
        }

        SGeometricVerificationParameters parameters_;
    };

    static_assert(GeometricVerifier<
                  CKnnFundamentalMatrixVerifier<DBoW2::FixedFloatDescriptorPolicy<1>>,
                  DBoW2::FixedFloatDescriptorPolicy<1>>);
} // namespace DLoopDetector
