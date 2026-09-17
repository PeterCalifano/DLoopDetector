/**
 * @file CKnnFundamentalMatrixVerifier.h
 * @brief Policy-based two-nearest matching and fundamental-matrix verification.
 * @author Dorian Galvez-Lopez, Pietro Califano, and Codex GPT-5.6
 * @date 2026-09-16
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
            validateParameters();
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
        [[nodiscard]] SGeometricVerificationResult verify(
            const std::span<const cv::KeyPoint> reference_keypoints,
            const std::span<const Descriptor> reference_descriptors,
            const std::span<const cv::KeyPoint> query_keypoints,
            const std::span<const Descriptor> query_descriptors) const
        {
            if (reference_keypoints.size() != reference_descriptors.size() ||
                query_keypoints.size() != query_descriptors.size())
            {
                throw std::invalid_argument("Verifier keypoint and descriptor counts differ.");
            }

            // Preserve the compatibility entry point while sharing the double-pixel core.
            std::vector<cv::Point2d> reference_points;
            std::vector<cv::Point2d> query_points;
            reference_points.reserve(reference_keypoints.size());
            query_points.reserve(query_keypoints.size());
            for (const cv::KeyPoint &keypoint : reference_keypoints)
            {
                reference_points.emplace_back(keypoint.pt.x, keypoint.pt.y);
            }

            for (const cv::KeyPoint &keypoint : query_keypoints)
            {
                query_points.emplace_back(keypoint.pt.x, keypoint.pt.y);
            }

            return verifyPoints(reference_points, reference_descriptors,
                                query_points, query_descriptors);
        }

        /**
         * @brief Match and verify original-image double-precision point coordinates.
         * @param reference_keypoints Reference image pixels aligned with descriptors.
         * @param reference_descriptors Descriptors aligned with reference pixels.
         * @param query_keypoints Query image pixels aligned with descriptors.
         * @param query_descriptors Descriptors aligned with query pixels.
         * @return Explicit outcome and fixed RANSAC inlier indices.
         * @throws std::invalid_argument If counts differ or any input is nonfinite.
         */
        [[nodiscard]] SGeometricVerificationResult verifyPoints(
            const std::span<const cv::Point2d> reference_keypoints,
            const std::span<const Descriptor> reference_descriptors,
            const std::span<const cv::Point2d> query_keypoints,
            const std::span<const Descriptor> query_descriptors) const
        {
            // Validate both frame contracts once before entering quadratic matching.
            validateFrame(reference_keypoints, reference_descriptors, "reference");
            validateFrame(query_keypoints, query_descriptors, "query");

            // Reject a frame that cannot supply the required number of unique matches.
            SGeometricVerificationResult result;
            if (reference_descriptors.size() < parameters_.minimum_correspondences ||
                query_descriptors.size() < parameters_.minimum_correspondences)
            {
                result.status = EGeometricVerificationStatus::insufficient_features;
                return result;
            }

            // Retain ratio-tested matches with at most one query per reference descriptor.
            std::vector<SDescriptorMatch> matches =
                matchUnique(reference_descriptors, query_descriptors);
            result.correspondence_count = matches.size();

            if (matches.size() < parameters_.minimum_correspondences)
            {
                result.status = EGeometricVerificationStatus::insufficient_matches;
                return result;
            }

            // Project descriptor indices into aligned image coordinates for RANSAC.
            std::vector<cv::Point2d> reference_points;
            std::vector<cv::Point2d> query_points;
            reference_points.reserve(matches.size());
            query_points.reserve(matches.size());

            for (const SDescriptorMatch &match : matches)
            {
                reference_points.push_back(reference_keypoints[match.reference_index]);
                query_points.push_back(query_keypoints[match.query_index]);
            }

            // Estimate one fundamental model and reject malformed or rank-deficient results.
            cv::Mat inlier_mask;
            const cv::Mat fundamental_matrix = cv::findFundamentalMat(
                reference_points, query_points, cv::FM_RANSAC,
                parameters_.maximum_reprojection_error, parameters_.ransac_confidence,
                parameters_.maximum_ransac_iterations, inlier_mask);

            if (isDegenerate(fundamental_matrix))
            {
                result.status = EGeometricVerificationStatus::degenerate_configuration;
                return result;
            }

            // Accept the valid model only after it reaches the configured inlier gate.
            if (inlier_mask.type() != CV_8UC1 ||
                inlier_mask.total() != matches.size() || !inlier_mask.isContinuous())
            {
                result.status = EGeometricVerificationStatus::degenerate_configuration;
                return result;
            }
            result.inlier_count = static_cast<std::size_t>(cv::countNonZero(inlier_mask));
            result.status = result.inlier_count >= parameters_.minimum_inliers
                                ? EGeometricVerificationStatus::accepted
                                : EGeometricVerificationStatus::ransac_rejected;

            if (result.accepted())
            {
                // Expose the fixed support set; leave final measurement fitting to the caller.
                result.inlier_index_pairs.reserve(result.inlier_count);
                const auto *mask_values = inlier_mask.ptr<uchar>();

                for (std::size_t index = 0; index < matches.size(); ++index)
                {
                    if (mask_values[index] != 0)
                    {
                        result.inlier_index_pairs.emplace_back(
                            matches[index].reference_index, matches[index].query_index);
                    }
                }
            }

            return result;
        }

      private:
        struct SDescriptorMatch
        {
            std::size_t reference_index = 0;
            std::size_t query_index = 0;
            double distance = 0.0;
        };

        void validateParameters() const
        {
            // Validate matching and RANSAC contracts before processing any frame data.
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

        static void validateFrame(
            const std::span<const cv::Point2d> keypoints,
            const std::span<const Descriptor> descriptors,
            const char *frame_name)
        {
            if (keypoints.size() != descriptors.size())
            {
                throw std::invalid_argument(std::string(frame_name) +
                                            " keypoint and descriptor counts differ.");
            }

            // Reject nonfinite image coordinates before invoking OpenCV geometry.
            for (const cv::Point2d &keypoint : keypoints)
            {
                if (!std::isfinite(keypoint.x) || !std::isfinite(keypoint.y))
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
        [[nodiscard]] std::vector<SDescriptorMatch> matchUnique(
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

            // Scan references in index order; keep the first on equal distances.
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

                // Apply the ratio gate; keep the first query on equal collision distances.
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
        [[nodiscard]] static bool isDegenerate(const cv::Mat &fundamental_matrix)
        {
            if (fundamental_matrix.empty() || fundamental_matrix.rows != 3 ||
                fundamental_matrix.cols != 3 || fundamental_matrix.type() != CV_64FC1 ||
                !cv::checkRange(fundamental_matrix))
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
