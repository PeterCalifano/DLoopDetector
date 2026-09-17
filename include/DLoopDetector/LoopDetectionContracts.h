/**
 * @file LoopDetectionContracts.h
 * @brief C++20 contracts and result types for split loop detection.
 * @author Pietro Califano, and Codex GPT-5.6
 * @date 2026-09-16
 * @copyright See the DLoopDetector LICENSE.txt file.
 */

#pragma once

#include <DBoW2/DescriptorPolicy.h>

#include <opencv2/features2d.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace DLoopDetector
{
    /** @brief Stable identifier supplied by the calling application. */
    using FrameId = std::uint64_t;

    /** @brief Outcome of candidate retrieval before geometric verification. */
    enum class ERetrievalStatus
    {
        candidate_ready,          ///< Candidate passed BoW, island, and temporal gates.
        local_exclusion,          ///< No old entry lies outside the local exclusion window.
        no_database_results,      ///< No eligible entry shares a non-stopped word.
        below_score_threshold,    ///< All database results fell below the score gate.
        no_islands,               ///< No result island met its minimum match count.
        temporal_inconsistency,   ///< Candidate island lacks required temporal support.
        duplicate_frame,          ///< External frame ID has already been processed.
        insufficient_descriptors ///< Descriptor batch was empty or gated by a composer.
    };

    /** @brief Candidate retrieval result and insertion decision for one frame. */
    struct SCandidateRetrievalResult
    {
        ERetrievalStatus status = ERetrievalStatus::no_database_results; ///< Expected outcome.
        FrameId query_frame_id = 0; ///< Application-owned query identifier.
        std::optional<FrameId> candidate_frame_id; ///< Best external candidate, when available.
        double candidate_score = 0.0; ///< Normalized [0, 1] similarity of the best candidate.
        bool inserted = false; ///< Whether the query was added after retrieval.
    };

    /** @brief Expected algorithmic outcomes of local geometric verification. */
    enum class EGeometricVerificationStatus
    {
        accepted,                 ///< Fundamental model passed the inlier gate.
        insufficient_features,   ///< Either frame cannot supply the minimum inputs.
        insufficient_matches,    ///< Ratio/uniqueness filtering left too few matches.
        degenerate_configuration, ///< Fundamental matrix is missing, malformed, or rank deficient.
        ransac_rejected           ///< Valid model has fewer than the required inliers.
    };

    /** @brief Geometric verification outcome with diagnostic match counts. */
    struct SGeometricVerificationResult
    {
        /// Expected verification outcome.
        EGeometricVerificationStatus status = EGeometricVerificationStatus::insufficient_features;
        std::size_t correspondence_count = 0; ///< Matches entering RANSAC.
        std::size_t inlier_count = 0; ///< RANSAC support count, including rejected loops.
        /// RANSAC inlier descriptor indices, ordered by reference index on acceptance only.
        std::vector<std::pair<std::size_t, std::size_t>> inlier_index_pairs;

        /**
         * @brief Return true only when the RANSAC model passes the inlier gate.
         * @return True for an accepted model; false for every rejection status.
         */
        [[nodiscard]] bool accepted() const noexcept
        {
            return status == EGeometricVerificationStatus::accepted;
        }
    };

    /**
     * @brief Stateful candidate-retrieval contract.
     * @tparam TRetriever Candidate retriever implementation.
     * @tparam TPolicy Descriptor policy consumed by the retriever.
     */
    template <typename TRetriever, typename TPolicy>
    concept CandidateRetriever = DBoW2::DescriptorPolicy<TPolicy> && requires(
        TRetriever &retriever,
        const TRetriever &const_retriever,
        FrameId frame_id,
        std::span<const typename TPolicy::Descriptor> descriptors)
    {
        { retriever.Process(frame_id, descriptors) } ->
            std::same_as<SCandidateRetrievalResult>;
        { retriever.Clear() } -> std::same_as<void>;
        { const_retriever.Size() } -> std::convertible_to<std::size_t>;
    };

    /**
     * @brief Stateless local-geometry verification contract.
     * @tparam TVerifier Geometric verifier implementation.
     * @tparam TPolicy Descriptor policy consumed by the verifier.
     */
    template <typename TVerifier, typename TPolicy>
    concept GeometricVerifier = DBoW2::DescriptorPolicy<TPolicy> && requires(
        const TVerifier &verifier,
        std::span<const cv::KeyPoint> keypoints,
        std::span<const cv::Point2d> points,
        std::span<const typename TPolicy::Descriptor> descriptors)
    {
        { verifier.verify(keypoints, descriptors, keypoints, descriptors) } ->
            std::same_as<SGeometricVerificationResult>;
        { verifier.verifyPoints(points, descriptors, points, descriptors) } ->
            std::same_as<SGeometricVerificationResult>;
    };
} // namespace DLoopDetector
