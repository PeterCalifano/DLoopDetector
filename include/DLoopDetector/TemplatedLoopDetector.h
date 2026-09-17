/**
 * @file TemplatedLoopDetector.h
 * @brief Compatibility detector composed from split retrieval and geometry components.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See the DLoopDetector LICENSE.txt file.
 */

#pragma once

#include "CBowCandidateRetriever.h"
#include "CKnnFundamentalMatrixVerifier.h"
#include "LoopDetectionContracts.h"

#include <DBoW2/DescriptorPolicy.h>
#include <DBoW2/TemplatedDatabase.h>
#include <DBoW2/TemplatedVocabulary.h>

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace DLoopDetector
{
    /** @brief Geometric-check selection retained by the compatibility API. */
    enum GeometricalCheck : int
    {
        GEOM_EXHAUSTIVE, ///< Use the unified deterministic policy KNN verifier.
        GEOM_DI,         ///< Compatibility name mapped to the policy KNN verifier.
        GEOM_FLANN,      ///< Compatibility name mapped to the policy KNN verifier.
        GEOM_NONE        ///< Accept candidate-ready retrievals without geometry.
    };

    /** @brief Detection outcomes retained for existing DLoopDetector callers. */
    enum DetectionStatus : int
    {
        LOOP_DETECTED,                 ///< Retrieval and enabled geometry accepted.
        CLOSE_MATCHES_ONLY,            ///< Only locally excluded entries exist.
        NO_DB_RESULTS,                 ///< No eligible database result exists.
        LOW_NSS_FACTOR,                ///< Retained historical status; not newly emitted.
        LOW_SCORES,                    ///< Results failed the configured score threshold.
        NO_GROUPS,                     ///< No temporal island met its size gate.
        NO_TEMPORAL_CONSISTENCY,       ///< Candidate island lacked temporal support.
        NO_GEOMETRICAL_CONSISTENCY,    ///< A valid model failed the inlier gate.
        INSUFFICIENT_FEATURES,         ///< Descriptors or unique matches were insufficient.
        DEGENERATE_GEOMETRY            ///< Geometry did not define a usable model.
    };

    /** @brief Compatibility result using dense sequential DBoW entry IDs. */
    struct DetectionResult
    {
        DetectionStatus status = NO_DB_RESULTS; ///< Expected detector outcome.
        DBoW2::EntryId query = 0; ///< Dense sequential query entry.
        /// Dense matched entry, or the maximum ID when no candidate is available.
        DBoW2::EntryId match = std::numeric_limits<DBoW2::EntryId>::max();

        /**
         * @brief Return true only when both retrieval and geometry accept the loop.
         * @return True only for LOOP_DETECTED.
         */
        [[nodiscard]] bool detection() const noexcept
        {
            return status == LOOP_DETECTED;
        }
    };

    /**
     * @brief Sequential loop detector preserving the historical entry point.
     *
     * The former monolithic implementation is now an adapter: BoW and temporal state belong to
     * CBowCandidateRetriever, while descriptor matching and fundamental-matrix RANSAC belong to
     * CKnnFundamentalMatrixVerifier. DI, FLANN, and exhaustive modes use the same deterministic
     * policy KNN verifier; GEOM_NONE explicitly bypasses geometry.
     *
     * @tparam TPolicy Local descriptor policy selected at compile time.
     */
    template <DBoW2::DescriptorPolicy TPolicy>
    class TemplatedLoopDetector
    {
      public:
        using Policy = TPolicy;
        using Descriptor = typename TPolicy::Descriptor;
        using Vocabulary = DBoW2::TemplatedVocabulary<TPolicy>;
        using Database = DBoW2::TemplatedDatabase<TPolicy>;

        /** @brief Historical detector parameters mapped onto the split components. */
        struct Parameters
        {
            int image_rows = 0; ///< Retained historical image height; not consumed.
            int image_cols = 0; ///< Retained historical image width; not consumed.
            bool use_nss = true; ///< Retained NSS switch; scoring uses the absolute alpha gate.
            float alpha = 0.3F; ///< Inclusive normalized BoW score threshold.
            int k = 4; ///< Required consecutive supporting candidate islands.
            GeometricalCheck geom_check = GEOM_DI; ///< Geometry mode or explicit bypass.
            int di_levels = 0; ///< Retained direct-index level; unified matching does not use DI.
            int dislocal = 20; ///< Most recent dense entries excluded from retrieval.
            int max_db_results = 50; ///< Maximum ranked results before filtering.
            /// Historical NSS-relative gate retained for source compatibility; not consumed.
            float min_nss_factor = 0.005F;
            int min_matches_per_group = 1; ///< Minimum results in one temporal island.
            int max_intragroup_gap = 3; ///< Maximum dense-entry gap within an island.
            int max_distance_between_groups = 3; ///< Candidate-island temporal gap.
            int max_distance_between_queries = 2; ///< Consecutive query-entry gap.
            int min_Fpoints = 12; ///< Minimum RANSAC inliers for loop acceptance.
            int max_ransac_iterations = 500; ///< Positive RANSAC iteration cap.
            double ransac_probability = 0.99; ///< RANSAC model confidence.
            double max_reprojection_error = 2.0; ///< RANSAC pixel-error threshold.
            double max_neighbor_ratio = 0.6; ///< Historical strict best/second-best match ratio.

            /** @brief Construct default parameters for a one-hertz stream. */
            Parameters() = default;

            /**
             * @brief Construct parameters and scale temporal distances by stream frequency.
             * @param height Expected image height.
             * @param width Expected image width.
             * @param frequency Expected positive frame frequency in hertz.
             * @param nss Retained normalized-score compatibility flag.
             * @param alpha_threshold Minimum BoW score.
             * @param temporal_consistency Required consecutive candidate islands.
             * @param geometry Geometric verification mode.
             * @param direct_index_levels Retained compatibility value.
             */
            Parameters(const int height, const int width, const float frequency = 1.0F,
                       const bool nss = true, const float alpha_threshold = 0.3F,
                       const int temporal_consistency = 3,
                       const GeometricalCheck geometry = GEOM_DI,
                       const int direct_index_levels = 0)
                : image_rows(height), image_cols(width), use_nss(nss),
                  alpha(alpha_threshold), k(temporal_consistency), geom_check(geometry),
                  di_levels(direct_index_levels)
            {
                constexpr double maximum_frequency =
                    static_cast<double>((std::numeric_limits<int>::max)()) / 50.0;
                if (!(frequency > 0.0F) || !std::isfinite(frequency) ||
                    static_cast<double>(frequency) > maximum_frequency)
                {
                    throw std::invalid_argument(
                        "Loop-detector frequency must be finite, positive, and representable.");
                }

                // Scale legacy temporal gates to the requested frame frequency.
                const double frequency_value = static_cast<double>(frequency);
                dislocal = std::max(0, static_cast<int>(20.0 * frequency_value));
                max_db_results = std::max(1, static_cast<int>(50.0 * frequency_value));
                min_matches_per_group = std::max(1, static_cast<int>(frequency_value));
                max_intragroup_gap = std::max(0, static_cast<int>(3.0 * frequency_value));
                max_distance_between_groups =
                    std::max(0, static_cast<int>(3.0 * frequency_value));
                max_distance_between_queries =
                    std::max(1, static_cast<int>(2.0 * frequency_value));
            }
        };

        /**
         * @brief Construct an unconfigured detector; set a vocabulary before detection.
         * @param parameters Compatibility parameters mapped to the split components.
         * @throws std::invalid_argument If a retrieval or geometry gate is invalid.
         */
        explicit TemplatedLoopDetector(const Parameters &parameters = {})
            : parameters_(parameters), verifier_(MakeGeometryParameters(parameters))
        {
        }

        /**
         * @brief Construct a detector from a trained vocabulary.
         * @param vocabulary Non-empty trained vocabulary copied into the detector.
         * @param parameters Compatibility parameters mapped to the split components.
         * @throws std::invalid_argument If the vocabulary or any mapped gate is invalid.
         */
        explicit TemplatedLoopDetector(const Vocabulary &vocabulary,
                                       const Parameters &parameters = {})
            : parameters_(parameters), verifier_(MakeGeometryParameters(parameters))
        {
            SetVocabulary(vocabulary);
        }

        /**
         * @brief Construct a detector from a database's vocabulary and clear its entries.
         * @param database Database supplying a non-null, non-empty vocabulary.
         * @param parameters Compatibility parameters mapped to the split components.
         * @throws std::invalid_argument If the vocabulary pointer or mapped state is invalid.
         */
        explicit TemplatedLoopDetector(const Database &database,
                                       const Parameters &parameters = {})
            : TemplatedLoopDetector(DatabaseVocabulary(database), parameters)
        {
        }

        /**
         * @brief Construct from a database-compatible type exposing getVocabulary().
         * @tparam TDatabase Type whose vocabulary pointer is convertible to this policy's type.
         * @param database Database-like object supplying a non-null, non-empty vocabulary.
         * @param parameters Compatibility parameters mapped to the split components.
         * @throws std::invalid_argument If the vocabulary pointer or mapped state is invalid.
         */
        template <typename TDatabase>
            requires requires(const TDatabase &candidate) {
                { candidate.getVocabulary() } -> std::convertible_to<const Vocabulary *>;
            }
        explicit TemplatedLoopDetector(const TDatabase &database,
                                       const Parameters &parameters = {})
            : TemplatedLoopDetector(DatabaseVocabulary(database), parameters)
        {
        }

        /**
         * @brief Return the owned database.
         * @return Const reference valid until the detector is destroyed or reconfigured.
         * @throws std::logic_error If the detector is unconfigured.
         */
        [[nodiscard]] const Database &getDatabase() const
        {
            RequireConfigured();
            return retriever_->GetDatabase();
        }

        /**
         * @brief Return the vocabulary used by the owned database.
         * @return Const reference valid until the detector is destroyed or reconfigured.
         * @throws std::logic_error If the detector is unconfigured.
         */
        [[nodiscard]] const Vocabulary &getVocabulary() const
        {
            return *getDatabase().getVocabulary();
        }

        /**
         * @brief Replace the database vocabulary and clear all detector state.
         * @tparam TDatabase Type whose vocabulary pointer is convertible to this policy's type.
         * @param database Database-like object supplying a non-null, non-empty vocabulary.
         * @throws std::invalid_argument If the vocabulary pointer or mapped state is invalid.
         */
        template <typename TDatabase>
            requires requires(const TDatabase &candidate) {
                { candidate.getVocabulary() } -> std::convertible_to<const Vocabulary *>;
            }
        void setDatabase(const TDatabase &database)
        {
            SetVocabulary(DatabaseVocabulary(database));
        }

        /**
         * @brief Replace the vocabulary and clear all detector state.
         * @param vocabulary Non-empty trained vocabulary copied into the detector.
         * @throws std::invalid_argument If the vocabulary or mapped retrieval state is invalid.
         */
        void setVocabulary(const Vocabulary &vocabulary)
        {
            SetVocabulary(vocabulary);
        }

        /**
         * @brief Reserve frame history storage for predictable streaming allocations.
         * @param number_of_entries Expected number of sequential frames.
         * @param expected_keypoints Retained compatibility hint; owned frames use their exact size.
         * @throws std::invalid_argument If either count is negative.
         */
        void allocate(const int number_of_entries, const int expected_keypoints = 0)
        {
            if (number_of_entries < 0 || expected_keypoints < 0)
            {
                throw std::invalid_argument("Loop-detector allocation counts cannot be negative.");
            }

            // Reserve both histories together to preserve their index alignment.
            keypoint_history_.reserve(static_cast<std::size_t>(number_of_entries));
            descriptor_history_.reserve(static_cast<std::size_t>(number_of_entries));
        }

        /**
         * @brief Query, verify, and insert one sequential local-feature frame.
         * @param keypoints Original-image keypoints aligned with descriptors.
         * @param descriptors Local descriptors for the current frame.
         * @param result Output compatibility result.
         * @return True when a loop passes retrieval, temporal, and geometry gates.
         * @throws std::invalid_argument If counts differ or any frame value violates its contract.
         * @throws std::logic_error If no vocabulary has been configured.
         */
        bool detectLoop(const std::vector<cv::KeyPoint> &keypoints,
                        const std::vector<Descriptor> &descriptors,
                        DetectionResult &result)
        {
            RequireConfigured();

            // Reject malformed frame alignment before cloning any caller-owned data.
            if (keypoints.size() != descriptors.size())
            {
                throw std::invalid_argument("Loop frame keypoint and descriptor counts differ.");
            }

            for (const cv::KeyPoint &keypoint : keypoints)
            {
                if (!std::isfinite(keypoint.pt.x) || !std::isfinite(keypoint.pt.y))
                {
                    throw std::invalid_argument(
                        "Loop frame keypoints contain nonfinite coordinates.");
                }
            }

            // Clone the frame before retrieval so later geometry never aliases caller storage.
            std::vector<cv::KeyPoint> owned_keypoints(keypoints);
            std::vector<Descriptor> owned_descriptors;
            owned_descriptors.reserve(descriptors.size());

            for (const Descriptor &descriptor : descriptors)
            {
                owned_descriptors.push_back(TPolicy::Clone(descriptor));
            }

            // Reserve both histories before retrieval to keep insertion completion non-throwing.
            ReserveHistoryForNextFrame();

            // Derive the compatibility entry ID from the aligned sequential history.
            const FrameId query_frame_id = static_cast<FrameId>(descriptor_history_.size());
            result = DetectionResult{};
            result.query = CheckedEntryId(query_frame_id);

            // Query and insert through the retriever before translating its public result.
            const SCandidateRetrievalResult retrieval =
                retriever_->Process(query_frame_id, descriptors);

            if (retrieval.candidate_frame_id)
            {
                result.match = CheckedEntryId(*retrieval.candidate_frame_id);
            }

            result.status = MapRetrievalStatus(retrieval.status);

            // Append both histories after retrieval inserts the corresponding database entry.
            if (retrieval.inserted)
            {
                keypoint_history_.push_back(std::move(owned_keypoints));
                descriptor_history_.push_back(std::move(owned_descriptors));
            }

            // Run geometry only after retrieval satisfies score, island, and temporal gates.
            if (retrieval.status == ERetrievalStatus::candidate_ready)
            {
                result.status = VerifyCandidate(
                    *retrieval.candidate_frame_id, keypoint_history_.back(),
                    descriptor_history_.back());
            }

            return result.detection();
        }

        /** @brief Clear retrieval, temporal, and frame-history state. */
        void clear()
        {
            if (retriever_)
            {
                retriever_->Clear();
            }

            keypoint_history_.clear();
            descriptor_history_.clear();
        }

      private:
        template <typename TFrame>
        static void ReserveForAppend(std::vector<TFrame> &history)
        {
            if (history.size() < history.capacity())
            {
                return;
            }

            const std::size_t maximum_size = history.max_size();
            if (history.size() == maximum_size)
            {
                throw std::length_error("Loop-detector frame history is full.");
            }

            // Grow geometrically so streaming does not reallocate the history on every frame.
            const std::size_t next_capacity = history.size() > maximum_size / 2U
                                                  ? maximum_size
                                                  : std::max<std::size_t>(1U, history.size() * 2U);
            history.reserve(next_capacity);
        }

        void ReserveHistoryForNextFrame()
        {
            ReserveForAppend(keypoint_history_);
            ReserveForAppend(descriptor_history_);
        }

        template <typename TDatabase>
        [[nodiscard]] static const Vocabulary &DatabaseVocabulary(
            const TDatabase &database)
        {
            const Vocabulary *vocabulary = database.getVocabulary();
            if (vocabulary == nullptr)
            {
                throw std::invalid_argument("Loop-detector database has no vocabulary.");
            }

            return *vocabulary;
        }

        /**
         * @brief Map historical retrieval settings to the split retriever contract.
         * @param parameters Legacy detector parameters supplied by the caller.
         * @return Mapped retrieval parameters ready for retriever validation.
         * @throws std::invalid_argument If a signed count or distance is outside its domain.
         */
        static SCandidateRetrievalParameters MakeRetrievalParameters(
            const Parameters &parameters)
        {
            // Reject invalid signed legacy values before converting them to size types.
            if (parameters.dislocal < 0 || parameters.max_db_results <= 0 ||
                parameters.min_matches_per_group <= 0 || parameters.k <= 0 ||
                parameters.max_intragroup_gap < 0 ||
                parameters.max_distance_between_groups < 0 ||
                parameters.max_distance_between_queries <= 0)
            {
                throw std::invalid_argument("Invalid legacy retrieval parameters.");
            }

            SCandidateRetrievalParameters mapped;
            mapped.local_exclusion = static_cast<std::size_t>(parameters.dislocal);
            mapped.maximum_database_results =
                static_cast<std::size_t>(parameters.max_db_results);
            mapped.minimum_score = parameters.alpha;
            mapped.minimum_matches_per_island =
                static_cast<std::size_t>(parameters.min_matches_per_group);
            mapped.maximum_intragroup_gap =
                static_cast<std::size_t>(parameters.max_intragroup_gap);
            mapped.required_temporal_consistency = static_cast<std::size_t>(parameters.k);
            mapped.maximum_temporal_query_gap =
                static_cast<std::size_t>(parameters.max_distance_between_queries);
            mapped.maximum_temporal_candidate_gap =
                static_cast<std::size_t>(parameters.max_distance_between_groups);
            return mapped;
        }

        /**
         * @brief Map historical geometry settings to the unified verifier contract.
         * @param parameters Legacy detector parameters supplied by the caller.
         * @return Mapped matching and RANSAC parameters ready for verifier validation.
         * @throws std::invalid_argument If the geometry mode or inlier gate is invalid.
         */
        static SGeometricVerificationParameters MakeGeometryParameters(
            const Parameters &parameters)
        {
            // Accept historical mode names while mapping enabled modes to one verifier.
            switch (parameters.geom_check)
            {
                case GEOM_EXHAUSTIVE:
                case GEOM_DI:
                case GEOM_FLANN:
                case GEOM_NONE:
                    break;
                default:
                    throw std::invalid_argument("Unknown legacy geometric-check mode.");
            }

            if (parameters.min_Fpoints < 1)
            {
                throw std::invalid_argument("min_Fpoints must be positive.");
            }

            SGeometricVerificationParameters mapped;
            mapped.maximum_neighbor_ratio = parameters.max_neighbor_ratio;
            mapped.minimum_correspondences = 8;
            mapped.minimum_inliers = static_cast<std::size_t>(parameters.min_Fpoints);
            mapped.maximum_reprojection_error = parameters.max_reprojection_error;
            mapped.ransac_confidence = parameters.ransac_probability;
            mapped.maximum_ransac_iterations = parameters.max_ransac_iterations;
            return mapped;
        }

        void SetVocabulary(const Vocabulary &vocabulary)
        {
            // Construct the replacement retriever before discarding aligned frame history.
            retriever_ = std::make_unique<CBowCandidateRetriever<TPolicy>>(
                vocabulary, MakeRetrievalParameters(parameters_));

            keypoint_history_.clear();
            descriptor_history_.clear();
        }

        void RequireConfigured() const
        {
            if (!retriever_)
            {
                throw std::logic_error("Loop detector has no configured vocabulary.");
            }
        }

        /**
         * @brief Verify one retrieved candidate and map its outcome to legacy status values.
         * @param candidate_frame_id Dense candidate index into the aligned frame histories.
         * @param query_keypoints Current frame keypoints in original-image coordinates.
         * @param query_descriptors Current frame descriptors aligned with the keypoints.
         * @return Historical detection status for the geometry outcome.
         * @throws std::logic_error If retrieval and retained frame history are inconsistent.
         */
        [[nodiscard]] DetectionStatus VerifyCandidate(
            const FrameId candidate_frame_id,
            const std::vector<cv::KeyPoint> &query_keypoints,
            const std::vector<Descriptor> &query_descriptors) const
        {
            // Bypass geometry only when the compatibility API explicitly disables it.
            if (parameters_.geom_check == GEOM_NONE)
            {
                return LOOP_DETECTED;
            }

            // Reject any retrieval/history divergence before indexing the stored frame.
            if (candidate_frame_id >= keypoint_history_.size())
            {
                throw std::logic_error("Retrieved candidate has no aligned frame history.");
            }

            // Translate the explicit verifier outcome into the historical status vocabulary.
            const SGeometricVerificationResult geometry = verifier_.verify(
                keypoint_history_[candidate_frame_id],
                descriptor_history_[candidate_frame_id], query_keypoints, query_descriptors);

            switch (geometry.status)
            {
                case EGeometricVerificationStatus::accepted:
                    return LOOP_DETECTED;
                case EGeometricVerificationStatus::insufficient_features:
                case EGeometricVerificationStatus::insufficient_matches:
                    return INSUFFICIENT_FEATURES;
                case EGeometricVerificationStatus::degenerate_configuration:
                    return DEGENERATE_GEOMETRY;
                case EGeometricVerificationStatus::ransac_rejected:
                    return NO_GEOMETRICAL_CONSISTENCY;
            }

            throw std::logic_error("Unhandled geometric verification status.");
        }

        [[nodiscard]] static DetectionStatus MapRetrievalStatus(
            const ERetrievalStatus status)
        {
            switch (status)
            {
                case ERetrievalStatus::candidate_ready:
                    return NO_GEOMETRICAL_CONSISTENCY;
                case ERetrievalStatus::local_exclusion:
                    return CLOSE_MATCHES_ONLY;
                case ERetrievalStatus::no_database_results:
                    return NO_DB_RESULTS;
                case ERetrievalStatus::below_score_threshold:
                    return LOW_SCORES;
                case ERetrievalStatus::no_islands:
                    return NO_GROUPS;
                case ERetrievalStatus::temporal_inconsistency:
                    return NO_TEMPORAL_CONSISTENCY;
                case ERetrievalStatus::duplicate_frame:
                    return NO_DB_RESULTS;
                case ERetrievalStatus::insufficient_descriptors:
                    return INSUFFICIENT_FEATURES;
            }

            throw std::logic_error("Unhandled candidate retrieval status.");
        }

        [[nodiscard]] static DBoW2::EntryId CheckedEntryId(const FrameId frame_id)
        {
            if (frame_id > std::numeric_limits<DBoW2::EntryId>::max())
            {
                throw std::overflow_error("Sequential frame ID exceeds DBoW EntryId capacity.");
            }

            return static_cast<DBoW2::EntryId>(frame_id);
        }

        Parameters parameters_;
        std::unique_ptr<CBowCandidateRetriever<TPolicy>> retriever_;
        CKnnFundamentalMatrixVerifier<TPolicy> verifier_;
        std::vector<std::vector<cv::KeyPoint>> keypoint_history_;
        std::vector<std::vector<Descriptor>> descriptor_history_;
    };
} // namespace DLoopDetector
