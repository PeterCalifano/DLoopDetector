/**
 * @file CBowCandidateRetriever.h
 * @brief Stateful DBoW candidate retrieval with temporal-island filtering.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See the DLoopDetector LICENSE.txt file.
 */

#pragma once

#include "LoopDetectionContracts.h"

#include <DBoW2/QueryResults.h>
#include <DBoW2/TemplatedDatabase.h>
#include <DBoW2/TemplatedVocabulary.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace DLoopDetector
{
    /** @brief Tunable gates for BoW retrieval and temporal consistency. */
    struct SCandidateRetrievalParameters
    {
        std::size_t local_exclusion = 20; ///< Most recent dense entries excluded from queries.
        std::size_t maximum_database_results = 50; ///< Ranked results retained before filtering.
        double minimum_score = 0.3; ///< Inclusive normalized BoW score gate.
        std::size_t minimum_matches_per_island = 1; ///< Smallest accepted temporal island.
        std::size_t maximum_intragroup_gap = 3; ///< Largest entry gap within one island.
        std::size_t required_temporal_consistency = 1; ///< Consecutive supporting queries.
        std::size_t maximum_temporal_query_gap = 2; ///< Largest consecutive query-entry gap.
        std::size_t maximum_temporal_candidate_gap = 3; ///< Largest candidate-island gap.
    };

    /**
     * @brief Own a DBoW database and return temporally consistent loop candidates.
     *
     * External frame identifiers are mapped to the database's dense internal entry IDs. A
     * well-formed frame is inserted after its query is evaluated; duplicate identifiers and empty
     * descriptor batches do not mutate any state.
     *
     * @tparam TPolicy Descriptor policy used by the vocabulary and database.
     */
    template <DBoW2::DescriptorPolicy TPolicy>
    class CBowCandidateRetriever
    {
      public:
        using Policy = TPolicy;
        using Descriptor = typename TPolicy::Descriptor;
        using Vocabulary = DBoW2::TemplatedVocabulary<TPolicy>;
        using Database = DBoW2::TemplatedDatabase<TPolicy>;

        /**
         * @brief Construct a retriever from a trained vocabulary.
         * @param vocabulary Non-empty vocabulary copied into the owned database.
         * @param parameters Retrieval and temporal gates.
         * @throws std::invalid_argument If the vocabulary or parameters are invalid.
         */
        explicit CBowCandidateRetriever(
            const Vocabulary &vocabulary,
            const SCandidateRetrievalParameters &parameters = {})
            : database_(vocabulary, false), parameters_(parameters)
        {
            if (vocabulary.empty())
            {
                throw std::invalid_argument("Candidate retrieval requires a trained vocabulary.");
            }

            ValidateParameters();
        }

        /**
         * @brief Query old entries, update temporal state, then insert the current frame.
         * @param frame_id Application-owned frame identifier.
         * @param descriptors Non-empty descriptor batch for the current frame.
         * @return Retrieval status, optional external candidate ID, score, and insertion decision.
         * @throws std::invalid_argument If a descriptor violates its policy contract.
         */
        [[nodiscard]] SCandidateRetrievalResult Process(
            const FrameId frame_id,
            const std::span<const Descriptor> descriptors)
        {
            SCandidateRetrievalResult result;
            result.query_frame_id = frame_id;

            // Reject frames that cannot participate before mutating retrieval state.
            if (external_frame_ids_.contains(frame_id))
            {
                result.status = ERetrievalStatus::duplicate_frame;
                return result;
            }

            if (descriptors.empty())
            {
                result.status = ERetrievalStatus::insufficient_descriptors;
                return result;
            }

            // Retain valid frames until the local-exclusion window admits an older candidate.
            const std::size_t query_entry = entry_to_external_frame_.size();
            if (query_entry <= parameters_.local_exclusion)
            {
                result.status = ERetrievalStatus::local_exclusion;
                ResetTemporalWindow();
                Insert(frame_id, descriptors, result);
                return result;
            }

            // Apply DBoW's exclusive upper bound to query only entries outside local exclusion.
            const std::size_t eligible_count = query_entry - parameters_.local_exclusion;
            if (eligible_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw std::overflow_error("Candidate database exceeds DBoW query ID capacity.");
            }

            // Query the bounded database prefix before inserting the current frame.
            DBoW2::QueryResults query_results;
            database_.query(
                descriptors, query_results,
                static_cast<int>(parameters_.maximum_database_results),
                static_cast<int>(eligible_count));

            if (query_results.empty())
            {
                result.status = ERetrievalStatus::no_database_results;
                ResetTemporalWindow();
                Insert(frame_id, descriptors, result);
                return result;
            }

            // Normalize and filter every scoring mode through the public similarity contract.
            NormalizeQueryScores(query_results);
            std::erase_if(query_results, [this](const DBoW2::Result &candidate) {
                return candidate.Score < parameters_.minimum_score;
            });

            if (query_results.empty())
            {
                result.status = ERetrievalStatus::below_score_threshold;
                ResetTemporalWindow();
                Insert(frame_id, descriptors, result);
                return result;
            }

            // Group surviving candidates into temporal islands and retain the strongest island.
            const std::vector<SIsland> islands = ComputeIslands(std::move(query_results));
            if (islands.empty())
            {
                result.status = ERetrievalStatus::no_islands;
                ResetTemporalWindow();
                Insert(frame_id, descriptors, result);
                return result;
            }

            const SIsland &best_island = *std::max_element(
                islands.begin(), islands.end(),
                [](const SIsland &first, const SIsland &second) {
                    return first.aggregate_score < second.aggregate_score;
                });

            result.candidate_frame_id = ExternalFrameId(best_island.best_entry);
            result.candidate_score = best_island.best_score;

            // Update temporal support before reporting and inserting the valid query frame.
            UpdateTemporalWindow(best_island, query_entry);
            result.status = temporal_consistency_ >=
                                    parameters_.required_temporal_consistency
                                ? ERetrievalStatus::candidate_ready
                                : ERetrievalStatus::temporal_inconsistency;

            Insert(frame_id, descriptors, result);
            return result;
        }

        /** @brief Clear the database, frame-ID mapping, and temporal state. */
        void Clear()
        {
            database_.clear();
            entry_to_external_frame_.clear();
            external_frame_ids_.clear();
            ResetTemporalWindow();
        }

        /**
         * @brief Return the number of inserted frames.
         * @return Number of dense database entries and external frame-ID mappings.
         */
        [[nodiscard]] std::size_t Size() const noexcept
        {
            return entry_to_external_frame_.size();
        }

        /**
         * @brief Return the owned database for diagnostics and persistence.
         * @return Const reference valid for the retriever's lifetime.
         */
        [[nodiscard]] const Database &GetDatabase() const noexcept
        {
            return database_;
        }

      private:
        struct SIsland
        {
            DBoW2::EntryId first = 0;
            DBoW2::EntryId last = 0;
            DBoW2::EntryId best_entry = 0;
            double aggregate_score = 0.0;
            double best_score = 0.0;
            std::size_t match_count = 0;
        };

        void ValidateParameters() const
        {
            if (parameters_.maximum_database_results == 0 ||
                parameters_.maximum_database_results >
                    static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument("maximum_database_results must fit a positive int.");
            }

            if (!std::isfinite(parameters_.minimum_score) ||
                parameters_.minimum_score < 0.0 || parameters_.minimum_score > 1.0)
            {
                throw std::invalid_argument("minimum_score must be in [0, 1].");
            }

            if (parameters_.minimum_matches_per_island == 0 ||
                parameters_.required_temporal_consistency == 0)
            {
                throw std::invalid_argument("Island and temporal counts must be positive.");
            }
        }

        /**
         * @brief Group temporally adjacent retrieval results into candidate islands.
         * @param query_results Query results consumed and reordered by dense database entry ID.
         * @return Islands that satisfy the configured minimum match count.
         */
        [[nodiscard]] std::vector<SIsland> ComputeIslands(
            DBoW2::QueryResults query_results) const
        {
            // Sort candidates by dense entry ID so temporal neighbors become contiguous.
            std::sort(query_results.begin(), query_results.end(), DBoW2::Result::ltId);

            std::vector<SIsland> islands;
            SIsland current;
            bool has_current = false;

            // Accumulate adjacent candidates and close each island at the configured gap.
            for (const DBoW2::Result &candidate : query_results)
            {
                const bool starts_new_island =
                    has_current &&
                    static_cast<std::size_t>(candidate.Id - current.last) >
                        parameters_.maximum_intragroup_gap;
                if (starts_new_island)
                {
                    AppendIfLargeEnough(current, islands);
                    has_current = false;
                }

                if (!has_current)
                {
                    current = SIsland{candidate.Id, candidate.Id, candidate.Id,
                                      candidate.Score, candidate.Score, 1};
                    has_current = true;
                    continue;
                }

                current.last = candidate.Id;
                current.aggregate_score += candidate.Score;
                ++current.match_count;

                if (candidate.Score > current.best_score)
                {
                    current.best_score = candidate.Score;
                    current.best_entry = candidate.Id;
                }
            }

            if (has_current)
            {
                AppendIfLargeEnough(current, islands);
            }

            return islands;
        }

        void AppendIfLargeEnough(const SIsland &island,
                                 std::vector<SIsland> &islands) const
        {
            if (island.match_count >= parameters_.minimum_matches_per_island)
            {
                islands.push_back(island);
            }
        }

        /**
         * @brief Update consecutive-query support for the selected candidate island.
         * @param island Strongest island for the current query.
         * @param query_entry Dense database position assigned to the current query.
         */
        void UpdateTemporalWindow(const SIsland &island, const std::size_t query_entry)
        {
            // Restart support when the current query is not consecutive with the previous one.
            const bool query_is_consecutive = previous_island_.has_value() &&
                query_entry - previous_query_entry_ <=
                    parameters_.maximum_temporal_query_gap;
            if (!query_is_consecutive)
            {
                temporal_consistency_ = 1;
            }
            else
            {
                // Extend support only when consecutive candidate islands overlap or remain close.
                const SIsland &previous = *previous_island_;
                const bool overlaps = island.first <= previous.last &&
                                      previous.first <= island.last;

                std::size_t gap = 0;
                if (island.first > previous.last)
                {
                    gap = island.first - previous.last;
                }
                else if (previous.first > island.last)
                {
                    gap = previous.first - island.last;
                }

                const bool candidate_is_consistent =
                    overlaps || gap <= parameters_.maximum_temporal_candidate_gap;
                temporal_consistency_ = candidate_is_consistent
                                            ? temporal_consistency_ + 1
                                            : 1;
            }

            previous_island_ = island;
            previous_query_entry_ = query_entry;
        }

        void ResetTemporalWindow() noexcept
        {
            previous_island_.reset();
            previous_query_entry_ = 0;
            temporal_consistency_ = 0;
        }

        /**
         * @brief Convert every DBoW scoring mode to the public similarity range.
         * @param query_results Results updated in place while preserving their ordering.
         * @throws std::logic_error If DBoW returns a nonfinite or out-of-domain score.
         */
        void NormalizeQueryScores(DBoW2::QueryResults &query_results) const
        {
            constexpr double score_tolerance =
                64.0 * std::numeric_limits<double>::epsilon();
            const DBoW2::ScoringType scoring =
                database_.getVocabulary()->getScoringType();

            // Bound exceptional modes monotonically; native similarities are already scaled.
            for (DBoW2::Result &candidate : query_results)
            {
                if (!std::isfinite(candidate.Score))
                {
                    throw std::logic_error("DBoW query produced a nonfinite score.");
                }

                if (scoring == DBoW2::KL)
                {
                    if (candidate.Score < -score_tolerance)
                    {
                        throw std::logic_error("DBoW KL divergence became negative.");
                    }

                    candidate.Score = std::exp(-std::max(0.0, candidate.Score));
                }
                else if (scoring == DBoW2::DOT_PRODUCT)
                {
                    if (candidate.Score < 0.0)
                    {
                        throw std::logic_error("DBoW dot-product score became negative.");
                    }

                    candidate.Score /= 1.0 + candidate.Score;
                }
                else
                {
                    if (candidate.Score < -score_tolerance ||
                        candidate.Score > 1.0 + score_tolerance)
                    {
                        throw std::logic_error(
                            "DBoW normalized similarity left the [0, 1] range.");
                    }

                    candidate.Score = std::clamp(candidate.Score, 0.0, 1.0);
                }
            }
        }

        [[nodiscard]] FrameId ExternalFrameId(const DBoW2::EntryId entry) const
        {
            if (entry >= entry_to_external_frame_.size())
            {
                throw std::logic_error("DBoW candidate has no external frame-ID mapping.");
            }

            return entry_to_external_frame_[entry];
        }

        /**
         * @brief Insert one descriptor batch while preserving database and ID-map alignment.
         * @param frame_id Application-owned identifier reserved for the new entry.
         * @param descriptors Descriptor batch added to the owned database.
         * @param result Retrieval result marked as inserted after all state updates succeed.
         * @throws std::logic_error If the frame identifier becomes duplicated during insertion.
         *
         * Roll back both external-ID mappings if database insertion throws, preserving the strong
         * state-consistency guarantee.
         */
        void Insert(const FrameId frame_id,
                    const std::span<const Descriptor> descriptors,
                    SCandidateRetrievalResult &result)
        {
            const auto [frame_position, frame_inserted] =
                external_frame_ids_.insert(frame_id);
            if (!frame_inserted)
            {
                throw std::logic_error("Candidate frame ID became duplicated during insertion.");
            }

            // Prepare both external mappings and roll them back if database insertion fails.
            try
            {
                entry_to_external_frame_.push_back(frame_id);

                try
                {
                    database_.add(descriptors);
                }
                catch (...)
                {
                    entry_to_external_frame_.pop_back();
                    throw;
                }
            }
            catch (...)
            {
                external_frame_ids_.erase(frame_position);
                throw;
            }

            result.inserted = true;
        }

        Database database_;
        SCandidateRetrievalParameters parameters_;
        std::vector<FrameId> entry_to_external_frame_;
        std::unordered_set<FrameId> external_frame_ids_;
        std::optional<SIsland> previous_island_;
        std::size_t previous_query_entry_ = 0;
        std::size_t temporal_consistency_ = 0;
    };

    static_assert(CandidateRetriever<
                  CBowCandidateRetriever<DBoW2::FixedFloatDescriptorPolicy<1>>,
                  DBoW2::FixedFloatDescriptorPolicy<1>>);
} // namespace DLoopDetector
