#include "core/tensor_sharded_ops.h"

namespace astate {

void CheckCandidatesCoverage(
    const std::vector<std::pair<ShardedKey, ATensor>>& candidates,
    const ShardedKey& target_shard,
    const ATensor& target_tensor) {
    // Define interval structure
    struct Interval {
        int64_t start;
        int64_t end;
        bool operator<(const Interval& other) const { return start < other.start; }
    };

    // Convert target to flat intervals
    auto target_intervals = ShardedTensorToFlatIntervals(target_shard, target_tensor);

    // Collect all intervals from all candidates
    std::vector<Interval> all_intervals;
    for (const auto& candidate : candidates) {
        auto candidate_intervals = ShardedTensorToFlatIntervals(candidate.first, candidate.second);

        // Add all intervals from this candidate
        for (const auto& interval : candidate_intervals) {
            all_intervals.push_back({interval.first, interval.second});
        }
    }

    if (all_intervals.empty()) {
        throw std::runtime_error("No candidates provide any coverage");
    }

    // Sort all intervals by start position
    std::sort(all_intervals.begin(), all_intervals.end());

    // Merge overlapping intervals
    std::vector<Interval> merged_intervals;
    merged_intervals.push_back(all_intervals[0]);

    for (size_t i = 1; i < all_intervals.size(); ++i) {
        Interval& last_interval = merged_intervals.back();
        const Interval& current_interval = all_intervals[i];

        if (current_interval.start <= last_interval.end) {
            // Intervals overlap or are adjacent, merge them
            last_interval.end = std::max(last_interval.end, current_interval.end);
        } else {
            // Intervals don't touch, add new interval
            merged_intervals.push_back(current_interval);
        }
    }

    // Now check if mergedIntervals cover all targetIntervals
    // For each target interval, check if it's fully covered by merged intervals
    for (const auto& target_interval : target_intervals) {
        int64_t target_start = target_interval.first;
        int64_t target_end = target_interval.second;
        bool covered = false;

        // Find a merged interval that could cover this target interval
        for (const auto& merged_interval : merged_intervals) {
            if (merged_interval.start <= target_start && merged_interval.end >= target_end) {
                covered = true;
                break;
            }
        }

        if (!covered) {
            throw std::runtime_error(
                "Cannot find covering candidate for shardedKey: " + target_shard.ToString()
                + ", targetTensor: " + target_tensor.GetTensorInfo() + ", missing interval: ["
                + std::to_string(target_start) + ", " + std::to_string(target_end) + ")");
        }
    }
} // check_candidates_coverage

std::vector<std::pair<int64_t, int64_t>>
ShardedTensorToFlatIntervals(const ShardedKey& sharded_key, const ATensor& tensor) {
    const std::vector<int64_t>& global_shape = sharded_key.global_shape;
    const std::vector<int64_t>& global_offset = sharded_key.global_offset;
    const auto dims = global_shape.size();

    std::vector<std::pair<int64_t, int64_t>> intervals;

    // For efficiency, we group contiguous elements into intervals
    // Elements are contiguous in the last dimension, so we iterate through all other dimensions
    // and create an interval for each "row" in the last dimension

    if (dims == 0) {
        // Scalar case
        return {{0, 1}};
    }

    // Calculate total number of positions in dimensions [0, dims-2]
    int64_t num_rows = 1;
    for (int i = 0; i < dims - 1; ++i) {
        num_rows *= tensor.size[i];
    }

    // For each "row" (fixed coordinates in all dimensions except the last)
    for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        // Convert linear row index to coordinates in dimensions [0, dims-2]
        std::vector<int64_t> coords(dims);
        int64_t temp = row_idx;
        for (auto i = static_cast<int64_t>(dims) - 2; i >= 0; --i) {
            coords[i] = (temp % tensor.size[i]) + global_offset[i];
            temp /= tensor.size[i];
        }

        // Set the last dimension coordinate to the start of the row
        coords[dims - 1] = global_offset[dims - 1];

        // Calculate flat index for the start of this row
        int64_t flat_start = CoordsToFlatIndex(coords, global_shape);

        // The row contains tensor.size(dims-1) contiguous elements
        int64_t row_length = tensor.size[dims - 1];
        intervals.emplace_back(flat_start, flat_start + row_length);
    }

    return intervals;
} // sharded_tensor_to_flat_intervals

std::vector<CopyOperation> FindCoveringCandidatesUnsafe(
    std::vector<std::pair<ShardedKey, ATensor>>& candidates,
    const ShardedKey& target_shard,
    const ATensor& target_tensor) {
    std::vector<CopyOperation> copy_operations;
    const auto dims = target_shard.global_shape.size();

    // Process each candidate to find overlapping regions with target
    for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
        const auto& candidate_key = candidates[candidate_idx].first;
        const auto& candidate_tensor = candidates[candidate_idx].second;

        // Calculate intersection between candidate and target
        std::vector<int64_t> intersect_start(dims);
        std::vector<int64_t> intersect_end(dims);
        bool has_intersection = true;

        for (int i = 0; i < dims; ++i) {
            int64_t cand_start = candidate_key.global_offset[i];
            int64_t cand_end = cand_start + candidate_tensor.size[i];
            int64_t target_start = target_shard.global_offset[i];
            int64_t target_end = target_start + target_tensor.size[i];

            intersect_start[i] = std::max(cand_start, target_start);
            intersect_end[i] = std::min(cand_end, target_end);

            if (intersect_start[i] >= intersect_end[i]) {
                has_intersection = false;
                break;
            }
        }

        if (!has_intersection) {
            continue; // No overlap with this candidate, skip
        }

        // Create copy operation for the overlapping region
        CopyOperation copy_op;
        copy_op.candidate_index = candidate_idx;
        copy_op.src_offset.resize(dims);
        copy_op.target_offset.resize(dims);
        copy_op.copy_shape.resize(dims);

        for (int i = 0; i < dims; ++i) {
            // Offset within source candidate tensor
            copy_op.src_offset[i] = intersect_start[i] - candidate_key.global_offset[i];
            // Offset within target tensor
            copy_op.target_offset[i] = intersect_start[i] - target_shard.global_offset[i];
            // Size of the region to copy
            copy_op.copy_shape[i] = intersect_end[i] - intersect_start[i];
        }

        copy_operations.push_back(std::move(copy_op));
    }

    return copy_operations;
} // find_covering_candidates_unsafe

void ValidateCandidateFilterParams(
    const std::vector<std::pair<ShardedKey, ATensor>>& candidates,
    const ShardedKey& target_shard,
    const ATensor& target_tensor) {
    if (candidates.empty()) {
        SPDLOG_ERROR(
            "candidates cannot be empty for shardedKey: {}, targetTensor: {}",
            target_shard.ToString(),
            target_tensor.GetTensorMetaInfo());
        throw std::runtime_error("candidates cannot be empty");
    }

    if (target_shard.global_shape.size() != static_cast<size_t>(target_tensor.dim_num)) {
        SPDLOG_ERROR(
            "targetShard dimensions must match targetTensor dimensions for "
            "shardedKey: {}, targetTensor: {}",
            target_shard.ToString(),
            target_tensor.GetTensorMetaInfo());
        throw std::runtime_error("targetShard dimensions must match targetTensor dimensions");
    }

    for (size_t i = 0; i < target_shard.global_shape.size(); ++i) {
        int64_t max_shape = target_shard.global_shape[i] - target_shard.global_offset[i];
        if (target_tensor.size[i] > max_shape) {
            SPDLOG_ERROR(
                "Target tensor shape must be less than or equal to maxShape at "
                "dimension {}: tensor={}, maxShape={} "
                "shardedKey: {}, targetTensor: {}",
                i,
                target_tensor.size[i],
                max_shape,
                target_shard.ToString(),
                target_tensor.GetTensorMetaInfo());
            throw std::runtime_error(
                "Target tensor shape must be less than or equal to maxShape at "
                "dimension "
                + std::to_string(i) + ": tensor=" + std::to_string(target_tensor.size[i])
                + ", maxShape=" + std::to_string(max_shape));
        }
    }

    // Validate all candidates have consistent dimensions and types
    size_t expected_dims = target_shard.global_shape.size();
    auto expected_dtype = target_tensor.dtype;

    for (const auto& candidate : candidates) {
        if (candidate.first.global_shape.size() != expected_dims) {
            SPDLOG_ERROR(
                "All candidates must have same dimension count as target for "
                "shardedKey: {} expected_dims: {} "
                "candidate_dims: {}",
                target_shard.key,
                expected_dims,
                candidate.first.global_shape.size());
            throw std::runtime_error("All candidates must have same dimension count as target");
        }
        if (candidate.second.dtype != expected_dtype) {
            SPDLOG_ERROR(
                "All candidates must have same data type as target for "
                "shardedKey: {} expected_dtype: {} "
                "candidate_dtype: {}",
                target_shard.key,
                static_cast<int>(expected_dtype),
                static_cast<int>(candidate.second.dtype));
            throw std::runtime_error("All candidates must have same data type as target");
        }
        for (size_t i = 0; i < expected_dims; ++i) {
            if (candidate.first.global_shape[i] != target_shard.global_shape[i]) {
                SPDLOG_ERROR(
                    "All candidates must have same globalShape as target for "
                    "shardedKey: {} dimension: {} "
                    "expected_globalShape: {} candidate_globalShape: {}",
                    target_shard.key,
                    i,
                    target_shard.global_shape[i],
                    candidate.first.global_shape[i]);
                throw std::runtime_error("All candidates must have same globalShape as target");
            }

            // int64_t max_shape = candidate.first.global_shape[i] - candidate.second.global_shape[i];
            // if (candidate.second.size[i] > max_shape) {
            //     SPDLOG_ERROR(
            //         "Candidate tensor shape must be less than or equal to "
            //         "maxShape at dimension {}: tensor={}, "
            //         "maxShape={} shardedKey: {}",
            //         i,
            //         candidate.second.size[i],
            //         max_shape,
            //         candidate.first.ToString());
            //     throw std::runtime_error(
            //         "Candidate tensor shape must be less than or equal to "
            //         "maxShape at dimension "
            //         + std::to_string(i) + ": tensor=" + std::to_string(candidate.second.size[i])
            //         + ", maxShape=" + std::to_string(max_shape));
            // }
        }
    }

    CheckCandidatesCoverage(candidates, target_shard, target_tensor);
}

std::vector<CopyOperation> FindCoveringCandidates(
    std::vector<std::pair<ShardedKey, ATensor>>& candidates,
    const ShardedKey& target_shard,
    const ATensor& target_tensor) {
    ValidateCandidateFilterParams(candidates, target_shard, target_tensor);
    return FindCoveringCandidatesUnsafe(candidates, target_shard, target_tensor);
} // find_covering_candidates

} // namespace astate
