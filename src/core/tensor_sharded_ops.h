#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include "common/string_utils.h"
#include "core/atensor.h"
#include "core/shardedkey.h"

namespace astate {


/**
 * @brief Validate parameters for ShardedKey tensor copy operations
 * @param srcShardedKey Source ShardedKey
 * @param srcTensor Source tensor
 * @param targetShardedKey Target ShardedKey
 * @param targetTensor Target tensor
 * @throws std::runtime_error if validation fails
 * @note Call this once before performing multiple copy operations
 */
inline void ValidateShardedCopyParams(
    const ShardedKey& src_sharded_key,
    const ATensor& src_tensor,
    const ShardedKey& target_sharded_key,
    const ATensor& target_tensor) {
    if (src_sharded_key.global_shape.size() != target_sharded_key.global_shape.size()) {
        throw std::runtime_error("Source and target ShardedKeys must have the "
                                 "same number of dimensions");
    }

    if (src_sharded_key.global_shape.size() != static_cast<size_t>(src_tensor.dim_num)
        || target_sharded_key.global_shape.size() != static_cast<size_t>(target_tensor.dim_num)) {
        throw std::runtime_error("ShardedKey dimensions must match corresponding tensor dimensions");
    }

    if (src_tensor.dtype != target_tensor.dtype) {
        throw std::runtime_error("Source and target tensors must have the same data type");
    }

    for (size_t i = 0; i < src_sharded_key.global_shape.size(); ++i) {
        if (src_sharded_key.global_shape[i] != target_sharded_key.global_shape[i]) {
            throw std::runtime_error("Source and target ShardedKeys must have the same shape");
        }

        // Check tensor shape <= globalShape - globalOffset
        int64_t src_max_shape = src_sharded_key.global_shape[i] - src_sharded_key.global_offset[i];
        int64_t target_max_shape = target_sharded_key.global_shape[i] - target_sharded_key.global_offset[i];

        if (src_tensor.size[i] > src_max_shape) {
            throw std::runtime_error(
                "Source tensor shape exceeds available space at dimension " + std::to_string(i)
                + ": tensor=" + std::to_string(src_tensor.size[i]) + ", available=" + std::to_string(src_max_shape));
        }
        if (target_tensor.size[i] > target_max_shape) {
            throw std::runtime_error(
                "Target tensor shape exceeds available space at dimension " + std::to_string(i) + ": tensor="
                + std::to_string(target_tensor.size[i]) + ", available=" + std::to_string(target_max_shape));
        }
    }
}

/**
 * @brief High-performance tensor copy based on ShardedKey coordinates (CPU tensors only)
 * @param srcShardedKey Source ShardedKey (pre-validated)
 * @param srcTensor Source tensor (pre-validated)
 * @param targetShardedKey Target ShardedKey (pre-validated)
 * @param targetTensor Target tensor (pre-validated, modified in-place)
 * @param non_blocking_copy Whether to perform non-blocking copy (default: false)
 * @note PERFORMANCE CRITICAL: Assumes all inputs have been validated
 * @note Call validate_sharded_copy_params() once before high-frequency usage
 */
inline bool CopyTensorWithShardedKeysUnsafe(
    const ShardedKey& src_sharded_key,
    const torch::Tensor& src_tensor,
    const ShardedKey& target_sharded_key,
    const torch::Tensor& target_tensor,
    const c10::cuda::CUDAStream* stream = nullptr,
    const bool non_blocking_copy = false) {
    const size_t dims = src_sharded_key.global_shape.size();

    // Pre-allocate vectors with known size for performance
    std::vector<int64_t> intersect_start;
    std::vector<int64_t> src_offset;
    std::vector<int64_t> target_offset;
    std::vector<int64_t> copy_shape;
    intersect_start.reserve(dims);
    src_offset.reserve(dims);
    target_offset.reserve(dims);
    copy_shape.reserve(dims);

    // Fast intersection calculation
    bool has_intersection = true;
    for (size_t i = 0; i < dims; ++i) {
        const int64_t src_start = src_sharded_key.global_offset[i];
        const int64_t src_end = src_start + src_tensor.size(static_cast<int64_t>(i));
        const int64_t target_start = target_sharded_key.global_offset[i];
        const int64_t target_end = target_start + target_tensor.size(static_cast<int64_t>(i));

        const int64_t intersect_dim_start = std::max(src_start, target_start);
        const int64_t intersect_dim_end = std::min(src_end, target_end);

        if (intersect_dim_start >= intersect_dim_end) {
            has_intersection = false;
            break;
        }

        intersect_start.emplace_back(intersect_dim_start);
        src_offset.emplace_back(intersect_dim_start - src_start);
        target_offset.emplace_back(intersect_dim_start - target_start);
        copy_shape.emplace_back(intersect_dim_end - intersect_dim_start);
    }

    if (!has_intersection) {
        return false;
    }


    // Fast tensor slicing and copy (optimized for CPU)
    torch::Tensor src_view = src_tensor;
    torch::Tensor target_view = target_tensor;

    for (size_t i = 0; i < dims; ++i) {
        src_view = src_view.narrow(static_cast<int64_t>(i), src_offset[i], copy_shape[i]);
        target_view = target_view.narrow(static_cast<int64_t>(i), target_offset[i], copy_shape[i]);
    }

    try {
        if ((src_view.device().is_cuda() || target_view.device().is_cuda()) && stream != nullptr && non_blocking_copy) {
            // Synchronous copy on specified stream
            c10::cuda::CUDAStreamGuard guard(*stream);
            target_view.copy_(src_view, /*non_blocking=*/true);
            stream->synchronize();
        } else {
            target_view.copy_(src_view);
        }
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error in copy_tensor_with_sharded_keys_unsafe: {}", e.what());
        SPDLOG_ERROR("Src tensor device: {}", toString(src_view.device()));
        SPDLOG_ERROR("Target tensor device: {}", toString(target_view.device()));
        SPDLOG_ERROR("Src tensor dtype: {}", src_view.dtype().name());
        SPDLOG_ERROR("Target tensor dtype: {}", target_view.dtype().name());
        SPDLOG_ERROR("Src tensor shape: {}", src_view.sizes().size());
        SPDLOG_ERROR("Target tensor shape: {}", target_view.sizes().size());
        SPDLOG_ERROR("Src tensor requires_grad: {}", src_view.requires_grad());
        SPDLOG_ERROR("Target tensor requires_grad: {}", target_view.requires_grad());
        throw e;
    }
}

/**
 * @brief Structure to hold copy operation information
 */
struct CopyOperation {
    size_t candidate_index; // Index of candidate in the original vector
    std::vector<int64_t> src_offset; // Offset in source candidate tensor
    std::vector<int64_t> target_offset; // Offset in target tensor
    std::vector<int64_t> copy_shape; // Shape of region to copy
};

/**
 * @brief Convert multi-dimensional tensor coordinates to flat index
 * @param coords Multi-dimensional coordinates
 * @param shape Shape of the tensor
 * @return Flattened index
 */
inline int64_t CoordsToFlatIndex(const std::vector<int64_t>& coords, const std::vector<int64_t>& shape) {
    int64_t flat_index = 0;
    int64_t stride = 1;

    // Row-major order: rightmost dimension changes fastest
    auto size = static_cast<int64_t>(shape.size());
    for (int64_t i = size - 1; i >= 0; --i) {
        flat_index += coords[i] * stride;
        stride *= shape[i];
    }

    return flat_index;
}

/**
 * @brief Convert ShardedKey with tensor to multiple flattened intervals
 * @param shardedKey ShardedKey defining the position in global space
 * @param tensor Tensor with actual data
 * @return Vector of (start, end) intervals in flattened space
 * @note When tensor dimensions don't match global dimensions, the elements are not contiguous in flattened space
 */
std::vector<std::pair<int64_t, int64_t>>
ShardedTensorToFlatIntervals(const ShardedKey& sharded_key, const ATensor& tensor);

/**
 * @brief Check if candidates can fully cover the target tensor using interval merging
 * @param candidates Vector of candidate shard pairs
 * @param targetShard Target shard key
 * @param targetTensor Target tensor
 * @return true if fully covered, false otherwise
 * @throws std::runtime_error with detailed error message if not covered
 */
void CheckCandidatesCoverage(
    const std::vector<std::pair<ShardedKey, ATensor>>& candidates,
    const ShardedKey& target_shard,
    const ATensor& target_tensor);

/**
 * @brief Validate parameters for candidate filtering operations
 * @param candidates Vector of candidate shard pairs
 * @param targetShard Target shard key
 * @param targetTensor Target tensor
 * @throws std::runtime_error if validation fails
 */
void ValidateCandidateFilterParams(
    const std::vector<std::pair<ShardedKey, ATensor>>& candidates,
    const ShardedKey& target_shard,
    const ATensor& target_tensor);

/**
 * @brief Region-based candidate filtering following exact user requirements (CPU tensors only)
 * @param candidates Vector of candidate shard pairs (pre-validated)
 * @param targetShard Target shard key (pre-validated)
 * @param targetTensor Target tensor (pre-validated)
 * @return std::vector<CopyOperation> List of copy operations needed to fill target
 * @throws std::runtime_error if any region cannot be covered by candidates
 * @note PERFORMANCE CRITICAL: Assumes all inputs have been validated
 * @note Call validate_candidate_filter_params() once before high-frequency usage
 *
 * @note IMPLEMENTATION NOTE: This is a relatively naive implementation based on two key assumptions:
 *       1. Tensor sharding during model training is complete and non-overlapping
 *       2. LLM tensors are typically 3-dimensional or less, keeping search costs manageable
 *       Under these assumptions, the current implementation is near-optimal. If these assumptions
 *       are violated, the algorithm will need optimization
 */
std::vector<CopyOperation> FindCoveringCandidatesUnsafe(
    std::vector<std::pair<ShardedKey, ATensor>>& candidates,
    const ShardedKey& target_shard,
    const ATensor& target_tensor);

/**
 * @brief Safe candidate filtering with simple validation (CPU tensors only)
 * @param candidates Vector of candidate shard pairs
 * @param targetShard Target shard key
 * @param targetTensor Target tensor
 * @return std::vector<CopyOperation> List of copy operations needed to fill target
 * @throws std::runtime_error if parameters are invalid or target cannot be covered
 * @note For high-frequency usage, prefer validate_candidate_filter_params() + find_covering_candidates_unsafe()
 */
std::vector<CopyOperation> FindCoveringCandidates(
    std::vector<std::pair<ShardedKey, ATensor>>& candidates,
    const ShardedKey& target_shard,
    const ATensor& target_tensor);

} // namespace astate
