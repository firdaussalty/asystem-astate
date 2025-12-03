#include "core/tensor_sharded_ops.h"

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include "core/atensor.h"
#include "core/shardedkey.h"
#include "core/utils.h"

using namespace astate;

class TensorShardedOpsTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // Set up common test data
    }

    // Helper function: create test ShardedKey
    ShardedKey
    createShardedKey(const std::string& key, const std::vector<int64_t>& shape, const std::vector<int64_t>& offset) {
        ShardedKey sk;
        sk.key = key;
        sk.global_shape = shape;
        sk.global_offset = offset;
        return sk;
    }

    // Helper function: create test torch::Tensor
    torch::Tensor
    createTensor(const std::vector<int64_t>& shape, torch::ScalarType dtype = torch::kFloat32, float value = 1.0f) {
        return torch::full(shape, value, torch::TensorOptions().dtype(dtype));
    }

    // Helper function: create test ATensor
    std::shared_ptr<ATensor> createATensor(const torch::Tensor& torch_tensor) { return TensorToATensor(torch_tensor); }
};

// ==================== ValidateShardedCopyParams Tests ====================

TEST_F(TensorShardedOpsTest, validate_sharded_copy_params_valid_case) {
    auto srcKey = createShardedKey("src", {10, 10}, {0, 0});
    auto src_torch_tensor = createTensor({2, 3});
    auto srcTensor = createATensor(src_torch_tensor);
    auto targetKey = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({2, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    // Should not throw exception
    EXPECT_NO_THROW(ValidateShardedCopyParams(srcKey, *srcTensor, targetKey, *targetTensor));
}

TEST_F(TensorShardedOpsTest, validate_sharded_copy_params_dimension_mismatch) {
    auto srcKey = createShardedKey("src", {10, 10}, {0, 0});
    auto src_torch_tensor = createTensor({2, 3});
    auto srcTensor = createATensor(src_torch_tensor);
    auto targetKey = createShardedKey("target", {10, 10, 10}, {0, 0, 0}); // Different dimensions
    auto target_torch_tensor = createTensor({2, 3, 4});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateShardedCopyParams(srcKey, *srcTensor, targetKey, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, validate_sharded_copy_params_shape_mismatch) {
    auto srcKey = createShardedKey("src", {10, 10}, {0, 0});
    auto src_torch_tensor = createTensor({12, 12}); // tensor shape > available space
    auto srcTensor = createATensor(src_torch_tensor);
    auto targetKey = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({2, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateShardedCopyParams(srcKey, *srcTensor, targetKey, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, validate_sharded_copy_params_dtype_mismatch) {
    auto srcKey = createShardedKey("src", {10, 10}, {0, 0});
    auto src_torch_tensor = createTensor({2, 3}, torch::kFloat32);
    auto srcTensor = createATensor(src_torch_tensor);
    auto targetKey = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({2, 3}, torch::kFloat64); // Different data types
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateShardedCopyParams(srcKey, *srcTensor, targetKey, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, invalid_offset_exceeds_shape) {
    auto srcKey = createShardedKey("src", {10, 10}, {8, 8});
    auto src_torch_tensor = createTensor({3, 3}); // 8+3 > 10, exceeds boundary
    auto srcTensor = createATensor(src_torch_tensor);
    auto targetKey = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({3, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateShardedCopyParams(srcKey, *srcTensor, targetKey, *targetTensor), std::runtime_error);
}

// ==================== CopyTensorWithShardedKeysUnsafe Tests ====================

TEST_F(TensorShardedOpsTest, copy_tensor_with_sharded_keys_unsafe_full_overlap) {
    auto srcKey = createShardedKey("src", {10, 10}, {0, 0});
    auto srcTensor = createTensor({2, 3}, torch::kFloat32, 5.0f);
    auto targetKey = createShardedKey("target", {10, 10}, {0, 0});
    auto targetTensor = createTensor({2, 3}, torch::kFloat32, 0.0f);

    CopyTensorWithShardedKeysUnsafe(srcKey, srcTensor, targetKey, targetTensor);

    EXPECT_TRUE(torch::allclose(targetTensor, srcTensor));
}

TEST_F(TensorShardedOpsTest, copy_tensor_with_sharded_keys_unsafe_partial_overlap) {
    auto srcKey = createShardedKey("src", {10, 10}, {0, 0});
    auto srcTensor = createTensor({2, 3}, torch::kFloat32, 5.0f);
    auto targetKey = createShardedKey("target", {10, 10}, {1, 1});
    auto targetTensor = createTensor({2, 3}, torch::kFloat32, 0.0f);

    CopyTensorWithShardedKeysUnsafe(srcKey, srcTensor, targetKey, targetTensor);

    // Only overlapping parts should be copied
    auto expected = torch::zeros({2, 3});
    expected.slice(0, 0, 1).slice(1, 0, 2) = 5.0f; // [0:1, 0:2] = 5.0

    EXPECT_TRUE(torch::allclose(targetTensor, expected));
}

TEST_F(TensorShardedOpsTest, copy_tensor_with_sharded_keys_unsafe_no_overlap) {
    auto srcKey = createShardedKey("src", {10, 10}, {0, 0});
    auto srcTensor = createTensor({2, 3}, torch::kFloat32, 5.0f);
    auto targetKey = createShardedKey("target", {10, 10}, {5, 5}); // No overlap
    auto targetTensor = createTensor({2, 3}, torch::kFloat32, 0.0f);

    CopyTensorWithShardedKeysUnsafe(srcKey, srcTensor, targetKey, targetTensor);

    // Target tensor should remain unchanged
    EXPECT_TRUE(torch::allclose(targetTensor, torch::zeros({2, 3})));
}

// ==================== CoordsToFlatIndex Tests ====================

TEST_F(TensorShardedOpsTest, coords_to_flat_index_2d) {
    std::vector<int64_t> shape = {4, 5}; // 4x5 matrix

    // Test first element
    EXPECT_EQ(CoordsToFlatIndex({0, 0}, shape), 0);

    // Test last element
    EXPECT_EQ(CoordsToFlatIndex({3, 4}, shape), 19);

    // Test middle element (1,2) = 1*5 + 2 = 7
    EXPECT_EQ(CoordsToFlatIndex({1, 2}, shape), 7);
}

TEST_F(TensorShardedOpsTest, coords_to_flat_index_3d) {
    std::vector<int64_t> shape = {2, 3, 4}; // 2x3x4 tensor

    // (0,0,0) = 0
    EXPECT_EQ(CoordsToFlatIndex({0, 0, 0}, shape), 0);

    // (1,2,3) = 1*(3*4) + 2*4 + 3 = 12 + 8 + 3 = 23
    EXPECT_EQ(CoordsToFlatIndex({1, 2, 3}, shape), 23);
}

TEST_F(TensorShardedOpsTest, coords_to_flat_index_1d) {
    std::vector<int64_t> shape = {10};
    EXPECT_EQ(CoordsToFlatIndex({5}, shape), 5);
}

// ==================== ShardedTensorToFlatIntervals Tests ====================

TEST_F(TensorShardedOpsTest, non_contiguous_memory_layout_2d) {
    // global_shape = [10, 10], offset = [2, 3], tensor size = [3, 4]
    // This means tensor occupies only part of global tensor
    auto shardKey = createShardedKey("shard", {10, 10}, {2, 3});
    auto src_torch_tensor = createTensor({3, 4}); // Less than available space (8, 7)
    auto srcTensor = createATensor(src_torch_tensor);

    auto intervals = ShardedTensorToFlatIntervals(shardKey, *srcTensor);

    // Should generate 3 intervals (one per row)
    EXPECT_EQ(intervals.size(), 3);

    // Verify first row start position: (2,3) in 10x10 = 2*10 + 3 = 23
    EXPECT_EQ(intervals[0].first, 23);
    EXPECT_EQ(intervals[0].second, 27); // 23 + 4
}

TEST_F(TensorShardedOpsTest, non_contiguous_memory_layout_3d) {
    // 3D scenario with more complex non-contiguous layout
    auto shardKey = createShardedKey("shard", {5, 5, 5}, {1, 1, 1});
    auto src_torch_tensor = createTensor({2, 2, 2});
    auto srcTensor = createATensor(src_torch_tensor);

    auto intervals = ShardedTensorToFlatIntervals(shardKey, *srcTensor);

    // Should generate 2*2 = 4 intervals
    EXPECT_EQ(intervals.size(), 4);
}

TEST_F(TensorShardedOpsTest, flatten_intervals_correctness) {
    // 2D case: verify interval calculations
    auto shardKey = createShardedKey("test", {10, 10}, {2, 3});
    auto src_torch_tensor = createTensor({2, 3});
    auto srcTensor = createATensor(src_torch_tensor);

    auto intervals = ShardedTensorToFlatIntervals(shardKey, *srcTensor);

    // Expect 2 intervals (2 rows)
    EXPECT_EQ(intervals.size(), 2);

    // First row: (2,3) -> flat index = 2*10 + 3 = 23, length = 3
    EXPECT_EQ(intervals[0].first, 23);
    EXPECT_EQ(intervals[0].second, 26);

    // Second row: (3,3) -> flat index = 3*10 + 3 = 33, length = 3
    EXPECT_EQ(intervals[1].first, 33);
    EXPECT_EQ(intervals[1].second, 36);
}

TEST_F(TensorShardedOpsTest, interval_merging_edge_cases) {
    // Create scenario with adjacent but non-overlapping intervals
    auto shardKey1 = createShardedKey("shard1", {10, 10}, {0, 0});
    auto src_torch_tensor1 = createTensor({1, 10}); // First row
    auto tensor1 = createATensor(src_torch_tensor1);

    auto shardKey2 = createShardedKey("shard2", {10, 10}, {1, 0});
    auto src_torch_tensor2 = createTensor({1, 10}); // Second row
    auto tensor2 = createATensor(src_torch_tensor2);

    auto intervals1 = ShardedTensorToFlatIntervals(shardKey1, *tensor1);
    auto intervals2 = ShardedTensorToFlatIntervals(shardKey2, *tensor2);

    // Verify intervals are adjacent
    EXPECT_EQ(intervals1[0].second, intervals2[0].first);
}

// ==================== ValidateCandidateFilterParams Tests ====================

TEST_F(TensorShardedOpsTest, validate_candidate_filter_params_valid) {
    auto src_torch_tensor1 = createTensor({2, 3});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({2, 3});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor1},
           {createShardedKey("c2", {10, 10}, {2, 0}), *srcTensor2}};
    auto targetShard = createShardedKey("target", {10, 10}, {1, 0});
    auto target_torch_tensor = createTensor({2, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_NO_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor));
}

TEST_F(TensorShardedOpsTest, validate_candidate_filter_params_empty_candidates) {
    std::vector<std::pair<ShardedKey, ATensor>> candidates;
    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({2, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, validate_candidate_filter_params_inconsistent_dimensions) {
    auto src_torch_tensor1 = createTensor({2, 3});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({2, 3});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {
        {createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor1},
        {createShardedKey("c2", {10, 10, 10}, {0, 0, 0}), *srcTensor2} // Different dimensions
    };
    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({2, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, validate_candidate_filter_params_inconsistent_dtype) {
    auto src_torch_tensor1 = createTensor({2, 3}, torch::kFloat32);
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({2, 3}, torch::kFloat64);
    auto srcTensor2 = createATensor(src_torch_tensor2);
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {
        {createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor1},
        {createShardedKey("c2", {10, 10}, {2, 0}), *srcTensor2} // Different types
    };
    auto targetShard = createShardedKey("target", {10, 10}, {1, 0});
    auto target_torch_tensor = createTensor({2, 3}, torch::kFloat32);
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, validate_candidate_filter_params_no_covering_candidate) {
    auto src_torch_tensor1 = createTensor({3, 3});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {
        {createShardedKey("c1", {20, 20}, {10, 10}), *srcTensor1} // Does not cover target
    };
    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({2, 2});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, validate_candidate_filter_params_partial_coverage_gap) {
    auto src_torch_tensor1 = createTensor({3, 3});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({3, 3});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    // Test exception when there's a gap that cannot be covered
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {
        {createShardedKey("shard_1", {10, 10}, {0, 0}), *srcTensor1},
        {createShardedKey("shard_2", {10, 10}, {5, 5}), *srcTensor2} // Gap in the middle
    };

    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({6, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, coverage_validation_with_exact_fit) {
    // Test exact coverage scenario
    auto src_torch_tensor1 = createTensor({5, 10});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({5, 10});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor1},
           {createShardedKey("c2", {10, 10}, {5, 0}), *srcTensor2}};

    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({10, 10});
    auto targetTensor = createATensor(target_torch_tensor);

    // Exact coverage should not throw exception
    EXPECT_NO_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor));
}

TEST_F(TensorShardedOpsTest, coverage_validation_with_small_gap) {
    // Test scenario with small gap
    auto src_torch_tensor1 = createTensor({4, 10});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({5, 10});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {
        {createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor1},
        {createShardedKey("c2", {10, 10}, {5, 0}), *srcTensor2} // 1 row gap in the middle
    };

    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({10, 10});
    auto targetTensor = createATensor(target_torch_tensor);

    // Should throw exception due to gap
    EXPECT_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, complex_multidimensional_coverage) {
    // Create a checkerboard coverage scenario
    std::vector<torch::Tensor> src_torch_tensors;
    std::vector<std::pair<ShardedKey, ATensor>> candidates;

    // Create 4x4 blocks to cover 8x8 area
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            auto src_torch_tensor = createTensor({4, 4});
            auto srcTensor = createATensor(src_torch_tensor);
            src_torch_tensors.push_back(src_torch_tensor);
            candidates.push_back(
                {createShardedKey("c_" + std::to_string(i * 2 + j), {8, 8}, {i * 4, j * 4}), *srcTensor});
        }
    }

    auto targetShard = createShardedKey("target", {8, 8}, {0, 0});
    auto target_torch_tensor = createTensor({8, 8});
    auto targetTensor = createATensor(target_torch_tensor);

    // Should successfully validate coverage
    EXPECT_NO_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor));
}

TEST_F(TensorShardedOpsTest, partial_tensor_size_coverage) {
    // Test when tensor size is less than maxShape
    auto src_torch_tensor1 = createTensor({10, 10});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({8, 8});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("c1", {20, 20}, {0, 0}), *srcTensor1}, // Uses only half of available space
           {createShardedKey("c2", {20, 20}, {5, 5}), *srcTensor2}};

    auto targetShard = createShardedKey("target", {20, 20}, {2, 2});
    auto target_torch_tensor = createTensor({6, 6});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_NO_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor));
}

TEST_F(TensorShardedOpsTest, completely_disjoint_candidates) {
    auto src_torch_tensor1 = createTensor({5, 5});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({5, 5});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("c1", {20, 20}, {0, 0}), *srcTensor1},
           {createShardedKey("c2", {20, 20}, {15, 15}), *srcTensor2}};

    // Target is in the middle, neither candidate covers it
    auto targetShard = createShardedKey("target", {20, 20}, {7, 7});
    auto target_torch_tensor = createTensor({6, 6});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, exact_boundary_alignment) {
    // Candidates exactly touch but don't overlap
    auto src_torch_tensor1 = createTensor({5, 5});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({5, 5});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    auto src_torch_tensor3 = createTensor({5, 5});
    auto srcTensor3 = createATensor(src_torch_tensor3);
    auto src_torch_tensor4 = createTensor({5, 5});
    auto srcTensor4 = createATensor(src_torch_tensor4);
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor1},
           {createShardedKey("c2", {10, 10}, {5, 0}), *srcTensor2},
           {createShardedKey("c3", {10, 10}, {0, 5}), *srcTensor3},
           {createShardedKey("c4", {10, 10}, {5, 5}), *srcTensor4}};

    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({10, 10});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_NO_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor));
}

TEST_F(TensorShardedOpsTest, error_message_accuracy) {
    auto src_torch_tensor = createTensor({3, 3});
    auto srcTensor = createATensor(src_torch_tensor);
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {{createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor}};

    auto targetShard = createShardedKey("target", {10, 10}, {5, 5});
    auto target_torch_tensor = createTensor({5, 5});
    auto targetTensor = createATensor(target_torch_tensor);

    try {
        ValidateCandidateFilterParams(candidates, targetShard, *targetTensor);
        FAIL() << "Expected exception was not thrown";
    } catch (const std::runtime_error& e) {
        // Verify error message contains correct offset and shape info
        std::string error_msg = e.what();
        SPDLOG_INFO("error_msg: {}", error_msg);
        EXPECT_TRUE(error_msg.find("global_offset=[5, 5]") != std::string::npos);
        EXPECT_TRUE(error_msg.find("global_shape=[10, 10]") != std::string::npos);
    }
}

TEST_F(TensorShardedOpsTest, one_dimensional_tensor_coverage_validation) {
    // 1D tensor coverage validation
    auto src_torch_tensor1 = createTensor({50});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({60});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("c1", {100}, {0}), *srcTensor1}, {createShardedKey("c2", {100}, {40}), *srcTensor2}};
    auto targetShard = createShardedKey("target", {100}, {10});
    auto target_torch_tensor = createTensor({80});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_NO_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor));
}

// ==================== FindCoveringCandidatesUnsafe Tests ====================

TEST_F(TensorShardedOpsTest, find_covering_candidates_unsafe_single_candidate_full_cover) {
    auto src_torch_tensor = createTensor({4, 4});
    auto srcTensor = createATensor(src_torch_tensor);
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {{createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor}};
    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({2, 2});
    auto targetTensor = createATensor(target_torch_tensor);

    auto copyOps = FindCoveringCandidatesUnsafe(candidates, targetShard, *targetTensor);

    EXPECT_GE(copyOps.size(), 1);
    EXPECT_EQ(copyOps[0].candidate_index, 0);
    // Verify correctness of copy operation
    EXPECT_EQ(copyOps[0].copy_shape.size(), 2);
}

TEST_F(TensorShardedOpsTest, find_covering_candidates_unsafe_sequential_candidates) {
    auto src_torch_tensor1 = createTensor({2, 3});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({2, 3});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    auto src_torch_tensor3 = createTensor({2, 3});
    auto srcTensor3 = createATensor(src_torch_tensor3);
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor1},
           {createShardedKey("c2", {10, 10}, {2, 0}), *srcTensor2},
           {createShardedKey("c3", {10, 10}, {4, 0}), *srcTensor3}};
    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({6, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    auto copyOps = FindCoveringCandidatesUnsafe(candidates, targetShard, *targetTensor);

    // Should find multiple copy operations to cover entire target area
    EXPECT_GT(copyOps.size(), 0);
    // Verify first operation comes from first candidate
    EXPECT_EQ(copyOps[0].candidate_index, 0);
}

TEST_F(TensorShardedOpsTest, new_algorithm_ordered_coverage) {
    // Test new algorithm's ability to cover target area
    auto src_torch_tensor1 = createTensor({4, 3});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({4, 3});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    auto src_torch_tensor3 = createTensor({4, 3});
    auto srcTensor3 = createATensor(src_torch_tensor3);
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("shard_1", {10, 10}, {0, 0}), *srcTensor1},
           {createShardedKey("shard_2", {10, 10}, {1, 0}), *srcTensor2},
           {createShardedKey("shard_3", {10, 10}, {2, 0}), *srcTensor3}};

    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({6, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    auto copyOps = FindCoveringCandidatesUnsafe(candidates, targetShard, *targetTensor);

    // Should find 3 copy operations
    EXPECT_EQ(copyOps.size(), 3);

    // Verify operation order
    EXPECT_EQ(copyOps[0].candidate_index, 0); // First shard
    EXPECT_EQ(copyOps[1].candidate_index, 1); // Second shard
    EXPECT_EQ(copyOps[2].candidate_index, 2); // Third shard
}

TEST_F(TensorShardedOpsTest, overlapping_candidates) {
    auto src_torch_tensor1 = createTensor({5, 5});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({5, 5});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    auto src_torch_tensor3 = createTensor({5, 5});
    auto srcTensor3 = createATensor(src_torch_tensor3);
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {
        {createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor1},
        {createShardedKey("c2", {10, 10}, {2, 2}), *srcTensor2}, // Overlaps with c1
        {createShardedKey("c3", {10, 10}, {4, 4}), *srcTensor3} // Overlaps with c2
    };

    auto targetShard = createShardedKey("target", {10, 10}, {3, 3});
    auto target_torch_tensor = createTensor({3, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    // Should handle overlapping candidates
    auto copyOps = FindCoveringCandidatesUnsafe(candidates, targetShard, *targetTensor);

    // Should have multiple copy operations from different overlapping candidates
    EXPECT_GE(copyOps.size(), 2);
}

TEST_F(TensorShardedOpsTest, non_uniform_candidate_distribution) {
    auto src_torch_tensor1 = createTensor({2, 2});
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({20, 20});
    auto srcTensor2 = createATensor(src_torch_tensor2);
    auto src_torch_tensor3 = createTensor({8, 30});
    auto srcTensor3 = createATensor(src_torch_tensor3);
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("small1", {30, 30}, {0, 0}), *srcTensor1},
           {createShardedKey("large", {30, 30}, {2, 0}), *srcTensor2},
           {createShardedKey("small2", {30, 30}, {22, 0}), *srcTensor3}};

    auto targetShard = createShardedKey("target", {30, 30}, {0, 0});
    auto target_torch_tensor = createTensor({30, 20});
    auto targetTensor = createATensor(target_torch_tensor);

    auto copyOps = FindCoveringCandidatesUnsafe(candidates, targetShard, *targetTensor);
    // Should have 3 operations from 3 different sized candidates
    EXPECT_EQ(copyOps.size(), 3);
}

TEST_F(TensorShardedOpsTest, copy_operation_correctness) {
    auto src_torch_tensor = createTensor({5, 5});
    auto srcTensor = createATensor(src_torch_tensor);
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {{createShardedKey("c1", {10, 10}, {2, 3}), *srcTensor}};

    auto targetShard = createShardedKey("target", {10, 10}, {4, 5});
    auto target_torch_tensor = createTensor({3, 3});
    auto targetTensor = createATensor(target_torch_tensor);

    auto copyOps = FindCoveringCandidatesUnsafe(candidates, targetShard, *targetTensor);

    ASSERT_EQ(copyOps.size(), 1);
    const auto& op = copyOps[0];

    // Verify src_offset: intersection start (4,5) - candidate start (2,3) = (2,2)
    EXPECT_EQ(op.src_offset[0], 2);
    EXPECT_EQ(op.src_offset[1], 2);

    // Verify target_offset: intersection start (4,5) - target start (4,5) = (0,0)
    EXPECT_EQ(op.target_offset[0], 0);
    EXPECT_EQ(op.target_offset[1], 0);

    // Verify copy_shape: intersection size should be (3,3)
    EXPECT_EQ(op.copy_shape[0], 3);
    EXPECT_EQ(op.copy_shape[1], 3);
}

// ==================== FindCoveringCandidates Tests ====================

TEST_F(TensorShardedOpsTest, find_covering_candidates_valid) {
    auto src_torch_tensor = createTensor({4, 4});
    auto srcTensor = createATensor(src_torch_tensor);
    std::vector<std::pair<ShardedKey, ATensor>> candidates = {{createShardedKey("c1", {10, 10}, {0, 0}), *srcTensor}};
    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({2, 2});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_NO_THROW(FindCoveringCandidates(candidates, targetShard, *targetTensor));
}

TEST_F(TensorShardedOpsTest, find_covering_candidates_invalid_params) {
    std::vector<std::pair<ShardedKey, ATensor>> candidates; // Empty candidates
    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({2, 2});
    auto targetTensor = createATensor(target_torch_tensor);

    EXPECT_THROW(FindCoveringCandidates(candidates, targetShard, *targetTensor), std::runtime_error);
}

TEST_F(TensorShardedOpsTest, large_number_of_candidates) {
    std::vector<torch::Tensor> src_torch_tensors;
    std::vector<std::pair<ShardedKey, ATensor>> candidates;

    // Create 100 small candidate blocks
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            auto src_torch_tensor = createTensor({10, 10});
            auto srcTensor = createATensor(src_torch_tensor);
            src_torch_tensors.push_back(src_torch_tensor);
            candidates.push_back(
                {createShardedKey("c_" + std::to_string(i * 10 + j), {100, 100}, {i * 10, j * 10}), *srcTensor});
        }
    }

    auto targetShard = createShardedKey("target", {100, 100}, {25, 25});
    auto target_torch_tensor = createTensor({50, 50});
    auto targetTensor = createATensor(target_torch_tensor);

    // Should handle large number of candidates
    EXPECT_NO_THROW(FindCoveringCandidates(candidates, targetShard, *targetTensor));
}

// ==================== Integration Tests ====================

TEST_F(TensorShardedOpsTest, integration_test_complete_workflow) {
    auto src_torch_tensor1 = createTensor({3, 4}, torch::kFloat, 1.0f);
    auto srcTensor1 = createATensor(src_torch_tensor1);
    auto src_torch_tensor2 = createTensor({3, 4}, torch::kFloat, 2.0f);
    auto srcTensor2 = createATensor(src_torch_tensor2);
    auto src_torch_tensor3 = createTensor({3, 4}, torch::kFloat, 3.0f);
    auto srcTensor3 = createATensor(src_torch_tensor3);
    // Create multiple candidate shards
    std::vector<std::pair<ShardedKey, ATensor>> candidates
        = {{createShardedKey("shard_2", {10, 10}, {1, 0}), *srcTensor2},
           {createShardedKey("shard_1", {10, 10}, {0, 0}), *srcTensor1},
           {createShardedKey("shard_3", {10, 10}, {2, 0}), *srcTensor3}};

    auto targetShard = createShardedKey("target", {10, 10}, {0, 0});
    auto target_torch_tensor = createTensor({4, 4}, torch::kFloat, 0.0f);
    auto targetTensor = createATensor(target_torch_tensor);

    // Step 1: Validate parameters
    EXPECT_NO_THROW(ValidateCandidateFilterParams(candidates, targetShard, *targetTensor));

    // Step 2: Find covering candidates (internally sorts)
    auto copyOps = FindCoveringCandidates(candidates, targetShard, *targetTensor);

    // Step 3: Verify results
    EXPECT_GT(copyOps.size(), 0);

    // Step 4: Verify validity of copy operations
    for (const auto& copyOp : copyOps) {
        EXPECT_LT(copyOp.candidate_index, candidates.size());
        EXPECT_EQ(copyOp.src_offset.size(), targetShard.global_shape.size());
        EXPECT_EQ(copyOp.target_offset.size(), targetShard.global_shape.size());
        EXPECT_EQ(copyOp.copy_shape.size(), targetShard.global_shape.size());
    }
}
