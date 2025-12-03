#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/Context.h>
#include <c10/core/TensorImpl.h>
#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "common/lock_utils.h"
#include "common/option.h"
#include "core/atensor.h"
#include "core/shardedkey.h"
#include "core/tensor_sharded_ops.h"
#include "core/tensor_table.h"
#include "core/utils.h"

namespace astate {

InMemoryTensorTable::InMemoryTensorTable(const std::string& name)
    : TensorTable(name) {
}

// InMemoryTensorTable implementation
bool InMemoryTensorTable::Put(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& py_tensor) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    try {
        const torch::Tensor& tensor_data = PyObjectToTensor(py_tensor);
        SPDLOG_INFO("put tensor {} dtype: {}", tensor_data.sizes().size(), toString(tensor_data.scalar_type()));
        table_data_[seq_id][tensor_key] = std::make_shared<torch::Tensor>(tensor_data.clone());
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error in put: {}", e.what());
        return false;
    }
}

bool InMemoryTensorTable::Get(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& py_tensor) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    const torch::Tensor& target_tensor = PyObjectToTensor(py_tensor);
    auto tensor_shards = GetTensorShards(tensor_key, seq_id, target_tensor);
    if (tensor_shards.empty()) {
        return false;
    }

    for (auto& pair : tensor_shards) {
        CopyTensorWithShardedKeysUnsafe(pair.first, pair.second, tensor_key, target_tensor);
    }
    cudaDeviceSynchronize();
    return true;
}

pybind11::object
InMemoryTensorTable::GetTensor(int64_t seq_id, const ShardedKey& tensor_key, const TorchTensorMeta& tensor_meta) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    auto target_tensor = torch::zeros(
        tensor_meta.size,
        torch::TensorOptions().dtype(tensor_meta.dtype).device(tensor_meta.device).requires_grad(false));
    auto tensor_shards = GetTensorShards(tensor_key, seq_id, target_tensor);
    if (tensor_shards.empty()) {
        return pybind11::none();
    }

    for (auto& pair : tensor_shards) {
        CopyTensorWithShardedKeysUnsafe(pair.first, pair.second, tensor_key, target_tensor);
    }
    cudaDeviceSynchronize();
    return TensorToPyObject(target_tensor);
}

bool InMemoryTensorTable::MultiPut(
    int64_t seq_id, const std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_list) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    try {
        auto& seq_data = table_data_[seq_id];
        for (const auto& pair : tensor_list) {
            const torch::Tensor& tensor_data = PyObjectToTensor(pair.second);
            SPDLOG_INFO("multi_put tensor {}", tensor_data.sizes().size());
            seq_data[pair.first] = std::make_shared<torch::Tensor>(tensor_data.clone());
        }
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error in multi_put: {}", e.what());
        return false;
    }
}

bool InMemoryTensorTable::MultiGet(int64_t seq_id, std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_list) {
    bool found_any = false;
    try {
        for (auto& pair : tensor_list) {
            ShardedKey key = pair.first;
            if (Get(seq_id, key, pair.second)) {
                found_any = true;
            } else {
                SPDLOG_ERROR("tensor {} not found", pair.first.key);
                return false;
            }
        }
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error in multi_get: {}", e.what());
        return false;
    }

    return found_any;
}

std::vector<std::pair<ShardedKey, pybind11::object>> InMemoryTensorTable::MultiGetTensor(
    int64_t seq_id, const std::vector<std::pair<ShardedKey, TorchTensorMeta>>& tensor_meta_list) {
    std::vector<std::pair<ShardedKey, pybind11::object>> ret;

    for (const auto& meta_pair : tensor_meta_list) {
        const ShardedKey& tensor_key = meta_pair.first;
        const TorchTensorMeta& tensor_meta = meta_pair.second;
        auto tensor = GetTensor(seq_id, tensor_key, tensor_meta);
        if (tensor.is_none()) {
            continue;
        }
        ret.emplace_back(tensor_key, tensor);
    }

    return ret;
}

void InMemoryTensorTable::Complete(int64_t seq_id) {
    // For in-memory table, complete is essentially a no-op
    // All operations are synchronous and immediately committed

    // Optional: Log completion for debugging
    SPDLOG_INFO("Complete called for seq_id: {}", seq_id);

    // In a more complex implementation, this might:
    // - Flush any pending writes
    // - Mark the sequence as completed
    // - Trigger cleanup or persistence operations
}

std::vector<std::pair<std::string, TorchTensorMeta>> InMemoryTensorTable::ScanTensorMeta(int64_t seq_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    std::vector<std::pair<std::string, TorchTensorMeta>> result;

    auto seq_it = table_data_.find(seq_id);
    if (seq_it == table_data_.end()) {
        return result; // Return empty vector if seq_id not found
    }

    for (const auto& tensor_pair : seq_it->second) {
        const ShardedKey& sharded_key = tensor_pair.first;
        const torch::Tensor& tensor = *tensor_pair.second;

        // Create TorchTensorMeta from the stored tensor
        TorchTensorMeta meta{tensor.scalar_type(), {}, tensor.device()};

        // Convert tensor sizes to std::vector<int64_t>
        auto tensor_sizes = tensor.sizes();
        meta.size.clear();
        meta.size.reserve(tensor_sizes.size());
        for (int64_t size : tensor_sizes) {
            meta.size.push_back(size);
        }

        // Add to result using the key string
        result.emplace_back(sharded_key.key, meta);
    }

    return result;
}

std::vector<std::pair<ShardedKey, torch::Tensor>> InMemoryTensorTable::GetTensorShards(
    const ShardedKey& sharded_key, int64_t seq_id, const torch::Tensor& target_tensor) {
    std::vector<std::pair<ShardedKey, torch::Tensor>> ret;
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    auto seq_it = table_data_.find(seq_id);
    if (seq_it == table_data_.end()) {
        return ret; // Return empty vector if seq_id not found
    }

    for (const auto& tensor_pair : seq_it->second) {
        const ShardedKey& cached_sharded_key = tensor_pair.first;
        const torch::Tensor& tensor = *tensor_pair.second;
        // Verify the tensor matches the expected metadata
        if (tensor.scalar_type() != target_tensor.scalar_type()) {
            throw std::runtime_error("Tensor dtype mismatch for key " + cached_sharded_key.key);
        }

        if (cached_sharded_key.key == sharded_key.key) {
            ret.emplace_back(cached_sharded_key, tensor);
        }
    }

    return ret;
}
} // namespace astate
