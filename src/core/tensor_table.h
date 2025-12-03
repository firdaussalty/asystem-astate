#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "astate/tensor_table.h"
#include "core/atensor.h"
#include "core/shardedkey.h"

namespace astate {

// Define TensorDict type
using TensorDict = std::unordered_map<ShardedKey, std::shared_ptr<torch::Tensor>, ShardedKeyHash>;
template <typename T>
using TensorExtDict = std::unordered_map<ShardedKey, std::pair<std::shared_ptr<torch::Tensor>, T>, ShardedKeyHash>;
using ShardedTensor = std::pair<ShardedKey, std::shared_ptr<torch::Tensor>>;
using TensorKeyList = std::vector<ShardedKey>;

using ATensorDict = std::unordered_map<ShardedKey, ATensor, ShardedKeyHash>;
using ShardedATensor = std::pair<ShardedKey, std::shared_ptr<ATensor>>;
using ShardedATensorTuple = std::tuple<ShardedKey, ShardedKey, ATensor>;

// <sharded_key, dim_index, start, offset>
using ReshardingInfo = std::tuple<ShardedKey, int, size_t, size_t>;
// Mapping from a tensor key to its shard keys, e.g. training tensors to the sharded inference tensors
using TensorShardingMap = std::unordered_map<ShardedKey, std::vector<ReshardingInfo>, ShardedKeyHash>;

#define GetShardedKey(reshard_info) std::get<0>(reshard_info)
#define SetShardedKey(reshard_info, value) std::get<0>(reshard_info) = value
#define GetDimIndex(reshard_info) std::get<1>(reshard_info)
#define SetDimIndex(reshard_info, value) std::get<1>(reshard_info) = value
#define GetStart(reshard_info) std::get<2>(reshard_info)
#define SetStart(reshard_info, value) std::get<2>(reshard_info) = value
#define GetOffset(reshard_info) std::get<3>(reshard_info)
#define SetOffset(reshard_info, value) std::get<3>(reshard_info) = value

class InMemoryTensorTable : public TensorTable {
 public:
    explicit InMemoryTensorTable(const std::string& name);
    ~InMemoryTensorTable() override = default;

    // Disable copy constructor and assignment operator
    InMemoryTensorTable(const InMemoryTensorTable&) = delete;
    InMemoryTensorTable& operator=(const InMemoryTensorTable&) = delete;
    InMemoryTensorTable(InMemoryTensorTable&&) = delete;
    InMemoryTensorTable& operator=(InMemoryTensorTable&&) = delete;

    bool Put(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& tensor) override;
    bool Get(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& py_tensor) override;
    pybind11::object
    GetTensor(int64_t seq_id, const ShardedKey& tensor_key, const TorchTensorMeta& tensor_meta) override;
    bool MultiPut(int64_t seq_id, const std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_list) override;
    bool MultiGet(int64_t seq_id, std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_list) override;
    std::vector<std::pair<ShardedKey, pybind11::object>> MultiGetTensor(
        int64_t seq_id, const std::vector<std::pair<ShardedKey, TorchTensorMeta>>& tensor_meta_list) override;
    void Complete(int64_t seq_id) override;
    std::vector<std::pair<std::string, TorchTensorMeta>> ScanTensorMeta(int64_t seq_id) override;

    void PrefetchCachedTensors(int64_t /*seq_id*/) override {
        throw std::runtime_error("prefetch_cached_tensors is not supported for in-memory table");
    }

 private:
    // Internal storage structure: seq_id -> (tensor_key -> tensor)
    using TableData = std::unordered_map<int64_t, TensorDict>;
    TableData table_data_;

    // Mutex for thread-safe access
    mutable std::recursive_mutex mutex_;

    std::vector<std::pair<ShardedKey, torch::Tensor>>
    GetTensorShards(const ShardedKey& sharded_key, int64_t seq_id, const torch::Tensor& target_tensor);
};

} // namespace astate
