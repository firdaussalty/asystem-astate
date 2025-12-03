#pragma once

#include <exception>
#include <unordered_map>

#include "astate/sharded_key.h"
#include "common/lock_utils.h"
#include "common/metric_utils.h"
#include "common/numa_aware_allocator.h"
#include "common/option.h"
#include "common/string_utils.h"
#include "common/thread_pool.h"
#include "core/atensor.h"
#include "core/atensor_storage.h"
#include "core/shardedkey.h"
#include "core/tensor_sharded_ops.h"
#include "core/tensor_table.h"
#include "core/utils.h"

namespace astate {

// Remote implementation of TensorTable
class RemoteTensorTable : public TensorTable {
 public:
    RemoteTensorTable(const std::string& name, std::shared_ptr<ATensorStorageCtx> ctx);
    ~RemoteTensorTable() override = default;

    // Disable copy constructor and assignment operator
    RemoteTensorTable(const RemoteTensorTable&) = delete;
    RemoteTensorTable& operator=(const RemoteTensorTable&) = delete;
    RemoteTensorTable(RemoteTensorTable&&) = delete;
    RemoteTensorTable& operator=(RemoteTensorTable&&) = delete;

    bool Put(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& py_tensor) override;

    bool Get(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& py_tensor) override;

    bool MultiPut(int64_t seq_id, const std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_list) override;

    bool MultiGet(int64_t seq_id, std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_dict) override;

    // New interface methods - currently not implemented for RemoteTensorTable
    pybind11::object
    GetTensor(int64_t seq_id, const ShardedKey& tensor_key, const TorchTensorMeta& tensor_meta) override;

    std::vector<std::pair<ShardedKey, pybind11::object>> MultiGetTensor(
        int64_t seq_id, const std::vector<std::pair<ShardedKey, TorchTensorMeta>>& tensor_meta_list) override;

    void Complete(int64_t seq_id) override;

    std::vector<std::pair<std::string, TorchTensorMeta>> ScanTensorMeta(int64_t seq_id) override;

    /**
     * @brief [Receiver] Compact the small tensors, e.g. KB, into a tensor for better performance in further transfer.
     * @param seq_id step id for current inferencing.
     * @param small_tensors
     * @return True if success.
     */
    bool MultiGetCompactTensors(
        int64_t seq_id, const std::unordered_map<ShardedKey, const torch::Tensor&, ShardedKeyHash>& small_tensors);

    /**
     * @brief [Receiver] Prefetch the tensors which are marked as cached in receiver, and this method is called in
     * transfer service, e.g. when inference node has received all tensor ready messages from senders.
     * Moreover, the number of tensors supposed to be cached are calculated based on the configuration of
     * cache buffer size "TRANSFER_ENGINE_LOCAL_CACHE_TENSORS_SIZE".
     * @param seq_id step id for current inferencing.
     */
    void PrefetchCachedTensors(int64_t seq_id) override;

 private:
    bool is_debug_mode_{false};
    std::shared_ptr<ATensorStorageCtx> ctx_;

    // Mutex for thread-safe access
    std::mutex mutex_;
    RWSpinLock rw_spin_lock_;

    std::mutex tensor_meta_mutex_;
    std::shared_ptr<std::vector<std::pair<ShardedKey, ATensor>>> tensor_meta_list_ = nullptr;
    GlobalParallelConfig training_parallel_config_;
    GlobalParallelConfig inference_parallel_config_;

    bool enable_write_gpu_async_copy_{false};
    bool enable_read_gpu_async_copy_{false};
    // Copy thread pool for parallel tensor operations
    std::unique_ptr<astate::CUDAStreamResourceThreadPool<torch::Tensor>> copy_thread_pool_;
    long copy_bucket_mem_size_;
    // Copy thread pool for large tensors
    std::unique_ptr<astate::CUDAStreamResourceThreadPool<torch::Tensor>> copy_large_thread_pool_;
    long copy_large_bucket_mem_size_;

    // tmp counter for copy task
    int32_t copy_task_counter_ = 0;
    int32_t copy_large_task_counter_ = 0;

    long small_tensor_compact_cache_size_ = 0;
    long small_tensor_size_ = 0;
    long small_tensor_compact_cache_offset_ = 0;
    torch::Tensor small_tensor_compact_cache_;
    std::vector<torch::Tensor> small_tensor_compact_cache_list_;

    std::unique_ptr<astate::CUDAStreamThreadPool> thread_pool_;

    // The cached remote tensor shards for current seq
    bool enable_local_cache_prefetch_ = false;
    // TODO(root): The cached remote tensor shards should be updated while the remote tensors were reallocated.
    std::unordered_map<ShardedKey, std::vector<ShardedATensorTuple>, ShardedKeyHash> shard_mapping_;
    // Cached local tensors which could be updated before the reading request submitted from inference engine.
    // The bool variable is whether the tensor is cached for current seq.
    TensorExtDict<bool> local_cached_tensors_;
    long local_cached_tensors_size_ = 0;
    long local_cached_tensors_size_limit_;
    std::mutex local_cached_tensors_mutex_;

    ATensorDict cached_small_tensor_shards_;
    std::vector<CompactTensorInfo> compact_tensor_infos_;

    // Mapping from training tensor keys to their sharded inference tensor keys, e.g. column parallel(TP) tensors
    TensorShardingMap tensor_resharding_map_;
    std::unordered_set<std::string> tensor_resharding_key_set_;
    // Local tensor mapping for storing tensor copies: ShardedKey -> (ShardedKey, shared_ptr<Tensor>)
    std::unordered_map<ShardedKey, std::pair<ShardedKey, std::shared_ptr<torch::Tensor>>, ShardedKeyHash>
        local_tensor_mapping_;

    // for reading tensors
    ATensorDict reading_tensors_meta_;

    // Track last seq_id for logging optimization
    std::atomic<int64_t> last_logged_seq_id_{-1};

    std::shared_ptr<PerfMetricsController> perf_metrics_controller_;

    bool pinned_memory_enabled_ = true; // Pinned memory is only avaiable in GPU env, not in CPU only env.

    astate::NumaAwareAllocator numa_allocator_;

    std::atomic_bool enable_numa_allocation_{true};

    bool enable_log_tensor_meta_{false};

    /**
     * @brief Check if logging should be performed for this seq_id
     * @param seq_id Current sequence ID
     * @return true if logging should be performed (seq_id changed), false otherwise
     */
    bool ShouldLogForSeqId(int64_t seq_id) {
        int64_t expected = last_logged_seq_id_.load();
        if (seq_id != expected) {
            // Ensure only one thread successfully updates the last logged seq_id
            if (last_logged_seq_id_.compare_exchange_strong(expected, seq_id)) {
                return true; // Successfully updated, should log
            }
        }
        return false; // seq_id is the same or update failed, do not log
    }

    /**
     * @brief Create a zero-initialized tensor with specified parameters
     * @param sizes Tensor dimensions
     * @param dtype Data type (default: torch::ScalarType::Byte)
     * @param device_type Device type (CPU/CUDA)
     * @param requires_grad Whether gradient is required (default: false)
     * @param pinned_memory Whether to use pinned memory (default: true)
     * @return Zero-initialized tensor
     */
    torch::Tensor CreateZeroTensor(
        torch::IntArrayRef sizes,
        torch::ScalarType dtype,
        torch::DeviceType device_type,
        bool requires_grad = false,
        bool pinned_memory = true);

    static size_t AlignStorageOffset(int64_t offset, torch::ScalarType dtype) {
        auto item_size = GetItemSizeFromDtype(dtype);
        if (offset % item_size == 0) {
            return offset;
        }
        return (offset + (item_size - 1)) / item_size * item_size;
    }

    /**
     * @brief [Sender] Get or create local tensor copy for tensor data.
     * @param tensor_key Sharded key.
     * @param source_tensor Source tensor, used to determine the shape, type and device of the copy.
     * @return std::shared_ptr<torch::Tensor> shared pointer to the local copy.
     */
    std::shared_ptr<torch::Tensor>
    GetOrCreateLocalTensorCopy(const ShardedKey& tensor_key, const torch::Tensor& source_tensor);

    /**
     * @brief [Sender] Reshard the source tensor, e.g. column parallel tensors, into several partial tensors which have
     * exclusive cpu/gpu memory space.
     * @param tensor_key Sharded key of source tensor.
     * @param source_tensor Source torch tensor.
     * @return Resharding info.
     */
    std::vector<ReshardingInfo> ReshardTensor(const ShardedKey& tensor_key, const torch::Tensor& source_tensor) const;

    /**
     * @brief [Sender] Try to get the local reshard tensors. If the tensor was needed to reshard and found in the
     * resharding map, the data of source tensor will be copied into several resharded tensors before sending to
     * receiver according to the reshard info, e.g., ajusted size and offset.
     * If the tensor not needs to be resharded, the origin one will be returned.
     * @param tensor_key Sharded key of source tensor.
     * @param source_tensor Source torch tensor.
     * @return The resharded tensors.
     */
    std::unordered_map<ShardedKey, torch::Tensor, ShardedKeyHash>
    GetOrCreateLocalReshardTensors(const ShardedKey& tensor_key, const torch::Tensor& source_tensor);

    /**
     * @brief [Receiver] Try to get the meta of remote tensors which have the data related to the target tensor.
     * @param shardedKey Sharded key of target tensor.
     * @param seq_id Step id.
     * @param target_tensor Target ATensor to read data.
     * @param try_prune_redundant_shard Indicate whether to prune the redundancy data in current tensor.
     @ @return The sharding info of remote tensors which will be used for reading.
     */
    const std::vector<ShardedATensorTuple>& GetRemoteTensorShards(
        const ShardedKey& sharded_key, int64_t seq_id, const ATensor& target_tensor, bool try_prune_redundant_shard);

    const std::vector<ShardedATensorTuple>&
    GetRemoteTensorShards(const ShardedKey& sharded_key, int64_t seq_id, const ATensor& target_tensor);

    /**
     * @brief [Receiver] Get the local cached tensors which were prefetch before reading. If local cached tensor was not
     * found and the limit size of cached buffer was not exceeded yet, create the new cached tensor and store it.
     * @param sharded_key Sharded key of target tensor.
     * @param target_tensor Target torch tensor.
     * @return The cached local tensor according the sharded key, or nullptr if not cached.
     */
    std::shared_ptr<torch::Tensor>
    GetLocalPrefetchCachedTensor(const ShardedKey& sharded_key, const torch::Tensor& target_tensor);

    /**
     * @brief [Receiver] Read the tensors from remote or local prefetch cache.
     * @param seq_id Step id of current inferencing.
     * @param shardedKey Sharded key of target tensor.
     * @param target_tensor Target torch tensor.
     * @param local_cache Temp local torch which is used to store the tensor data read.
     * @return The tensors which will be copied into the target tensor.
     */
    std::shared_ptr<TensorDict> ReadTensors(
        int64_t seq_id, const ShardedKey& sharded_key, const torch::Tensor& target_tensor, torch::Tensor& local_cache);

    /**
     * @brief [Receiver] Submit the async task to read the data for the specified target tensor.
     */
    std::future<void>
    SubmitTransferTask(int64_t seq_id, const ShardedKey& sharded_key, const torch::Tensor& target_tensor);

    // [Receiver] Record the tensor metas in first step.
    void UpdateReadingTensorsMeta(const int64_t seq_id, const ShardedKey& tensor_key, const torch::Tensor& atensor) {
        // only update the reading_tensors_meta_ when first step (-1)
        if (seq_id != -1) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        reading_tensors_meta_[tensor_key] = *TensorToATensor(atensor);
    }

    // [Receiver] Log the tensor metas in first step.
    void LogReadingTensorsMeta(const int64_t seq_id) {
        // only log the reading_tensors_meta_ when first step (-1)
        if (seq_id != -1) {
            return;
        }

        SPDLOG_INFO("Local reading tensors: size={}", reading_tensors_meta_.size());
        for (const auto& pair : reading_tensors_meta_) {
            SPDLOG_INFO("  [{}]:", pair.first.ToString());
            SPDLOG_INFO("    - {}", pair.second.GetTensorInfo());
        }
    }

    static GlobalParallelConfig ParseParallelConfig(const std::vector<std::string>& config_str) {
        GlobalParallelConfig config;
        for (const auto& str : config_str) {
            auto pos = str.find('=');
            if (pos != std::string::npos) {
                std::string key = str.substr(0, pos);
                int32_t value = std::stoi(str.substr(pos + 1));
                if (key == "dp") {
                    config.dp_size = value;
                } else if (key == "tp") {
                    config.tp_size = value;
                } else if (key == "pp") {
                    config.pp_size = value;
                } else if (key == "cp") {
                    config.cp_size = value;
                } else if (key == "ep") {
                    config.ep_size = value;
                } else if (key == "etp") {
                    config.etp_size = value;
                }
            }
        }
        return config;
    }

    void UpdateGlobalParallelConfig(const Options& options) {
        auto train_config_str
            = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_TRAINING_PARALLEL_CONFIG);
        auto infer_config_str
            = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_INFERENCE_PARALLEL_CONFIG);
        auto reshard_keys_str
            = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_TENSOR_RESHARDING_KEYS);

        try {
            training_parallel_config_ = ParseParallelConfig(train_config_str);
            inference_parallel_config_ = ParseParallelConfig(infer_config_str);
            SPDLOG_INFO(
                "Parsed parallel config[T={}, I={}]: train={}, infer={}",
                ToString(train_config_str),
                ToString(infer_config_str),
                training_parallel_config_.ToString(),
                inference_parallel_config_.ToString());

            if (!reshard_keys_str.empty()) {
                for (const auto& key : reshard_keys_str) {
                    if (!key.empty()) {
                        tensor_resharding_key_set_.insert(key);
                    }
                }
                SPDLOG_INFO("Parsed tensor sharding map keys: {}", ToString(reshard_keys_str));
            }
        } catch (std::exception& e) {
            SPDLOG_ERROR(
                "Failed to parse parallel config: train={}, infer={}, error={}",
                ToString(train_config_str),
                ToString(infer_config_str),
                e.what());
        }
    }

    bool IsTensorReshardingKey(const ShardedKey& tensor_key) {
        return std::any_of(
            tensor_resharding_key_set_.begin(),
            tensor_resharding_key_set_.end(),
            [&tensor_key](const auto& reshard_key) { return tensor_key.key.find(reshard_key) != std::string::npos; });
    }
};

} // namespace astate
