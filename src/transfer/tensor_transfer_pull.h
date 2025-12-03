#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <httplib.h>

#include <spdlog/spdlog.h>

#include "common/metric_utils.h"
#include "common/option.h"
#include "common/thread_pool.h"
#include "common/time_utils.h"
#include "core/atensor.h"
#include "core/atensor_storage.h"
#include "core/shardedkey.h"
#include "discovery/discovery_manager.h"
#include "protocol/messages.h"
#include "transfer/tensor_transfer_service.h"
#include "transfer/types.h"
#include "transport/base_transport.h"
#include "transport/rdma_transporter.h"

namespace astate {
/*
 * TensorTransferPull is a class that implements the TensorTransferService interface for pulling model weights.
 * It uses rdma data transport to pull model weights from remote nodes, while using control transport to send control
 * messages.
 */
constexpr int64_t INIT_SEQ_ID = -1;

constexpr const char* TENSOR_RDMA_META_REQUEST = "publish_tensor_rdma_meta";
constexpr const char* WEIGHT_READY_REQUEST = "weight_ready";
constexpr const char* WEIGHT_CONSUMED_REQUEST = "weight_consumed";

class TensorTransferPull : public TensorTransferService {
 public:
    TensorTransferPull();
    explicit TensorTransferPull(ATensorStorageCtx* ctx);
    ~TensorTransferPull() override;

    bool Start(const Options& options, const AParallelConfig& parallel_config) override;
    void Stop() override;
    [[nodiscard]] bool IsRunning() const override;

    bool Put(int64_t seq_id, const ShardedKey& tensor_key, const ATensor& atensor) override;
    bool MultiPut(int64_t seq_id, const std::vector<std::pair<ShardedKey, ATensor>>& atensors) override;

    bool Get(int64_t seq_id, const ShardedKey& tensor_key, ATensor& atensor) override;
    bool MultiGet(int64_t seq_id, std::vector<std::pair<ShardedKey, ATensor>>& atensors) override;

    bool
    RawGet(int64_t seq_id, const ATStorage& astorage, const NodeInfo& node_info, const void* remote_addr, size_t len)
        override;

    bool PreRegisterMemory(ATStorage& atensor_storage) override;

    void Complete() override;

    void SetPeerHosts(const std::vector<NodeInfo>& peer_hosts);

    [[nodiscard]] std::vector<std::pair<ShardedKey, ATensor>>
    GetAllTensorShards(int64_t seq_id, std::function<bool(const ShardedKey&)> filter) override;

    [[nodiscard]] std::vector<CompactTensorInfo>
    GetCompactTensorInfos(int64_t seq_id, std::unordered_map<ShardedKey, ATensor, ShardedKeyHash> atensors) override;

 protected:
    ATensorStorageCtx* ctx_ = nullptr;

    bool is_debug_mode_{false};

    std::unique_ptr<RDMATransporter> data_rdma_transport_;
    std::unique_ptr<BaseControlTransport> control_transport_;
    std::unique_ptr<DiscoveryManager> discovery_manager_;

    AParallelConfig parallel_config_{};

    NodeInfo local_node_info_;
    std::vector<NodeInfo> peer_hosts_; // hosts of the remote peers
    std::vector<NodeInfo> group_hosts_; // hosts in the same group
    ARole role_{};

    std::mutex ctrl_message_mutex_;
    std::mutex data_mutex_;

    // std::unique_ptr<MutexWaitQueueThreadPool> thread_pool_;
    std::unique_ptr<ThreadPool> thread_pool_;

    std::atomic<uint8_t> current_data_operation_{NO_OP};

    int64_t current_seq_id_{INIT_SEQ_ID};

    bool is_publish_meta_{false};
    int64_t last_completed_seq_id_{INIT_SEQ_ID};

    std::unordered_set<NodeInfo, NodeInfoHash> consumed_nodes_;

    // Remote tensor meta cache
    TransferCache remote_tensor_cache_; // seq_id -> tensor_transfer_meta collection
    // Record which nodes have all data ready
    std::unordered_set<NodeInfo, NodeInfoHash> ready_nodes_;
    // Waiting timeout for tensor ready / sequence ready
    int64_t tensor_ready_timeout_ms_{};

    // Random number generator for load balancing
    std::random_device rd_;
    std::mt19937 random_gen_;

    // skip rdma exception for test environment when rdma not working
    bool skip_rdma_exception_for_test_{false};

    bool enable_log_tensor_meta_{false};
    std::shared_ptr<PerfMetricsController> perf_metrics_controller_;
    uint64_t perf_stats_interval_ms_{};
    std::unordered_map<uint64_t, size_t> throughput_stats_;
    std::unordered_map<std::string, std::unordered_map<uint64_t, size_t>> throughput_stats_per_host_;
    std::mutex throughput_stats_mutex_;

    bool enable_local_cache_prefetch_{false};

    // Send control messages when sync model weights
    bool SendTensorRDMAMeta(const TensorRDMAMetaPublishMessage& meta);
    bool SendWeightReady(const WeightReadyMessage& msg);
    bool SendWeightConsumed(const WeightConsumedMessage& msg);

    // Receive control messages when sync model weights
    ResponseStatus HandleTensorRDMAMeta(const std::string& request, const void* message, size_t message_size);
    ResponseStatus HandleWeightReady(const std::string& request, const void* message, size_t message_size);
    ResponseStatus HandleWeightConsumed(const std::string& request, const void* message, size_t message_size);

    void RegisterHandlers() {
        control_transport_->RegisterHandler(
            TENSOR_RDMA_META_REQUEST, [this](const std::string& request, const void* message, size_t message_size) {
                return this->HandleTensorRDMAMeta(request, message, message_size);
            });
        control_transport_->RegisterHandler(
            WEIGHT_READY_REQUEST, [this](const std::string& request, const void* message, size_t message_size) {
                return this->HandleWeightReady(request, message, message_size);
            });
        control_transport_->RegisterHandler(
            WEIGHT_CONSUMED_REQUEST, [this](const std::string& request, const void* message, size_t message_size) {
                return this->HandleWeightConsumed(request, message, message_size);
            });
    }

    bool ValidateSeqId(int64_t seq_id) {
        std::lock_guard<std::mutex> lock(ctrl_message_mutex_);
        if (seq_id != current_seq_id_) {
            SPDLOG_ERROR(
                "Cannot send weight ready message with mismatched seq_id: {}, "
                "current_seq_id: {}",
                seq_id,
                current_seq_id_);
            return false;
        }
        return true;
    };

    bool CheckAndUpdateCurrentSeqId(int64_t seq_id) {
        std::lock_guard<std::mutex> lock(ctrl_message_mutex_);
        if (current_seq_id_ == INIT_SEQ_ID) {
            current_seq_id_ = seq_id;
        } else if (seq_id != current_seq_id_) {
            SPDLOG_ERROR("The seq id [{}] mismatched current id [{}]", seq_id, current_seq_id_);
            return false;
        }
        return true;
    };

    bool WaitForAllTensorReady(const int64_t /*seq_id*/, int64_t max_wait_ms = 60000) {
        return WaitCondition(
            [this]() { return ready_nodes_.size() == peer_hosts_.size(); }, "wait_for_all_tensor_ready", max_wait_ms);
    };

    bool WaitForTensorReady(
        const int64_t seq_id, const ShardedKey& tensor_key, int max_wait_ms = 60000, int interval_ms = 1000) {
        if (is_publish_meta_) {
            {
                // If published meta before, use the last cache meta.
                std::lock_guard<std::mutex> lock(ctrl_message_mutex_);
                if (remote_tensor_cache_.find(seq_id) == remote_tensor_cache_.end()) {
                    auto cache_it = remote_tensor_cache_.find(last_completed_seq_id_);
                    if (cache_it == remote_tensor_cache_.end()) {
                        SPDLOG_ERROR(
                            "Cannot find remote_tensor_cache, using "
                            "key(last_seq_id)={}",
                            last_completed_seq_id_);
                        throw std::runtime_error(
                            "Cannot find remote_tensor_cache, using "
                            "key(last_seq_id)="
                            + std::to_string(last_completed_seq_id_));
                    }
                    TransferTensorMeta* transfer_meta = &(cache_it->second);
                    remote_tensor_cache_.emplace(seq_id, *transfer_meta);
                    // Remove the last cache meta.
                    remote_tensor_cache_.erase(last_completed_seq_id_);
                }
            }

            // TODO(root): Wait for all put nodes finished. Only need to wait the specified tensor ready in the
            // future.
            return WaitCondition(
                [this]() { return ready_nodes_.size() == peer_hosts_.size(); },
                "wait_for_all_tensor_ready",
                max_wait_ms);
        }
        return WaitCondition(
            [this, seq_id, tensor_key]() {
                auto transfer_meta = remote_tensor_cache_.find(seq_id);
                if (transfer_meta == remote_tensor_cache_.end()) {
                    return false;
                }
                if (transfer_meta->second.find(tensor_key) == transfer_meta->second.end()) {
                    return false;
                }
                return true;
            },
            "wait_for_tensor_ready",
            max_wait_ms,
            interval_ms);
    };

    // Send & receive control messages
    bool SendCtrlMessage(
        const std::string& request_name,
        int64_t seq_id,
        const void* message,
        size_t message_size,
        const NodeInfo& node_info);
    bool SendCtrlMessageToMultiPeers(
        const std::string& request_name,
        int64_t seq_id,
        const void* message,
        size_t message_size,
        const std::vector<NodeInfo>& node_infos);

    // Helper method to register memory with consistent error handling (throws on failure)
    void RegisterMemoryOrThrow(void* addr, size_t size, bool is_cuda, int device_index);

    void UpdateThroughputStatistic(const std::string& node_info, size_t tx_data_bytes) {
        if (!perf_metrics_controller_->IsPerfMetricsEnabled()) {
            return;
        }

        auto now = std::chrono::high_resolution_clock::now();
        uint64_t current_time_ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        uint64_t time_window_ms = current_time_ms - (current_time_ms % perf_stats_interval_ms_);

        std::lock_guard<std::mutex> lock(throughput_stats_mutex_);
        if (throughput_stats_.find(time_window_ms) == throughput_stats_.end()) {
            throughput_stats_[time_window_ms] = tx_data_bytes;
        }
        throughput_stats_[time_window_ms] += tx_data_bytes;

        if (throughput_stats_per_host_.find(node_info) == throughput_stats_per_host_.end()) {
            throughput_stats_per_host_[node_info] = std::unordered_map<uint64_t, size_t>();
        }
        auto& throughput_stats = throughput_stats_per_host_[node_info];
        if (throughput_stats.find(time_window_ms) == throughput_stats.end()) {
            throughput_stats[time_window_ms] = tx_data_bytes;
        } else {
            throughput_stats[time_window_ms] += tx_data_bytes;
        }
    }

    void LogRemoteTensorMeta() {
        if (!is_publish_meta_ && enable_log_tensor_meta_) {
            // print the remote_tensor_cache_
            auto it = remote_tensor_cache_.find(current_seq_id_);
            if (it == remote_tensor_cache_.end()) {
                SPDLOG_ERROR(
                    "Cannot find remote_tensor_cache, using "
                    "key(current_seq_id_)={}",
                    current_seq_id_);
            } else {
                auto& tensor_cache = it->second;
                SPDLOG_INFO("Remote tensors: size={}", tensor_cache.size());
                for (const auto& pair : tensor_cache) {
                    SPDLOG_INFO("  [{}]:", pair.first.ToString());
                    for (const auto& tensor_meta : pair.second) {
                        SPDLOG_INFO("    - {}", tensor_meta.ToString());
                    }
                }
            }
        }
    }

    void LogThroughputStatistic() {
        if (!perf_metrics_controller_->IsPerfMetricsEnabled()) {
            return;
        }

        SPDLOG_INFO("[Seq {}] throughput stats: ", current_seq_id_);
        uint64_t total_throughput = 0;
        for (auto& pair : throughput_stats_) {
            SPDLOG_INFO("  - {}: {} MB", pair.first, BYTES_TO_MB(pair.second));
            total_throughput += pair.second;
        }
        SPDLOG_INFO("[Seq {}] total throughput: {} MB", current_seq_id_, BYTES_TO_MB(total_throughput));
        throughput_stats_.clear();

        for (auto& pair : throughput_stats_per_host_) {
            SPDLOG_INFO("[Seq {} {}] throughput stats: ", current_seq_id_, pair.first);
            uint64_t throughput_per_host = 0;
            for (auto& throughput_pair : pair.second) {
                SPDLOG_INFO("  - {}: {} MB", throughput_pair.first, BYTES_TO_MB(throughput_pair.second));
                throughput_per_host += throughput_pair.second;
            }
            SPDLOG_INFO(
                "[Seq {} {}] total throughput: {} MB", current_seq_id_, pair.first, BYTES_TO_MB(throughput_per_host));
        }
        throughput_stats_per_host_.clear();
    }
};

} // namespace astate
