#include "tensor_transfer_pull.h"

#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>

#include <spdlog/spdlog.h>

#include "common/option.h"
#include "common/string_utils.h"
#include "common/thread_pool.h"
#include "common/time_utils.h"
#include "core/atensor.h"
#include "core/shardedkey.h"
#include "core/utils.h"
#include "discovery/discovery_manager.h"
#include "protocol/messages.h"
#include "tensor_transfer_service.h"
#include "transport/base_transport.h"
#include "transport/brpc_transport.h"
#include "transport/rdma_transporter.h"
#include "types.h"

namespace astate {

TensorTransferPull::TensorTransferPull()
    : random_gen_(rd_()) {
    data_rdma_transport_ = std::make_unique<RDMATransporter>();
    control_transport_ = std::make_unique<BrpcTransport>();
}

TensorTransferPull::TensorTransferPull(ATensorStorageCtx* ctx)
    : ctx_(ctx),
      random_gen_(rd_()) {
    data_rdma_transport_ = std::make_unique<RDMATransporter>();
    // control_transport_ = std::make_unique<HTTPTransporter>();
    control_transport_ = std::make_unique<BrpcTransport>();
}

TensorTransferPull::~TensorTransferPull() {
    if (IsRunning()) {
        Stop();
    }
}

inline void TensorTransferPull::RegisterMemoryOrThrow(void* addr, size_t size, bool is_cuda, int device_index) {
    try {
        bool success = data_rdma_transport_->RegisterMemory(addr, size, is_cuda, device_index);
        if (!success) {
            throw std::runtime_error("Failed to register memory");
        }
    } catch (const std::runtime_error& e) {
        if (!skip_rdma_exception_for_test_) {
            SPDLOG_ERROR(
                "Failed to register memory, addr: {}, size: {}, is_cuda: {}, "
                "device_index: {}, error: {}",
                addr,
                size,
                is_cuda,
                device_index,
                e.what());
            throw;
        }
        // In test mode, silently ignore the exception
    }
}

bool TensorTransferPull::Start(const Options& options, const AParallelConfig& parallel_config) {
    is_debug_mode_ = GetOptionValue<bool>(options, ASTATE_DEBUG_MODE);

    bool init_success = true;
    try {
        parallel_config_ = parallel_config;
        role_ = parallel_config.role;
        tensor_ready_timeout_ms_ = GetOptionValue<int>(options, TRANSFER_ENGINE_SERVICE_TENSOR_READY_TIMEOUT_MS);
        int read_thread_num = GetOptionValue<int>(options, TRANSFER_ENGINE_READ_THREAD_NUM);
        thread_pool_ = std::make_unique<ThreadPool>(read_thread_num);
        skip_rdma_exception_for_test_ = GetOptionValue<bool>(options, TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION);
        SPDLOG_INFO(
            "Start TensorTransferPull with role: {}, read thread num: {}, "
            "tensor ready timeout ms: {}",
            RoleToString(role_),
            read_thread_num,
            tensor_ready_timeout_ms_);
        SPDLOG_INFO("Parallel Config: {}", parallel_config_.ToString());
        if (skip_rdma_exception_for_test_) {
            SPDLOG_WARN(
                "Skip RDMA exception for test: {}, this is for test only, DO "
                "NOT USE IN PRODUCTION",
                skip_rdma_exception_for_test_);
        }

        // Start RDMA data transport service
        init_success &= data_rdma_transport_->Start(options, parallel_config);
        if (init_success) {
            SPDLOG_INFO("RDMA transport service started successfully.");
        }

        // Start control transport service
        RegisterHandlers();
        init_success &= control_transport_->Start(options);
        if (init_success) {
            SPDLOG_INFO("TensorTransferPull service started successfully.");
            local_node_info_ = NodeInfo{
                data_rdma_transport_->GetLocalServerName(),
                data_rdma_transport_->GetBindPort(),
                control_transport_->GetBindPort()};
        }

        bool skip_discovery = GetOptionValue<bool>(options, TRANSFER_ENGINE_SERVICE_SKIP_DISCOVERY);
        if (!skip_discovery) {
            discovery_manager_ = DiscoveryManager::CreateFromOptions(
                options, parallel_config, data_rdma_transport_->GetBindPort(), control_transport_->GetBindPort());
            discovery_manager_->Start();
            discovery_manager_->RegisterCurrentNode();
            auto nodes = discovery_manager_->DiscoverAllNodes();
            SPDLOG_INFO("discovery nodes size: {}", nodes.size());
            for (const auto& node : nodes) {
                if (node.role != role_) {
                    peer_hosts_.push_back(NodeInfo{
                        node.node_info.hostname_or_ip, node.node_info.rdma_port, node.node_info.ctrl_flow_port});
                } else {
                    group_hosts_.push_back(NodeInfo{
                        node.node_info.hostname_or_ip, node.node_info.rdma_port, node.node_info.ctrl_flow_port});
                }
            }
        } else {
            auto peers_host = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_PEERS_HOST);
            auto group_host = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_GROUP_HOST);
            peer_hosts_ = NodeInfo::GetNodeInfos(peers_host);
            group_hosts_ = NodeInfo::GetNodeInfos(group_host);
            SPDLOG_INFO(
                "Skip discovery, this use only for test, peers_host: {}, "
                "group_host: {}",
                ToString(peers_host),
                ToString(group_host));
        }

        enable_log_tensor_meta_ = GetOptionValue<bool>(options, TRANSFER_ENGINE_LOG_TENSOR_META);
        perf_metrics_controller_ = std::make_shared<PerfMetricsController>("tensor_transfer_pull_service", options);
        perf_stats_interval_ms_ = GetOptionValue<int64_t>(options, TRANSFER_ENGINE_PERF_STATS_INTERVAL_MS);
        SPDLOG_INFO(
            "Enbale log tensor meta: {}, Enable perf metrics: {}, perf stats "
            "interval ms: {}",
            enable_log_tensor_meta_,
            perf_metrics_controller_->IsPerfMetricsEnabled(),
            perf_stats_interval_ms_);

        enable_local_cache_prefetch_ = GetOptionValue<bool>(options, TRANSFER_ENGINE_ENABLE_LOCAL_CACHE_PREFETCH);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to start tensor transfer service [PULL]: {}", e.what());
        init_success = false;
    }
    SPDLOG_INFO("peer_hosts_: {}, group_hosts_: {}", peer_hosts_.size(), group_hosts_.size());

    SPDLOG_INFO("Tensor transfer pull service started successfully.");
    return init_success;
}

void TensorTransferPull::Stop() {
    if (data_rdma_transport_ != nullptr && data_rdma_transport_->IsRunning()) {
        data_rdma_transport_->Stop();
    }
    if (control_transport_ != nullptr && control_transport_->IsRunning()) {
        control_transport_->Stop();
    }

    SPDLOG_INFO("Succesfully stop all transport services.");
}

bool TensorTransferPull::IsRunning() const {
    return (data_rdma_transport_ != nullptr && data_rdma_transport_->IsRunning())
        && (control_transport_ != nullptr && control_transport_->IsRunning());
}

bool TensorTransferPull::Put(int64_t seq_id, const ShardedKey& tensor_key, const ATensor& atensor) {
    // Check if service is running before proceeding with operations
    if (!IsRunning()) {
        SPDLOG_WARN("Service is not running, Put operation cannot proceed");
        return false;
    }

    if (!atensor.IsValid()) {
        SPDLOG_ERROR("Invalid tensor: {}", atensor.GetTensorInfo());
        return false;
    }

    SetWrite(current_data_operation_);
    if (!CheckAndUpdateCurrentSeqId(seq_id)) {
        return false;
    }

    // Register memory for further rdma transport
    {
        // TODO(root): Temporary lock to avoid register memory concurrently.
        // It could be removed after libutrans supports the concurrent registration of memory.
        std::lock_guard<std::mutex> lock(ctrl_message_mutex_);
        RegisterMemoryOrThrow(
            atensor.storage.data,
            atensor.storage.GetStorageDataSize(),
            atensor.storage.device.device_type == ATDeviceType::CUDA,
            atensor.storage.device.device_index);
    }

    // Build RDMA meta message
    TensorMemoryRDMAInfo rdma_info{atensor.storage.data, atensor.storage.GetStorageDataSize(), "", atensor};

    TensorRDMAMetaPublishMessage meta_msg;
    meta_msg.seq_id = seq_id;
    meta_msg.node_info = local_node_info_;
    meta_msg.tensor_rdma_metas[tensor_key] = rdma_info;

    // Send the tensor rdma meta infos to all peers
    // TODO(root): send weight ready message to all peers directly instead of publishing metas first.
    if (!is_publish_meta_) {
        return SendTensorRDMAMeta(meta_msg);
    }
    return true;
}

bool TensorTransferPull::MultiPut(int64_t seq_id, const std::vector<std::pair<ShardedKey, ATensor>>& atensors) {
    if (!IsRunning()) {
        SPDLOG_WARN("Service is not running, MultiPut operation cannot proceed");
        return false;
    }

    if (atensors.empty()) {
        SPDLOG_WARN("No tensors to put for seq_id: {}", seq_id);
        return false;
    }

    SetWrite(current_data_operation_);
    if (!CheckAndUpdateCurrentSeqId(seq_id)) {
        return false;
    }

    // Send the tensor rdma meta infos to all peers if the metas have not been published yet.
    // TODO(root): handle the case that the metas were changed.
    if (!is_publish_meta_) {
        TensorRDMAMetaPublishMessage meta_msg;
        meta_msg.seq_id = seq_id;
        meta_msg.node_info = local_node_info_;

        for (const auto& pair : atensors) {
            // Register memory for further rdma transport

            // TODO(root): Temporary lock to avoid register memory concurrently.
            // It could be removed after libutrans supports the concurrent registration of memory.
            std::lock_guard<std::mutex> lock(ctrl_message_mutex_);

            RegisterMemoryOrThrow(
                pair.second.storage.data,
                pair.second.storage.GetStorageDataSize(),
                pair.second.storage.device.device_type == ATDeviceType::CUDA,
                pair.second.storage.device.device_index);

            // Build RDMA meta message
            TensorMemoryRDMAInfo rdma_info{
                pair.second.storage.data, pair.second.storage.GetStorageDataSize(), "", pair.second};
            meta_msg.tensor_rdma_metas[pair.first] = rdma_info;
        }

        return SendTensorRDMAMeta(meta_msg);
    }
    return true;
}

bool TensorTransferPull::Get(int64_t seq_id, const ShardedKey& tensor_key, ATensor& atensor) {
    auto start_time = std::chrono::high_resolution_clock::now();
    if (!atensor.IsValid()) {
        SPDLOG_ERROR("Invalid tensor: {}", atensor.GetTensorInfo());
        return false;
    }

    // Check if service is running before proceeding with operations
    if (!IsRunning()) {
        SPDLOG_WARN("Service is not running, Get operation cannot proceed");
        return false;
    }

    SetRead(current_data_operation_);
    if (!CheckAndUpdateCurrentSeqId(seq_id)) {
        return false;
    }

    RegisterMemoryOrThrow(
        atensor.storage.data,
        atensor.storage.GetStorageDataSize(),
        atensor.storage.device.device_type == ATDeviceType::CUDA,
        atensor.storage.device.device_index);
    auto register_end = std::chrono::high_resolution_clock::now();

    if (!WaitForTensorReady(seq_id, tensor_key, static_cast<int>(tensor_ready_timeout_ms_))) {
        return false;
    }
    auto wait_end = std::chrono::high_resolution_clock::now();

    auto cache_it = remote_tensor_cache_.find(seq_id);
    if (cache_it == remote_tensor_cache_.end()) {
        SPDLOG_ERROR("Tensor RDMA info not found for seq_id: {}", seq_id);
        throw std::runtime_error("illegal state: Tensor RDMA info not found "
                                 "with corresponding seq_id");
    }
    const auto* rdma_info_list = GetTensorRDMAInfoVector(tensor_key, cache_it->second);
    if (rdma_info_list == nullptr || rdma_info_list->empty()) {
        SPDLOG_ERROR("Tensor RDMA info not found for tensor_key: {}", tensor_key.key);
        throw std::runtime_error("illegal state: Tensor RDMA info not found");
    }

    size_t random_index = parallel_config_.role_rank % rdma_info_list->size();
    const auto* rdma_info = &((*rdma_info_list)[random_index]);
    // if (atensor.storage_offset != rdma_info->atensor->storage_offset) {
    //     SPDLOG_ERROR("storage_offset mismatch, tensor_key: {}, atensor.storage_offset: {},
    //     rdma_info->atensor->storage_offset: {}", tensor_key.key
    //, atensor.storage_offset
    //, rdma_info->atensor->storage_offset);
    //     throw std::runtime_error("illegal state: storage_offset mismatch");
    // }

    // TODO(root): (echo.zxj) this is a temporary solution to fix the issue that the remote tensor is not aligned with the
    // local tensor.
    //       we should fix this issue in the future.
    auto remote_byte_offset = GetStorageByteOffset(atensor.dtype, rdma_info->atensor->storage_offset);
    if (atensor.storage_offset != rdma_info->atensor->storage_offset && atensor.storage_offset != 0) {
        remote_byte_offset = GetStorageByteOffset(atensor.dtype, atensor.storage_offset);
    }
    auto byte_size = GetTensorTotalByteSize(atensor);
    if (remote_byte_offset + byte_size > rdma_info->size) {
        SPDLOG_ERROR(
            "Tensor data size exceeds the size of the remote storage, "
            "tensor_key: {}, byte_offset: {}, byte_size: {}, "
            "remote_storage_size: {}",
            tensor_key.key,
            remote_byte_offset,
            byte_size,
            rdma_info->size);
        throw std::runtime_error("illegal state: Tensor data size exceeds the "
                                 "size of the remote storage");
    }

    auto* remote_addr = static_cast<char*>(rdma_info->addr) + remote_byte_offset;
    auto* local_addr = static_cast<char*>(atensor.storage.data);
    ExtendInfo extend_info = GetExtendInfoFromRemoteAddr(remote_addr);

    auto read_prepare_end = std::chrono::high_resolution_clock::now();

    bool ret = data_rdma_transport_->Receive(
        local_addr, byte_size, rdma_info->node_info.hostname_or_ip, rdma_info->node_info.rdma_port, &extend_info);
    auto end_time = std::chrono::high_resolution_clock::now();

    UpdateThroughputStatistic(rdma_info->node_info.GetHostWithRdmaPort(), byte_size);
    auto stats_end = std::chrono::high_resolution_clock::now();

    if (perf_metrics_controller_->ShouldLogPerfMetric(seq_id)) {
        auto register_duration = std::chrono::duration_cast<std::chrono::microseconds>(register_end - start_time);
        auto wait_duration = std::chrono::duration_cast<std::chrono::microseconds>(wait_end - register_end);
        auto read_prepare_duration = std::chrono::duration_cast<std::chrono::microseconds>(read_prepare_end - wait_end);
        auto read_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - read_prepare_end);
        auto stats_duration = std::chrono::duration_cast<std::chrono::microseconds>(stats_end - end_time);
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(stats_end - start_time);

        SPDLOG_INFO(
            "Get tensor_key: {}, seq_id: {}, total cost {} us (register {} us, "
            "wait {} us, read_prepare {} us, read {} "
            "us, stats {} us), throughput {} MB/s from host {}, "
            "local_thread_pool_pending_tasks: {}",
            tensor_key.key,
            seq_id,
            total_duration.count(),
            register_duration.count(),
            wait_duration.count(),
            read_prepare_duration.count(),
            read_duration.count(),
            stats_duration.count(),
            BYTES_TO_MB(byte_size) / US_TO_SEC(total_duration.count()),
            rdma_info->node_info.GetHostWithRdmaPort(),
            thread_pool_->GetTaskCount());
    }

    return ret;
}

bool TensorTransferPull::MultiGet(int64_t seq_id, std::vector<std::pair<ShardedKey, ATensor>>& atensors) {
    if (atensors.empty()) {
        SPDLOG_WARN("No tensors to get for seq_id: {}", seq_id);
        return false;
    }

    // Check if service is running before proceeding with operations
    if (!IsRunning()) {
        SPDLOG_WARN("Service is not running, MultiGet operation cannot proceed");
        return false;
    }

    SetRead(current_data_operation_);
    if (!CheckAndUpdateCurrentSeqId(seq_id)) {
        return false;
    }

    // Use multithreading with std::ref to safely pass references
    std::vector<std::future<bool>> futures;
    futures.reserve(atensors.size());

    for (auto& pair : atensors) {
        // Copy the key and use std::ref for the tensor
        ShardedKey key_copy = pair.first;
        std::reference_wrapper<ATensor> tensor_ref = std::ref(pair.second);

        futures.emplace_back(thread_pool_->Submit([this, seq_id, key_copy, tensor_ref]() -> bool {
            // Get the actual reference from std::reference_wrapper
            return Get(seq_id, key_copy, tensor_ref.get());
        }));
    }

    // Wait for all operations to complete
    bool success = true;
    for (auto& future : futures) {
        success &= future.get();
    }

    return success;
}

bool TensorTransferPull::PreRegisterMemory(ATStorage& atensor_storage) {
    RegisterMemoryOrThrow(
        atensor_storage.data,
        atensor_storage.GetStorageDataSize(),
        atensor_storage.device.device_type == ATDeviceType::CUDA,
        atensor_storage.device.device_index);
    return true;
}

void TensorTransferPull::Complete() {
    // If reading finished, send weight consumed message to all peers
    if (IsRead(current_data_operation_)) {
        SPDLOG_INFO("Complete read");
        SendWeightConsumed(WeightConsumedMessage{current_seq_id_, local_node_info_});
    }

    // If writing finished, wait for all peers to consume the weights
    if (IsWrite(current_data_operation_)) {
        SPDLOG_INFO("Complete write");

        SendWeightReady(WeightReadyMessage{current_seq_id_, local_node_info_});

        std::chrono::milliseconds wait_time_ms(0);
        while (true) {
            // Check if all remote nodes have finished the data reading
            if (consumed_nodes_.size() == peer_hosts_.size()) {
                SPDLOG_INFO("Seq {} completed, all nodes have received weights", current_seq_id_);
                break;
            }

            if (wait_time_ms.count() > 0 && wait_time_ms.count() % ONE_MINUTE_MS == 0) {
                std::lock_guard<std::mutex> lock(ctrl_message_mutex_);

                std::set<std::string> missing_nodes;
                for (const auto& peer : peer_hosts_) {
                    if (consumed_nodes_.find(peer) == consumed_nodes_.end()) {
                        missing_nodes.insert(peer.hostname_or_ip + ":" + std::to_string(peer.rdma_port));
                    }
                }
                SPDLOG_INFO(
                    "Seq {} progress: {}/{} nodes completed. Missing "
                    "nodes: {}",
                    current_seq_id_,
                    peer_hosts_.size() - missing_nodes.size(),
                    peer_hosts_.size(),
                    (missing_nodes.empty() ? "none" : *missing_nodes.begin()));
            }


            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_time_ms += std::chrono::milliseconds(10);
        }
    }

    SPDLOG_INFO(
        "Complete() done, current_data_operation_: {}, current_seq_id_: {}",
        GetDataOperationString(current_data_operation_),
        current_seq_id_);

    LogRemoteTensorMeta();
    LogThroughputStatistic();

    // After all data operations are finished, reset current data operation and sequence id
    Clear(current_data_operation_);
    last_completed_seq_id_ = current_seq_id_;
    current_seq_id_ = -1;
    ready_nodes_.clear();
    consumed_nodes_.clear();
    // TODO(root): update is_publish_meta_ to false when source nodes' infos were changed.
    is_publish_meta_ = true;
}

void TensorTransferPull::SetPeerHosts(const std::vector<NodeInfo>& peer_hosts) {
    peer_hosts_ = peer_hosts;
}

bool TensorTransferPull::SendTensorRDMAMeta(const TensorRDMAMetaPublishMessage& meta) {
    auto message_data = Serialize(ToJson(meta));
    return SendCtrlMessageToMultiPeers(
        TENSOR_RDMA_META_REQUEST, meta.seq_id, message_data.c_str(), message_data.size(), peer_hosts_);
}

bool TensorTransferPull::SendWeightReady(const WeightReadyMessage& msg) {
    auto message_data = Serialize(ToJson(msg));
    return SendCtrlMessageToMultiPeers(
        WEIGHT_READY_REQUEST, msg.seq_id, message_data.c_str(), message_data.size(), peer_hosts_);
}

bool TensorTransferPull::SendWeightConsumed(const WeightConsumedMessage& msg) {
    auto message_data = Serialize(ToJson(msg));
    return SendCtrlMessageToMultiPeers(
        WEIGHT_CONSUMED_REQUEST, msg.seq_id, message_data.c_str(), message_data.size(), peer_hosts_);
}

ResponseStatus
TensorTransferPull::HandleTensorRDMAMeta(const std::string& /*request*/, const void* message, size_t message_size) {
    try {
        std::string message_str(static_cast<const char*>(message), message_size);
        if (is_debug_mode_) {
            SPDLOG_INFO("Received RDMA meta message: {}", message_str);
        }
        auto json = Deserialize(message_str);
        TensorRDMAMetaPublishMessage msg = FromJson(json, TensorRDMAMetaPublishMessage{});

        {
            std::lock_guard<std::mutex> lock(ctrl_message_mutex_);

            // First check if TransferTensorMeta exists for this seq_id
            auto cache_it = remote_tensor_cache_.find(msg.seq_id);
            TransferTensorMeta* transfer_meta = nullptr;

            if (cache_it != remote_tensor_cache_.end()) {
                // Exists, use existing one
                transfer_meta = &(cache_it->second);
            } else {
                // Does not exist, create new one
                auto result = remote_tensor_cache_.emplace(msg.seq_id, TransferTensorMeta{});
                transfer_meta = &(result.first->second);
                SPDLOG_INFO("Created new transfer data for seq_id={}", msg.seq_id);
            }

            // Process tensor information in the message
            for (const auto& pair : msg.tensor_rdma_metas) {
                const ShardedKey& key = pair.first;
                const TensorMemoryRDMAInfo& protocol_info = pair.second;

                // Use emplace to construct TensorRDMAInfo directly in map, avoiding copy
                // Since TensorRDMAInfo contains references, we need to create static NodeInfo
                static std::unordered_map<int64_t, NodeInfo> node_info_cache;
                NodeInfo& node_info = node_info_cache[msg.seq_id];
                node_info.hostname_or_ip = msg.node_info.hostname_or_ip;
                node_info.rdma_port = msg.node_info.rdma_port;
                node_info.ctrl_flow_port = msg.node_info.ctrl_flow_port;

                EmplaceTensorRDMAInfo(
                    *transfer_meta,
                    key,
                    protocol_info.addr,
                    protocol_info.size,
                    protocol_info.rkey,
                    node_info,
                    std::make_shared<ATensor>(protocol_info.atensor_meta));
            }
        }

        return ResponseStatus{true, "Success", ExtendInfo{}};
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to process RDMA meta message: {}", e.what());
        return ResponseStatus{false, e.what(), ExtendInfo{}};
    }
}

ResponseStatus
TensorTransferPull::HandleWeightReady(const std::string& /*request*/, const void* message, size_t message_size) {
    try {
        std::string message_str(static_cast<const char*>(message), message_size);
        if (is_debug_mode_) {
            SPDLOG_INFO("Received weight ready message: {}", message_str);
        }
        auto json = Deserialize(message_str);
        WeightReadyMessage msg = FromJson(json, WeightReadyMessage{});

        std::lock_guard<std::mutex> lock(ctrl_message_mutex_);
        ready_nodes_.insert(msg.node_info);

        if (ready_nodes_.size() == peer_hosts_.size() && enable_local_cache_prefetch_) {
            int64_t seq_id = msg.seq_id;
            if (ctx_ != nullptr && ctx_->tensor_table != nullptr) {
                thread_pool_->Submit([this, seq_id]() {
                    SPDLOG_INFO(
                        "All weights are ready of seq-{}, start to prefetch "
                        "cached tensors",
                        seq_id);
                    ctx_->tensor_table->PrefetchCachedTensors(seq_id);
                });
            }
        }

        return ResponseStatus{true, "Success", ExtendInfo{}};
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to process weight ready message: {}", e.what());
        return ResponseStatus{false, e.what(), ExtendInfo{}};
    }
}

ResponseStatus
TensorTransferPull::HandleWeightConsumed(const std::string& /*request*/, const void* message, size_t message_size) {
    try {
        std::string message_str(static_cast<const char*>(message), message_size);
        if (is_debug_mode_) {
            SPDLOG_INFO("Received weight consumed message: {}", message_str);
        }
        auto json = Deserialize(message_str);
        WeightConsumedMessage msg = FromJson(json, WeightConsumedMessage{});

        std::lock_guard<std::mutex> lock(ctrl_message_mutex_);
        if (msg.seq_id < current_seq_id_) {
            SPDLOG_ERROR(
                "Received outdated weight consumed message, seq_id: {}, "
                "current_seq_id: {}",
                msg.seq_id,
                current_seq_id_);
            return ResponseStatus{false, "Outdated sequence ID", ExtendInfo{}};
        }
        consumed_nodes_.insert(msg.node_info);
        return ResponseStatus{true, "Success", ExtendInfo{}};

    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to process weight consumed message: {}", e.what());
        return ResponseStatus{false, e.what(), ExtendInfo{}};
    }
}

bool TensorTransferPull::SendCtrlMessage(
    const std::string& request_name,
    const int64_t seq_id,
    const void* message,
    size_t message_size,
    const NodeInfo& node_info) {
    if (is_debug_mode_) {
        SPDLOG_INFO("Sending [{}] message for step {}", request_name, seq_id);
    }

    try {
        if (!ValidateSeqId(seq_id)) {
            return false;
        }
        return control_transport_->Send(
            request_name, message, message_size, node_info.hostname_or_ip, node_info.ctrl_flow_port, nullptr);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to send [{}] message for {}: {}", request_name, seq_id, e.what());
        return false;
    }
}

bool TensorTransferPull::SendCtrlMessageToMultiPeers(
    const std::string& request_name,
    const int64_t seq_id,
    const void* message,
    size_t message_size,
    const std::vector<NodeInfo>& node_infos) {
    try {
        if (node_infos.empty()) {
            SPDLOG_ERROR("No peers to send [{}] message for step {}", request_name, seq_id);
            return false;
        }

        // Send weight ready message to all peers synchronously
        std::vector<std::future<bool>> futures;
        futures.reserve(node_infos.size());
        for (const auto& peer : node_infos) {
            futures.push_back(thread_pool_->Submit([this, request_name, seq_id, message, message_size, peer]() {
                return SendCtrlMessage(request_name, seq_id, message, message_size, peer);
            }));
        }

        // Wait for all requests to complete
        bool all_success = true;
        for (auto& future : futures) {
            all_success &= future.get();
        }
        return all_success;
    } catch (const std::exception& e) {
        SPDLOG_ERROR(
            "Failed to send [{}] message for step {} to {} peers: {}",
            request_name,
            seq_id,
            node_infos.size(),
            e.what());
        return false;
    }
}

std::vector<std::pair<ShardedKey, ATensor>>
TensorTransferPull::GetAllTensorShards(int64_t seq_id, std::function<bool(const ShardedKey&)> filter) {
    std::vector<std::pair<ShardedKey, ATensor>> result;

    if (WaitForAllTensorReady(seq_id, tensor_ready_timeout_ms_)) {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex_));

        auto cache_it = remote_tensor_cache_.find(seq_id);
        if (cache_it != remote_tensor_cache_.end()) {
            const auto& transfer_meta = cache_it->second;

            for (const auto& entry : transfer_meta) {
                const ShardedKey& sharded_key = entry.first;
                const std::vector<TensorRDMAInfo>& rdma_info_list = entry.second;

                if (filter(sharded_key) && !rdma_info_list.empty()) {
                    result.emplace_back(sharded_key, *rdma_info_list.back().atensor);
                }
            }
        }
    }
    return result;
}

bool TensorTransferPull::RawGet(
    int64_t seq_id, const ATStorage& astorage, const NodeInfo& node_info, const void* remote_addr, size_t len) {
    if (astorage.GetStorageDataSize() < len) {
        SPDLOG_ERROR(
            "Memory size mismatch, local_addr: {}, local_storage_size: {}, "
            "remote_addr: {}, remote_storage_size: {}",
            astorage.data,
            astorage.GetStorageDataSize(),
            remote_addr,
            len);
        throw std::runtime_error("illegal state: memory size mismatch");
    }

    // Check if service is running before proceeding with operations
    if (!IsRunning()) {
        SPDLOG_WARN("Service is not running, Get operation cannot proceed");
        return false;
    }

    SetRead(current_data_operation_);
    if (!CheckAndUpdateCurrentSeqId(seq_id)) {
        return false;
    }

    if (!WaitForAllTensorReady(seq_id, tensor_ready_timeout_ms_)) {
        return false;
    }

    RegisterMemoryOrThrow(
        astorage.data,
        astorage.GetStorageDataSize(),
        astorage.device.device_type == ATDeviceType::CUDA,
        astorage.device.device_index);
    ExtendInfo extend_info = GetExtendInfoFromRemoteAddr(remote_addr);

    UpdateThroughputStatistic(node_info.GetHostWithRdmaPort(), len);
    return data_rdma_transport_->Receive(
        astorage.data, len, node_info.hostname_or_ip, node_info.rdma_port, &extend_info);
}

using TensorShardMap = std::unordered_map<ShardedKey, ATensor, ShardedKeyHash>;
using AddrMap = std::unordered_map<void*, TensorShardMap>;
using NodeMap = std::unordered_map<NodeInfo, AddrMap, NodeInfoHash>;

std::vector<CompactTensorInfo> CompactTensorInfos(
    const std::unordered_map<ShardedKey, ATensor, ShardedKeyHash>& atensors,
    NodeMap& node_map,
    TransferTensorMeta& target_transfer_meta) {
    std::vector<CompactTensorInfo> compact_tensor_infos{};
    for (const auto& entry : atensors) {
        const ShardedKey& sharded_key = entry.first;
        const ATensor& atensor = entry.second;

        auto rdma_info_it = target_transfer_meta.find(sharded_key);
        if (rdma_info_it == target_transfer_meta.end()) {
            // already contained in results
            continue;
        }

        const TensorRDMAInfo& rdma_info = rdma_info_it->second.back();
        auto node_it = node_map.find(rdma_info.node_info);
        if (node_it == node_map.end()) {
            SPDLOG_ERROR(
                "no meta info found for node: {}:{}",
                rdma_info.node_info.hostname_or_ip,
                rdma_info.node_info.rdma_port);
            throw std::runtime_error("illegal state: no meta info found for corresponding node");
        }

        auto addr_it = node_it->second.find(rdma_info.addr);
        if (addr_it == node_it->second.end()) {
            SPDLOG_ERROR(
                "no meta info found for addr: {}:{} {}",
                rdma_info.node_info.hostname_or_ip,
                rdma_info.node_info.rdma_port,
                rdma_info.addr);
            throw std::runtime_error("illegal state: no meta info found for corresponding addr");
        }

        auto shard_it = addr_it->second.find(sharded_key);
        if (shard_it == addr_it->second.end()) {
            SPDLOG_ERROR("no meta info found for shard: {}", sharded_key.key);
            throw std::runtime_error("illegal state: no meta info found for corresponding shard");
        }

        CompactTensorInfo compact_tensor_info{
            rdma_info.addr, rdma_info.size, rdma_info.rkey, rdma_info.node_info, {{sharded_key, atensor}}};

        for (const auto& shard : addr_it->second) {
            const ShardedKey& shard_key = shard.first;

            if (target_transfer_meta.erase(shard_key) > 0) {
                const ATensor& shard_atensor = shard.second;
                compact_tensor_info.atensors.emplace(shard_key, shard_atensor);
            }
        }
        compact_tensor_infos.push_back(compact_tensor_info);
    }
    return compact_tensor_infos;
}

std::vector<CompactTensorInfo> TensorTransferPull::GetCompactTensorInfos(
    int64_t seq_id, std::unordered_map<ShardedKey, ATensor, ShardedKeyHash> atensors) {
    NodeMap node_map{};
    TransferTensorMeta target_transfer_meta{};
    if (WaitForAllTensorReady(seq_id, tensor_ready_timeout_ms_)) {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex_));

        auto cache_it = remote_tensor_cache_.find(seq_id);
        if (cache_it != remote_tensor_cache_.end()) {
            const auto& transfer_meta = cache_it->second;

            for (const auto& entry : transfer_meta) {
                const ShardedKey& sharded_key = entry.first;
                auto atensor_it = atensors.find(sharded_key);
                if (atensor_it == atensors.end()) {
                    // no need to read this tensor
                    continue;
                }
                target_transfer_meta.emplace(sharded_key, entry.second);

                const std::vector<TensorRDMAInfo>& rdma_info_list = entry.second;
                for (const auto& rdma_info : rdma_info_list) {
                    auto node_it = node_map.find(rdma_info.node_info);
                    if (node_it == node_map.end()) {
                        node_it = node_map.emplace(rdma_info.node_info, AddrMap{}).first;
                    }

                    auto addr_it = node_it->second.find(rdma_info.addr);
                    if (addr_it == node_it->second.end()) {
                        node_it->second.emplace(rdma_info.addr, TensorShardMap{});
                        addr_it = node_it->second.find(rdma_info.addr);
                    }

                    addr_it->second.emplace(sharded_key, *rdma_info.atensor);
                }
            }
        }
    }

    if (target_transfer_meta.size() != atensors.size()) {
        SPDLOG_ERROR(
            "target_transfer_meta.size() != atensors.size(), "
            "target_transfer_meta.size(): {}, atensors.size(): {}",
            target_transfer_meta.size(),
            atensors.size());
        throw std::runtime_error("illegal state: target_transfer_meta.size() != atensors.size()");
    }
    return CompactTensorInfos(atensors, node_map, target_transfer_meta);
}
} // namespace astate
