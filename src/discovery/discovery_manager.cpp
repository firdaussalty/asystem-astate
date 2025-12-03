#include "discovery/discovery_manager.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "common/network_utils.h"
#include "common/option.h"
#include "core/atensor.h"
#include "discovery/file_config_center.h"
#include "discovery/http_config_center.h"
#include "discovery/service_discovery.h"
#include "discovery/tcpstore_discovery.h"

namespace astate {

// 静态工厂方法实现
std::unique_ptr<DiscoveryManager> DiscoveryManager::CreateFromOptions(
    const Options& options, const AParallelConfig& parallel_config, int rdma_port, int ctrl_port) {
    // 从Options中获取配置类型，默认为FILE
    ConfigCenterType config_type{};
    auto config_type_str = GetOptionValue<std::string>(options, DISCOVERY_CONFIG_CENTER_TYPE);
    if (config_type_str == "TCPStore") {
        config_type = ConfigCenterType::TCPStore;
    } else if (config_type_str == "HTTP") {
        config_type = ConfigCenterType::HTTP;
    } else if (config_type_str == "FILE") {
        config_type = ConfigCenterType::FILE;
    } else {
        SPDLOG_ERROR("Unknown config center type: {}", config_type_str);
        throw std::runtime_error("Unknown config center type: " + config_type_str);
    }

    // 从Options中获取发现类型，默认为TCPStore
    DiscoveryType discovery_type = DiscoveryType::TCPStore;
    auto discovery_type_str = options.find("DISCOVERY_TYPE");
    if (discovery_type_str != options.end()) {
        if (discovery_type_str->second == "TCPStore") {
            discovery_type = DiscoveryType::TCPStore;
        }
    }

    SPDLOG_INFO(
        "Creating DiscoveryManager with config_type={}, discovery_type={}, "
        "rdma_port={}, ctrl_port={}",
        static_cast<int>(config_type),
        static_cast<int>(discovery_type),
        rdma_port,
        ctrl_port);

    return std::make_unique<DiscoveryManager>(
        config_type, discovery_type, parallel_config, rdma_port, ctrl_port, options);
}

DiscoveryManager::DiscoveryManager(
    ConfigCenterType config_type,
    DiscoveryType discovery_type,
    AParallelConfig parallel_config,
    int rdma_port,
    int ctrl_port,
    Options options)
    : parallel_config_(parallel_config),
      options_(std::move(options)) {
    auto local_addr_ip = astate::GetLocalHostnameOrIP();
    current_node_.rank = parallel_config_.role_rank;
    current_node_.role = parallel_config_.role;
    current_node_.node_info.hostname_or_ip = local_addr_ip;
    current_node_.node_info.rdma_port = rdma_port;
    current_node_.node_info.ctrl_flow_port = ctrl_port;

    // 根据类型创建配置中心 client
    config_center_ = CreateConfigCenterByType(config_type);
    if (!config_center_) {
        SPDLOG_ERROR("Failed to create config center with type: {}", static_cast<int>(config_type));
        return;
    }

    // 根据类型创建服务发现
    service_discovery_ = CreateServiceDiscoveryByType(discovery_type);
    if (!service_discovery_) {
        SPDLOG_ERROR("Failed to create service discovery with type: {}", static_cast<int>(discovery_type));
        return;
    }

    should_stop_.store(false);
    if (enable_discovery_thread_) {
        discovery_thread_ = std::thread(&DiscoveryManager::DiscoveryThreadFunc, this);
    }

    SPDLOG_INFO(
        "DiscoveryManager initialized with config_type={}, discovery_type={}",
        static_cast<int>(config_type),
        static_cast<int>(discovery_type));
}

DiscoveryManager::~DiscoveryManager() {
    Stop();
}

bool DiscoveryManager::Start() {
    if (started_.load()) {
        SPDLOG_WARN("DiscoveryManager already started");
        return true;
    }

    started_.store(true);
    SPDLOG_INFO("DiscoveryManager started");
    return true;
}

void DiscoveryManager::Stop() {
    if (!started_.load()) {
        return;
    }

    should_stop_.store(true);
    if (enable_discovery_thread_ && discovery_thread_.joinable()) {
        discovery_thread_.join();
    }

    if (current_node_registered_.load()) {
        UnregisterCurrentNode();
    }

    started_.store(false);
    SPDLOG_INFO("DiscoveryManager stopped");
}

bool DiscoveryManager::RegisterCurrentNode() {
    if (service_discovery_->RegisterNode(current_node_)) {
        current_node_registered_.store(true);
        SPDLOG_INFO("Current node registered successfully");
        return true;
    }
    SPDLOG_ERROR("Failed to register current node");
    return false;
}

std::vector<NodeEntry> DiscoveryManager::DiscoverAllNodes() {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout = kDefaultTimeout;

    bool use_batch = GetOptionValue<bool>(options_, DISCOVERY_USE_BATCH_API);
    if (use_batch) {
        SPDLOG_INFO("Using batch discovery");
    }

    while (std::chrono::steady_clock::now() - start_time < timeout) {
        std::vector<NodeEntry> nodes;
        if (use_batch) {
            nodes = service_discovery_->DiscoverNodesBatch();
        } else {
            nodes = service_discovery_->DiscoverNodes();
        }
        if (CheckNodesCount(nodes)) {
            SPDLOG_INFO("All nodes discovered successfully");
            return nodes;
        }

        std::this_thread::sleep_for(kDiscoveryInterval);
    }

    SPDLOG_ERROR("Timeout waiting for all nodes to be discovered");
    throw std::runtime_error("Timeout waiting for all nodes to be discovered");
    return service_discovery_->DiscoverNodes();
}

std::vector<NodeEntry> DiscoveryManager::DiscoverNodes() {
    return service_discovery_->DiscoverNodes();
}

bool DiscoveryManager::UnregisterCurrentNode() {
    if (!current_node_registered_.load()) {
        return false;
    }

    if (service_discovery_->Unregister(current_node_)) {
        current_node_registered_.store(false);
        SPDLOG_INFO("Current node unregistered successfully");
        return true;
    }
    SPDLOG_ERROR("Failed to unregister current node");
    return false;
}

void DiscoveryManager::SetNodesReadyCallback(const std::function<void(const std::vector<NodeEntry>&)>& callback) {
    nodes_ready_callback_ = callback;
}

NodeEntry DiscoveryManager::GetCurrentNode() const {
    return current_node_;
}

bool DiscoveryManager::CheckAllNodesReady() const {
    return all_nodes_ready_.load();
}

bool DiscoveryManager::WaitForAllNodes(std::chrono::milliseconds timeout) {
    auto start_time = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start_time < timeout) {
        if (all_nodes_ready_.load()) {
            return true;
        }
        std::this_thread::sleep_for(kDiscoveryInterval);
    }

    return false;
}

std::vector<NodeEntry> DiscoveryManager::GetAllTrainingNodes() {
    std::unique_lock<std::mutex> lock(cache_mutex_);

    // 等待缓存就绪或超时
    if (!cache_cv_.wait_for(lock, kCacheWaitTimeout, [this]() { return nodes_cached_.load(); })) {
        throw std::runtime_error("Timeout waiting for training nodes cache to "
                                 "be ready (10 minutes)");
    }

    return cached_training_nodes_;
}

std::vector<NodeEntry> DiscoveryManager::GetAllInferenceNodes() {
    std::unique_lock<std::mutex> lock(cache_mutex_);

    // 等待缓存就绪或超时
    if (!cache_cv_.wait_for(lock, kCacheWaitTimeout, [this]() { return nodes_cached_.load(); })) {
        throw std::runtime_error("Timeout waiting for inference nodes cache to "
                                 "be ready (10 minutes)");
    }

    return cached_inference_nodes_;
}

std::unique_ptr<ConfigCenter> DiscoveryManager::CreateConfigCenterByType(ConfigCenterType type) {
    switch (type) {
        case ConfigCenterType::TCPStore:
            SPDLOG_INFO("Creating TCPStoreConfigCenter");
            return std::make_unique<TCPStoreConfigCenter>();
        case ConfigCenterType::HTTP:
            SPDLOG_ERROR("Creating HttpConfigCenter");
            return std::make_unique<HttpConfigCenter>();
        case ConfigCenterType::FILE:
            SPDLOG_INFO("Creating FileConfigCenter");
            return std::make_unique<FileConfigCenter>();
        default:
            SPDLOG_ERROR("Unknown config center type: {}", static_cast<int>(type));
            throw std::runtime_error("Unknown config center type: " + std::to_string(static_cast<int>(type)));
    }
}

std::unique_ptr<ServiceDiscovery> DiscoveryManager::CreateServiceDiscoveryByType(DiscoveryType type) {
    switch (type) {
        case DiscoveryType::TCPStore: {
            SPDLOG_INFO("Creating TCPStoreServiceDiscovery");
            // 根据当前节点的角色确定world_size
            return std::make_unique<TCPStoreServiceDiscovery>(
                current_node_.node_info.hostname_or_ip,
                current_node_.rank,
                current_node_.role,
                parallel_config_.role_size,
                true,
                config_center_.get(),
                options_);
        }
        default:
            SPDLOG_ERROR("Unknown discovery type: {}", static_cast<int>(type));
            throw std::runtime_error("Unknown discovery type: " + std::to_string(static_cast<int>(type)));
    }
}

void DiscoveryManager::DiscoveryThreadFunc() {
    while (!should_stop_.load()) {
        try {
            auto nodes = service_discovery_->DiscoverNodes();

            {
                std::lock_guard<std::mutex> lock(discovery_mutex_);
                discovered_nodes_ = nodes;
            }

            bool all_ready = CheckNodesCount(nodes);
            if (all_ready && !all_nodes_ready_.load()) {
                all_nodes_ready_.store(true);
                PrintNodesInfo(nodes);

                // 更新本地缓存
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    cached_training_nodes_.clear();
                    cached_inference_nodes_.clear();

                    for (const auto& node : nodes) {
                        if (node.role == ARole::TRAINING) {
                            cached_training_nodes_.push_back(node);
                        } else if (node.role == ARole::INFERENCE) {
                            cached_inference_nodes_.push_back(node);
                        }
                    }
                    nodes_cached_.store(true);
                }
                cache_cv_.notify_all();

                if (nodes_ready_callback_) {
                    nodes_ready_callback_(nodes);
                }
                SPDLOG_INFO("All nodes are ready, stopping discovery thread");
                should_stop_.store(true);
            }

        } catch (const std::exception& e) {
            SPDLOG_ERROR("Error in discovery thread: {}", e.what());
        }

        std::this_thread::sleep_for(kDiscoveryInterval);
    }
}

bool DiscoveryManager::CheckNodesCount(const std::vector<NodeEntry>& nodes) const {
    int train_count = 0;
    int inference_count = 0;

    for (const auto& node : nodes) {
        if (node.role == ARole::TRAINING) {
            train_count++;
        } else {
            inference_count++;
        }
    }

    int train_world_size = service_discovery_->GetTrainWorldSize();
    int inference_world_size = service_discovery_->GetInferenceWorldSize();

    bool train_ready = train_count >= train_world_size;
    bool inference_ready = inference_count >= inference_world_size;

    SPDLOG_INFO(
        "Node count check: train={}/{}, inference={}/{}, all_ready={}",
        train_count,
        train_world_size,
        inference_count,
        inference_world_size,
        (train_ready && inference_ready));

    return train_ready && inference_ready;
}

void DiscoveryManager::PrintNodesInfo(const std::vector<NodeEntry>& nodes) {
    SPDLOG_INFO("=== All nodes discovered ===");
    for (const auto& node : nodes) {
        SPDLOG_INFO(
            "Node: rank={}, role={}, host={}, rdma_port={}, ctrl_port={}",
            node.rank,
            RoleToString(node.role),
            node.node_info.hostname_or_ip,
            node.node_info.rdma_port,
            node.node_info.ctrl_flow_port);
    }
    SPDLOG_INFO("============================");
}

} // namespace astate
