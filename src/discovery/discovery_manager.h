#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "common/option.h"
#include "core/atensor.h"
#include "discovery/service_discovery.h"

namespace astate {

enum class ConfigCenterType : uint8_t { TCPStore, HTTP, FILE };
enum class DiscoveryType : uint8_t { TCPStore };

class DiscoveryManager {
 public:
    explicit DiscoveryManager(
        ConfigCenterType config_type,
        DiscoveryType discovery_type,
        AParallelConfig parallel_config,
        int rdma_port,
        int ctrl_port,
        Options options = {});

    ~DiscoveryManager();

    DiscoveryManager(const DiscoveryManager&) = delete;
    DiscoveryManager& operator=(const DiscoveryManager&) = delete;

    static std::unique_ptr<DiscoveryManager>
    CreateFromOptions(const Options& options, const AParallelConfig& parallel_config, int rdma_port, int ctrl_port);

    bool Start();

    void Stop();

    bool RegisterCurrentNode();

    std::vector<NodeEntry> DiscoverAllNodes();

    std::vector<NodeEntry> DiscoverNodes();

    bool UnregisterCurrentNode();

    void SetNodesReadyCallback(const std::function<void(const std::vector<NodeEntry>&)>& callback);

    NodeEntry GetCurrentNode() const;

    ConfigCenter* GetConfigCenter() const { return config_center_.get(); }

    int GetTrainWorldSize() const { return service_discovery_ ? service_discovery_->GetTrainWorldSize() : 0; }
    int GetInferenceWorldSize() const { return service_discovery_ ? service_discovery_->GetInferenceWorldSize() : 0; }

    std::vector<NodeEntry> GetAllTrainingNodes();
    std::vector<NodeEntry> GetAllInferenceNodes();

    bool CheckAllNodesReady() const;

    bool WaitForAllNodes(std::chrono::milliseconds timeout);

 private:
    static std::unique_ptr<ConfigCenter> CreateConfigCenterByType(ConfigCenterType type);

    std::unique_ptr<ServiceDiscovery> CreateServiceDiscoveryByType(DiscoveryType type);

    void DiscoveryThreadFunc();

    bool CheckNodesCount(const std::vector<NodeEntry>& nodes) const;

    static void PrintNodesInfo(const std::vector<NodeEntry>& nodes);

    AParallelConfig parallel_config_;

    Options options_;

    std::unique_ptr<ConfigCenter> config_center_;
    std::unique_ptr<ServiceDiscovery> service_discovery_;

    NodeEntry current_node_;
    std::atomic<bool> current_node_registered_{false};

    std::atomic<bool> started_{false};
    std::atomic<bool> should_stop_{false};

    bool enable_discovery_thread_{false};
    std::thread discovery_thread_;
    std::mutex discovery_mutex_;
    std::vector<NodeEntry> discovered_nodes_;
    std::atomic<bool> all_nodes_ready_{false};

    std::vector<NodeEntry> cached_training_nodes_;
    std::vector<NodeEntry> cached_inference_nodes_;
    std::atomic<bool> nodes_cached_{false};
    std::mutex cache_mutex_;
    std::condition_variable cache_cv_;

    std::function<void(const std::vector<NodeEntry>&)> nodes_ready_callback_;

    static constexpr bool kDefaultControlMaster = true;
    static constexpr auto kDiscoveryInterval = std::chrono::seconds(1);
    static constexpr auto kDefaultTimeout = std::chrono::seconds(1800);
    static constexpr auto kCacheWaitTimeout = std::chrono::minutes(10);
};

} // namespace astate
