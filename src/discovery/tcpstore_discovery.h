#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "core/atensor.h"
#include "discovery/service_discovery.h"

namespace astate {

class TCPStoreServiceDiscovery : public ServiceDiscovery {
 public:
    TCPStoreServiceDiscovery(
        const std::string& local_addr,
        int rank,
        ARole role,
        int role_world_size,
        bool control_master,
        ConfigCenter* config_center,
        const astate::Options& options = {});
    ~TCPStoreServiceDiscovery() override;

    bool RegisterMyWorldSize();

    void StartFetchWorldSize();

    bool WaitForWorldSizeReady(std::chrono::milliseconds timeout = std::chrono::minutes(15));

    bool RegisterNode(const NodeEntry& entry) override;
    std::vector<NodeEntry> DiscoverNodes() override;
    bool Refresh(const NodeEntry& entry) override;
    bool Unregister(const NodeEntry& entry) override;

    int GetTrainWorldSize() override;
    int GetInferenceWorldSize() override;

    std::vector<NodeEntry> DiscoverNodesBatch() override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    static constexpr int kPortStart = 49500;
    static constexpr int kPortEnd = 49600;
};

} // namespace astate
