#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "protocol/messages.h"

namespace astate {

struct NodeEntry {
    int rank{};
    NodeInfo node_info{};
    ARole role{};
};

const std::string kNamespace = "astate";
const std::string kNodeKeyPrefix = "node_";
constexpr int kMaxRetry = 3;
constexpr int kGetConfMaxRetry = 10;
constexpr std::chrono::milliseconds kDefaultTimeout = std::chrono::seconds(10);

class ServiceDiscovery {
 public:
    virtual ~ServiceDiscovery() = default;

    virtual bool RegisterNode(const NodeEntry& entry) = 0;

    virtual std::vector<NodeEntry> DiscoverNodes() = 0;

    virtual bool Refresh(const NodeEntry& entry) = 0;

    virtual bool Unregister(const NodeEntry& entry) = 0;

    virtual std::vector<NodeEntry> DiscoverNodesBatch() { return DiscoverNodes(); }

    virtual int GetTrainWorldSize() = 0;

    virtual int GetInferenceWorldSize() = 0;

    static std::string NodeToString(const NodeEntry& entry) {
        char buf[256];
        snprintf(
            buf,
            sizeof(buf),
            "%c:%d:%s:%d:%d",
            entry.role == ARole::TRAINING ? 'T' : 'I',
            entry.rank,
            entry.node_info.hostname_or_ip.c_str(),
            entry.node_info.rdma_port,
            entry.node_info.ctrl_flow_port);
        return buf;
    }

    static NodeEntry StringToNode(const std::string& str) {
        NodeEntry entry;
        char role_char = 0;
        char ip[128];
        int rank = 0;
        int rdma = 0;
        int ctrl = 0;
        int n = sscanf(str.c_str(), "%c:%d:%127[^:]:%d:%d", &role_char, &rank, ip, &rdma, &ctrl);
        if (n != 5) {
            throw std::runtime_error("node string parse error: " + str);
        }
        entry.role = (role_char == 'T') ? ARole::TRAINING : ARole::INFERENCE;
        entry.rank = rank;
        entry.node_info.hostname_or_ip = ip;
        entry.node_info.rdma_port = rdma;
        entry.node_info.ctrl_flow_port = ctrl;
        return entry;
    }
};

class ConfigCenter {
 public:
    virtual ~ConfigCenter() = default;

    virtual bool SetConfig(const std::string& key, const std::string& value) = 0;

    virtual bool GetConfig(const std::string& key, std::string& value) = 0;

    virtual bool RemoveConfig(const std::string& key) = 0;
};

class TCPStoreConfigCenter : public ConfigCenter {
 public:
    TCPStoreConfigCenter();

    TCPStoreConfigCenter(const std::string& master_addr, int master_port) {
        c10d::TCPStoreOptions opts;
        opts.port = master_port;
        opts.isServer = false;
        opts.waitWorkers = false;
        opts.timeout = kDefaultTimeout;
        store_ = std::make_unique<c10d::TCPStore>(master_addr, opts);
    }

    bool SetConfig(const std::string& key, const std::string& value) override {
        std::vector<uint8_t> data(value.begin(), value.end());
        store_->set(key, data);
        SPDLOG_INFO("set_config: {} = {}", key, value);
        return true;
    }

    bool GetConfig(const std::string& key, std::string& value) override {
        try {
            std::vector<uint8_t> data = store_->get(key);
            value = std::string(data.begin(), data.end());
            SPDLOG_INFO("get_config: {} = '{}'", key, value);
            return true;
        } catch (...) {
            return false;
        }
    }
    bool RemoveConfig(const std::string& key) override { return store_->deleteKey(key); }

 private:
    std::unique_ptr<c10d::TCPStore> store_;

    void LoadConfigFromEnv();
};
} // namespace astate
