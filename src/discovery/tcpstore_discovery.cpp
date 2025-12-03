#include "tcpstore_discovery.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <json/json.h>
#include <spdlog/spdlog.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "common/retry/counting_retry.h"
#include "common/retry/retry_utils.h"
#include "common/retry/timeout_retry.h"
#include "core/atensor.h"
#include "discovery/service_discovery.h"

namespace astate {

static std::string GenerateKeyPrefix() {
    std::string prefix = "astate";
    return prefix;
}

static std::string NodeKey(int rank, ARole role) {
    std::ostringstream oss;
    std::string prefix = GenerateKeyPrefix();
    if (role == ARole::TRAINING) {
        oss << prefix << ".node_train_" << rank;
    } else {
        oss << prefix << ".node_inference_" << rank;
    }
    return oss.str();
}

static std::string WorldSizeKey(ARole role) {
    std::ostringstream oss;
    std::string prefix = GenerateKeyPrefix();
    if (role == ARole::TRAINING) {
        oss << prefix << ".train_world_size";
    } else {
        oss << prefix << ".inference_world_size";
    }
    return oss.str();
}

static std::string TcpStoreAddressKey() {
    std::ostringstream oss;
    std::string prefix = GenerateKeyPrefix();
    oss << prefix << ".tcpstore.address";
    return oss.str();
}

struct TCPStoreServiceDiscovery::Impl {
    std::unique_ptr<c10d::TCPStore> store;
    std::unique_ptr<c10d::TCPStore> master_store;
    std::atomic<bool> master_running{false};
    std::string local_addr;
    std::string master_addr;
    int master_port{};
    int rank;
    int role_world_size;
    int train_world_size = 0;
    int inference_world_size = 0;
    bool control_master = true;
    ARole role;
    ConfigCenter* config_center = nullptr;
    astate::Options options;
    std::unordered_map<std::string, NodeEntry> discovered_nodes_cache;

    Impl(
        const std::string& addr,
        int rank,
        ARole role,
        int role_world_size,
        bool control_master,
        ConfigCenter* config_center = nullptr,
        astate::Options options = {})
        : local_addr(addr),
          master_addr(addr),
          rank(rank),
          role_world_size(role_world_size),
          control_master(control_master),
          role(role),
          config_center(config_center),
          options(std::move(options)) {
        if (role == ARole::TRAINING) {
            train_world_size = role_world_size;
        } else {
            inference_world_size = role_world_size;
        }

        if (control_master && role == ARole::TRAINING && rank == 0) {
            auto init = StartMasterService();
            if (!init) {
                SPDLOG_ERROR("Failed to start TCPStore master service");
                throw std::runtime_error("Failed to start master service");
            }
        } else {
            FetchMasterConfig();
        }
        RegisterAndFetchWorldSizes();
    }

    ~Impl() {
        if (control_master && IsMasterService()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            SPDLOG_INFO("Stopping master service");
            config_center->RemoveConfig(TcpStoreAddressKey());
            StopMasterService();
        }
    }

    void FetchMasterConfig() {
        auto retry_policy = std::make_unique<TimeoutRetry>(600000, 10000);

        SPDLOG_INFO("Starting to fetch master config with 10 minutes timeout");

        try {
            RetryUtils::Retry(
                "fetch master config",
                [this, &retry_policy]() {
                    SPDLOG_INFO("Fetching master config attempt #{}", retry_policy->GetAttemptCount());

                    std::string addr_port;
                    if (!config_center->GetConfig(TcpStoreAddressKey(), addr_port)) {
                        SPDLOG_WARN(
                            "Failed to fetch master config on attempt #{}, "
                            "config not found",
                            retry_policy->GetAttemptCount());
                        throw std::runtime_error("Config not found");
                    }

                    auto pos = addr_port.find(':');
                    if (pos == std::string::npos || pos + 1 >= addr_port.length()) {
                        SPDLOG_WARN(
                            "Invalid master config format: {} on attempt #{}",
                            addr_port,
                            retry_policy->GetAttemptCount());
                        throw std::runtime_error("Invalid config format: " + addr_port);
                    }

                    std::string port_str = addr_port.substr(pos + 1);
                    if (port_str.empty() || port_str.find_first_not_of("0123456789") != std::string::npos) {
                        SPDLOG_WARN(
                            "Invalid port in master config: {} on attempt #{}",
                            port_str,
                            retry_policy->GetAttemptCount());
                        throw std::runtime_error("Invalid port: " + port_str);
                    }

                    try {
                        master_port = std::stoi(port_str);
                        master_addr = addr_port.substr(0, pos);
                        SPDLOG_INFO(
                            "Successfully fetched master config:  (IP: {}, "
                            "Port: {}) on attempt #{}",
                            master_addr,
                            master_port,
                            retry_policy->GetAttemptCount());
                    } catch (const std::exception& e) {
                        SPDLOG_WARN(
                            "Failed to parse port: {} on attempt #{}, error: "
                            "{}",
                            port_str,
                            retry_policy->GetAttemptCount(),
                            e.what());
                        throw std::runtime_error("Failed to parse port: " + port_str);
                    }
                },
                *retry_policy);

        } catch (const std::exception& e) {
            SPDLOG_ERROR(
                "Failed to fetch master config after {} attempts over 10 "
                "minutes: {}",
                retry_policy->GetAttemptCount(),
                e.what());
            throw std::runtime_error(
                "Failed to fetch master config after 10 minutes timeout: " + std::string(e.what()));
        }
    }

    void MakeSureConnect() {
        if (!store) {
            for (int retry = 0; retry < kMaxRetry; ++retry) {
                try {
                    c10d::TCPStoreOptions opts;
                    opts.port = master_port;
                    opts.isServer = false;
                    opts.waitWorkers = false;
                    opts.timeout = kDefaultTimeout;
                    store = std::make_unique<c10d::TCPStore>(master_addr, opts);
                    SPDLOG_INFO("Successfully connected to master at {}:{}", master_addr, master_port);
                    return;
                } catch (const std::exception& e) {
                    if (retry == kMaxRetry - 1) {
                        throw;
                    }
                    SPDLOG_WARN(
                        "Failed to connect to master at {}:{}, retry {}/{}, "
                        "error: {}",
                        master_addr,
                        master_port,
                        (retry + 1),
                        kMaxRetry,
                        e.what());
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                }
            }
        }
    }

    bool StartMasterService() {
        if (master_running.load()) {
            SPDLOG_WARN("Master service already running");
            return true;
        }

        master_addr = local_addr;
        SPDLOG_INFO("Starting TCPStore master service...");

        for (int port = kPortStart; port <= kPortEnd; ++port) {
            try {
                c10d::TCPStoreOptions opts;
                opts.port = port;
                opts.isServer = true;
                opts.waitWorkers = false;

                opts.numWorkers = role_world_size * 2;

                master_store = std::make_unique<c10d::TCPStore>(master_addr, opts);
                master_port = port;
                master_running.store(true);

                SPDLOG_INFO("Master service started on {}:{}", master_addr, port);

                std::string addr_port = master_addr + ":" + std::to_string(port);
                config_center->SetConfig(TcpStoreAddressKey(), addr_port);
                SPDLOG_INFO("Master address saved to config center: {}", addr_port);
                return true;
            } catch (const std::exception& e) {
                SPDLOG_WARN("Port {} bind failed: {}", port, e.what());
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }

        SPDLOG_ERROR("No available port in range [{}, {}]", kPortStart, kPortEnd);
        master_running.store(false);
        return false;
    }

    void StopMasterService() {
        if (!master_running.load()) {
            return;
        }

        master_running.store(false);
        master_store.reset();
        SPDLOG_INFO("Master service stopped");
    }

    bool IsMasterService() const { return master_running.load(); }

    void RegisterAndFetchWorldSizes() {
        int retry_count = GetOptionValue<int>(options, DISCOVERY_WORLD_SIZE_RETRY_COUNT);
        auto retry_policy = std::make_unique<CountingRetry>(retry_count);

        SPDLOG_INFO("Starting to register and fetch world sizes with max {} retries", retry_count);

        try {
            RetryUtils::Retry(
                "register and fetch world sizes",
                [this, &retry_policy]() {
                    SPDLOG_INFO("Register and fetch world sizes attempt #{}", retry_policy->GetAttemptCount());

                    MakeSureConnect();

                    std::string my_key;
                    std::string other_key;
                    int my_world_size = 0;

                    if (role == ARole::TRAINING) {
                        my_key = WorldSizeKey(ARole::TRAINING);
                        other_key = WorldSizeKey(ARole::INFERENCE);
                        my_world_size = train_world_size;
                    } else {
                        my_key = WorldSizeKey(ARole::INFERENCE);
                        other_key = WorldSizeKey(ARole::TRAINING);
                        my_world_size = inference_world_size;
                    }

                    // 先检查是否已经注册
                    bool should_register = true;
                    try {
                        std::vector<uint8_t> existing_data = store->get(my_key);
                        std::string existing_value(existing_data.begin(), existing_data.end());
                        int existing_world_size = std::stoi(existing_value);

                        if (existing_world_size == my_world_size) {
                            SPDLOG_INFO(
                                "{} world size already registered correctly: "
                                "{}={}",
                                RoleToString(role),
                                my_key,
                                my_world_size);
                            should_register = false;
                        } else {
                            SPDLOG_ERROR(
                                "{} world size mismatch for {}: existing={}, "
                                "current={}",
                                RoleToString(role),
                                my_key,
                                existing_world_size,
                                my_world_size);
                            throw std::runtime_error(
                                RoleToString(role) + " world size mismatch for " + my_key + ": existing="
                                + std::to_string(existing_world_size) + ", current=" + std::to_string(my_world_size));
                        }

                    } catch (const std::exception& e) {
                        SPDLOG_INFO("{} world size not found for {}, will register", RoleToString(role), my_key);
                    }

                    // 注册自己的world_size
                    if (should_register) {
                        std::string value_str = std::to_string(my_world_size);
                        std::vector<uint8_t> value_data(value_str.begin(), value_str.end());
                        store->set(my_key, value_data);
                        SPDLOG_INFO("Registered world size: {}={}", my_key, my_world_size);
                    }

                    // 获取另一个Role的world_size
                    try {
                        std::vector<uint8_t> other_data = store->get(other_key);
                        std::string other_value(other_data.begin(), other_data.end());
                        int other_world_size = std::stoi(other_value);

                        if (role == ARole::TRAINING) {
                            inference_world_size = other_world_size;
                        } else {
                            train_world_size = other_world_size;
                        }
                        SPDLOG_INFO(
                            "World size fetched: train_world_size={}, "
                            "inference_world_size={}",
                            train_world_size,
                            inference_world_size);

                        // 成功获取到另一个role的world size，不需要重试
                        return;

                    } catch (const std::exception& e) {
                        SPDLOG_WARN(
                            "'{}' not available yet, on attempt #{}, error: {}",
                            other_key,
                            retry_policy->GetAttemptCount(),
                            e.what());

                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        throw std::runtime_error("Other world size not available, will retry");
                    }
                },
                *retry_policy);

        } catch (const std::exception& e) {
            SPDLOG_ERROR(
                "Failed to fetch other world size after {} attempts: {}", retry_policy->GetAttemptCount(), e.what());
            throw std::runtime_error(
                "Failed to fetch other world size after " + std::to_string(retry_policy->GetAttemptCount())
                + " attempts: " + std::string(e.what()));
        }
    }
};

TCPStoreServiceDiscovery::TCPStoreServiceDiscovery(
    const std::string& local_addr,
    int rank,
    ARole role,
    int role_world_size,
    bool control_master,
    ConfigCenter* config_center,
    const astate::Options& options)
    : impl_(new Impl(local_addr, rank, role, role_world_size, control_master, config_center, options)) {
}

TCPStoreServiceDiscovery::~TCPStoreServiceDiscovery() = default;

bool TCPStoreServiceDiscovery::RegisterNode(const NodeEntry& entry) {
    try {
        impl_->MakeSureConnect();
        std::string key = NodeKey(entry.rank, entry.role);
        std::string value = ServiceDiscovery::NodeToString(entry);
        std::vector<uint8_t> data(value.begin(), value.end());
        impl_->store->set(key, data);
        SPDLOG_INFO("Registered node: {} with value: {}", key, value);
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to register node: {}", e.what());
        return false;
    }
}

// original way: discover nodes one by one
std::vector<NodeEntry> TCPStoreServiceDiscovery::DiscoverNodes() {
    std::vector<NodeEntry> result;
    try {
        impl_->MakeSureConnect();

        // 扫描训练节点
        int train_found = 0;
        if (impl_->train_world_size > 0) {
            for (int i = 0; i < impl_->train_world_size; ++i) {
                std::string key = NodeKey(i, ARole::TRAINING);
                auto it = impl_->discovered_nodes_cache.find(key);
                if (it != impl_->discovered_nodes_cache.end()) {
                    result.push_back(it->second);
                    train_found++;
                    SPDLOG_INFO("Used cached TRAIN node: {}, value: {}", key, it->second.node_info.hostname_or_ip);
                    continue;
                }
                try {
                    std::vector<uint8_t> data = impl_->store->get(key);
                    std::string value(data.begin(), data.end());

                    NodeEntry entry = ServiceDiscovery::StringToNode(value);
                    result.push_back(entry);
                    impl_->discovered_nodes_cache[key] = entry;
                    train_found++;
                    SPDLOG_INFO("Discovered and cached TRAIN node: {}, value: {}", key, value);
                } catch (const std::exception& e) {
                    SPDLOG_INFO("TRAIN node {} not found yet: {}", key, e.what());
                }
            }
            if (train_found == impl_->train_world_size) {
                SPDLOG_INFO("All train nodes found, total {}", train_found);
            } else {
                SPDLOG_ERROR(
                    "Only {} train nodes found, expected {}, missing {}",
                    train_found,
                    impl_->train_world_size,
                    (impl_->train_world_size - train_found));
            }

        } else {
            SPDLOG_INFO("Train world size not available yet, skipping train "
                        "nodes discovery");
        }

        // 扫描推理节点
        int inference_found = 0;
        if (impl_->inference_world_size > 0) {
            for (int i = 0; i < impl_->inference_world_size; ++i) {
                std::string key = NodeKey(i, ARole::INFERENCE);
                auto it = impl_->discovered_nodes_cache.find(key);
                if (it != impl_->discovered_nodes_cache.end()) {
                    result.push_back(it->second);
                    inference_found++;
                    SPDLOG_INFO("Used cached INFERENCE node: {}, value: {}", key, it->second.node_info.hostname_or_ip);
                    continue;
                }
                try {
                    std::vector<uint8_t> data = impl_->store->get(key);
                    std::string value(data.begin(), data.end());
                    NodeEntry entry = ServiceDiscovery::StringToNode(value);
                    result.push_back(entry);
                    impl_->discovered_nodes_cache[key] = entry;
                    inference_found++;
                    SPDLOG_INFO("Discovered and cached INFERENCE node: {}, value: {}", key, value);
                } catch (const std::exception& e) {
                    SPDLOG_INFO("INFERENCE node {} not found yet: {}", key, e.what());
                }
            }
            if (inference_found == impl_->inference_world_size) {
                SPDLOG_INFO("All inference nodes found, total {}", inference_found);
            } else {
                SPDLOG_ERROR(
                    "Only {} inference nodes found, expected {}, missing {}",
                    inference_found,
                    impl_->inference_world_size,
                    (impl_->inference_world_size - inference_found));
            }
        } else {
            SPDLOG_INFO("Inference world size not available yet, skipping "
                        "inference nodes discovery");
        }

        SPDLOG_INFO(
            "Total discovered nodes: {}, train_world_size={}, "
            "inference_world_size={}",
            result.size(),
            impl_->train_world_size,
            impl_->inference_world_size);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to discover nodes: {}", e.what());
    }
    return result;
}

std::vector<NodeEntry> TCPStoreServiceDiscovery::DiscoverNodesBatch() {
    std::vector<NodeEntry> result;
    std::vector<std::string> expected_train_keys;
    std::vector<std::string> expected_inference_keys;
    std::vector<std::string> missing_keys;
    int train_found = 0;
    int inference_found = 0;

    // Collect train keys
    if (impl_->train_world_size > 0) {
        for (int i = 0; i < impl_->train_world_size; ++i) {
            std::string key = NodeKey(i, ARole::TRAINING);
            expected_train_keys.push_back(key);
            if (impl_->discovered_nodes_cache.count(key) != 0U) {
                train_found++;
            } else {
                missing_keys.push_back(key);
            }
        }
    }

    // Collect inference keys
    if (impl_->inference_world_size > 0) {
        for (int i = 0; i < impl_->inference_world_size; ++i) {
            std::string key = NodeKey(i, ARole::INFERENCE);
            expected_inference_keys.push_back(key);
            if (impl_->discovered_nodes_cache.count(key) != 0U) {
                inference_found++;
            } else {
                missing_keys.push_back(key);
            }
        }
    }

    // Wait for missing keys
    if (!missing_keys.empty()) {
        SPDLOG_INFO("Waiting for missing nodes: {}", missing_keys.size());
        auto start_time = std::chrono::steady_clock::now();
        try {
            impl_->store->wait(missing_keys, std::chrono::minutes(5));
            SPDLOG_INFO(
                "Batch wait for missing nodes completed in {} seconds",
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time)
                    .count());
        } catch (const std::exception& e) {
            SPDLOG_WARN("Batch wait for missing nodes timed out: {}", e.what());
            throw std::runtime_error(
                "Batch wait for missing nodes timed out: " + std::string(e.what()) + ", retry for next round");
        }

        // sleep for a random time to avoid flooding the TCPStore
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, impl_->train_world_size + impl_->inference_world_size);
        int sleep_time = dis(gen);
        SPDLOG_INFO("Sleeping for {} seconds to avoid flooding the TCPStore", sleep_time);
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));

        try {
            std::vector<std::vector<uint8_t>> data_vec = impl_->store->multiGet(missing_keys);
            for (size_t i = 0; i < missing_keys.size(); ++i) {
                const std::string& key = missing_keys[i];
                std::vector<uint8_t> data = data_vec[i];
                std::string value(data.begin(), data.end());
                NodeEntry entry = ServiceDiscovery::StringToNode(value);

                // Parse role and rank from key like "astate.node_train_0" or "astate.node_inference_0"
                // Find the last underscore to get the rank
                size_t last_underscore = key.rfind('_');
                if (last_underscore == std::string::npos) {
                    SPDLOG_ERROR("Invalid key format, no underscore found: {}", key);
                    continue;
                }

                // Extract rank from after the last underscore
                std::string rank_str = key.substr(last_underscore + 1);
                int expected_rank = 0;
                try {
                    expected_rank = std::stoi(rank_str);
                } catch (const std::exception& stoi_ex) {
                    SPDLOG_ERROR(
                        "Failed to parse rank from key: {}, rank_str: '{}', "
                        "error: {}",
                        key,
                        rank_str,
                        stoi_ex.what());
                    continue;
                }

                ARole expected_role{};
                if (key.find("train") != std::string::npos) {
                    expected_role = ARole::TRAINING;
                } else if (key.find("inference") != std::string::npos) {
                    expected_role = ARole::INFERENCE;
                } else {
                    SPDLOG_ERROR("Unknown role in key: {}", key);
                    continue;
                }

                if (entry.role != expected_role || entry.rank != expected_rank) {
                    SPDLOG_ERROR(
                        "Invalid entry for node: {} expected role: {} rank: {}",
                        value,
                        static_cast<int>(expected_role),
                        expected_rank);
                    continue;
                }

                impl_->discovered_nodes_cache[key] = entry;
                result.push_back(entry);
                if (expected_role == ARole::TRAINING) {
                    train_found++;
                    SPDLOG_INFO("Discovered and cached TRAIN node: {}, value: {}", key, value);
                } else {
                    inference_found++;
                    SPDLOG_INFO("Discovered and cached INFERENCE node: {}, value: {}", key, value);
                }
            }
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Multi-get failed: {}", e.what());
        }
    }

    if (impl_->train_world_size > 0) {
        if (train_found == impl_->train_world_size) {
            SPDLOG_INFO("All train nodes found, total {}", train_found);
        } else {
            SPDLOG_ERROR(
                "Only {} train nodes found, expected {}, missing {}",
                train_found,
                impl_->train_world_size,
                (impl_->train_world_size - train_found));
        }
    }

    if (impl_->inference_world_size > 0) {
        if (inference_found == impl_->inference_world_size) {
            SPDLOG_INFO("All inference nodes found, total {}", inference_found);
        } else {
            SPDLOG_ERROR(
                "Only {} inference nodes found, expected {}, missing {}",
                inference_found,
                impl_->inference_world_size,
                (impl_->inference_world_size - inference_found));
        }
    }

    return result;
}

bool TCPStoreServiceDiscovery::Refresh(const NodeEntry& entry) {
    return RegisterNode(entry);
}

bool TCPStoreServiceDiscovery::Unregister(const NodeEntry& entry) {
    try {
        impl_->MakeSureConnect();
        std::string key = NodeKey(entry.rank, entry.role);
        impl_->store->deleteKey(key);
        SPDLOG_INFO("Unregistered node: {}", key);
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to unregister node: {}", e.what());
        return false;
    }
}

int TCPStoreServiceDiscovery::GetTrainWorldSize() {
    return impl_->train_world_size;
}

int TCPStoreServiceDiscovery::GetInferenceWorldSize() {
    return impl_->inference_world_size;
}

TCPStoreConfigCenter::TCPStoreConfigCenter() {
    LoadConfigFromEnv();
}

void TCPStoreConfigCenter::LoadConfigFromEnv() {
    const char* master_addr_env = std::getenv("ASTATE_TCPSTORE_MASTER_ADDR");
    const char* master_port_env = std::getenv("ASTATE_TCPSTORE_MASTER_PORT");

    std::string master_addr;
    int master_port = 0;

    if (master_addr_env != nullptr) {
        master_addr = std::string(master_addr_env);
        SPDLOG_INFO("ASTATE_TCPSTORE_MASTER_ADDR: {}", master_addr);
    } else {
        master_addr = "127.0.0.1";
        SPDLOG_WARN("ASTATE_TCPSTORE_MASTER_ADDR not set, using default: {}", master_addr);
    }

    if (master_port_env != nullptr) {
        try {
            master_port = std::stoi(master_port_env);
            SPDLOG_INFO("ASTATE_TCPSTORE_MASTER_PORT: {}", master_port);
        } catch (const std::exception& e) {
            master_port = 29500;
            SPDLOG_WARN("Invalid ASTATE_TCPSTORE_MASTER_PORT: {}, using default: {}", master_port_env, master_port);
        }
    } else {
        master_port = 29500;
        SPDLOG_WARN("ASTATE_TCPSTORE_MASTER_PORT not set, using default: {}", master_port);
    }

    c10d::TCPStoreOptions opts;
    opts.port = master_port;
    opts.isServer = false;
    opts.waitWorkers = false;
    opts.timeout = kDefaultTimeout;
    store_ = std::make_unique<c10d::TCPStore>(master_addr, opts);

    SPDLOG_INFO("TCPStoreConfigCenter initialized with {}:{}", master_addr, master_port);
}

} // namespace astate
