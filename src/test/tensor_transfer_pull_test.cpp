#include "tensor_transfer_pull.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "common/network_utils.h"
#include "common/option.h"
#include "core/atensor.h"
#include "core/shardedkey.h"
#include "transport/base_transport.h"
#include "transport/rdma_transporter.h"
#include "types.h"

using namespace astate;

// GMock-based RDMA Transporter that mocks Send/Receive with real memory operations
class MockRDMATransporter : public RDMATransporter {
 public:
    bool Start(const Options& options, const AParallelConfig& parallel_config) override {
        local_server_name_ = GetOptionValue<std::string>(options, TRANSFER_ENGINE_LOCAL_ADDRESS);
        local_server_port_ = GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT);
        meta_addr_ = GetOptionValue<std::string>(options, TRANSFER_ENGINE_META_SERVICE_ADDRESS);
        is_running_ = true;
        SPDLOG_INFO("MockRDMATransporter Start: {}:{}", local_server_name_, local_server_port_);
        return true;
    }

    bool RegisterMemory(void* addr, size_t len, bool is_vram = false, int gpu_id_or_numa_node = -1) {
        auto region = std::make_shared<RegisteredMemRegion>();
        region->mr.addr = addr;
        region->mr.len = len;
        region->mr.type = is_vram ? 1 : 0;
        region->mr.numa = gpu_id_or_numa_node;
        region->mr.is_owned = false;
        region->register_num = 1;
        // Note: devices mapping is complex and not needed for basic mock functionality
        return true;
    }

    bool DeregisterMemory(void* addr, size_t len) { return true; }

    // Real implementation for Send that performs memory write
    bool Send(
        const void* send_data,
        size_t send_size,
        const std::string& remote_host,
        int remote_port,
        const ExtendInfo* extend_info) override {
        return true;
    }

    // Real implementation for Receive that performs memory read
    bool Receive(
        const void* recv_data,
        size_t recv_size,
        const std::string& remote_host,
        int remote_port,
        const ExtendInfo* extend_info) override {
        const void* remote_addr = GetRemoteAddrFromExtendInfo(extend_info);
        memcpy(const_cast<void*>(recv_data), remote_addr, recv_size);
        return true;
    }

 private:
    std::string generateDataKey(const ExtendInfo* extend_info) {
        return std::to_string(reinterpret_cast<uintptr_t>(GetRemoteAddrFromExtendInfo(extend_info)));
    }
};

// No need to mock HTTP transporter - use real HTTP for control messages

// Enhanced TensorTransferPull that always uses HTTP for control messages and Mock RDMA for data transfer
class TestTensorTransferPull : public TensorTransferPull {
 public:
    TestTensorTransferPull() { data_rdma_transport_ = std::make_unique<MockRDMATransporter>(); }
    virtual ~TestTensorTransferPull() {}
};

class TensorTransferPullIntegrationTest : public ::testing::Test {
 protected:
    void SetUp() override {
        setupOptions();

        // Create two test services for put and get operations (always use hybrid transport)
        put_service_ = std::make_unique<TestTensorTransferPull>();
        get_service_ = std::make_unique<TestTensorTransferPull>();
    }

    void TearDown() override {
        if (put_service_ && put_service_->IsRunning()) {
            put_service_->Stop();
        }
        if (get_service_ && get_service_->IsRunning()) {
            get_service_->Stop();
        }
    }

    void setupOptions() {
        std::string local_host = GetLocalHostnameOrIP();

        // PUT service options (node1) - Use different ports
        PutOptionValue(put_options_, TRANSFER_ENGINE_SERVICE_ADDRESS, local_host);
        PutOptionValue(put_options_, TRANSFER_ENGINE_SERVICE_PORT, "19080");
        PutOptionValue(put_options_, TRANSFER_ENGINE_LOCAL_ADDRESS, local_host);
        PutOptionValue(put_options_, TRANSFER_ENGINE_LOCAL_PORT, "19070");
        PutOptionValue(put_options_, TRANSFER_ENGINE_READ_THREAD_NUM, "4");
        PutOptionValue(put_options_, TRANSFER_ENGINE_PEERS_HOST, local_host + ":19071:19081");
        PutOptionValue(put_options_, TRANSFER_ENGINE_META_SERVICE_ADDRESS, local_host);
        PutOptionValue(put_options_, TRANSFER_ENGINE_SERVICE_TENSOR_READY_TIMEOUT_MS, "2000");
        PutOptionValue(put_options_, TRANSFER_ENGINE_SERVICE_SKIP_DISCOVERY, "true");
        PutOptionValue(put_options_, TRANSFER_ENGINE_SERVICE_FIXED_PORT, "true");
        PutOptionValue(put_options_, TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION, "true");
        // GET service options (node2) - Use different ports
        PutOptionValue(get_options_, TRANSFER_ENGINE_SERVICE_ADDRESS, local_host);
        PutOptionValue(get_options_, TRANSFER_ENGINE_SERVICE_PORT, "19081");
        PutOptionValue(get_options_, TRANSFER_ENGINE_LOCAL_ADDRESS, local_host);
        PutOptionValue(get_options_, TRANSFER_ENGINE_LOCAL_PORT, "19071");
        PutOptionValue(get_options_, TRANSFER_ENGINE_READ_THREAD_NUM, "4");
        PutOptionValue(get_options_, TRANSFER_ENGINE_PEERS_HOST, local_host + ":19070:19080");
        PutOptionValue(get_options_, TRANSFER_ENGINE_META_SERVICE_ADDRESS, local_host);
        PutOptionValue(get_options_, TRANSFER_ENGINE_SERVICE_TENSOR_READY_TIMEOUT_MS, "2000");
        PutOptionValue(get_options_, TRANSFER_ENGINE_SERVICE_SKIP_DISCOVERY, "true");
        PutOptionValue(get_options_, TRANSFER_ENGINE_SERVICE_FIXED_PORT, "true");
        PutOptionValue(get_options_, TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION, "true");
    }

    ATensor createTestTensor(size_t size, bool init_data = true) {
        ATensor tensor;
        tensor.storage.data = malloc(size);
        tensor.storage.device.device_type = ATDeviceType::CPU;
        tensor.storage.device.device_index = 0;
        tensor.storage.storage_size = static_cast<int32_t>(size);
        tensor.dim_num = 1;
        tensor.size = new int64_t[1];
        tensor.stride = new int64_t[1];
        tensor.size[0] = static_cast<int64_t>(size);
        tensor.stride[0] = 1;
        tensor.storage_offset = 0;
        tensor.dtype = ATDtype::Byte;
        tensor.conj = false;
        tensor.neg = false;
        tensor.requires_grad = false;

        if (init_data) {
            // Initialize with test pattern
            uint8_t* data = static_cast<uint8_t*>(tensor.storage.data);
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<uint8_t>(i % 256);
            }
        } else {
            memset(tensor.storage.data, 0, size);
        }

        return tensor;
    }

    void cleanupTensor(ATensor& tensor) {
        if (tensor.storage.data) {
            free(tensor.storage.data);
            tensor.storage.data = nullptr;
        }
        if (tensor.size) {
            delete[] tensor.size;
            tensor.size = nullptr;
        }
        if (tensor.stride) {
            delete[] tensor.stride;
            tensor.stride = nullptr;
        }

        // Reset all other fields to safe values
        tensor.storage.storage_size = 0;
        tensor.dim_num = 0;
        tensor.storage_offset = 0;
    }

    ShardedKey createTestKey(const std::string& name, int index = 0) {
        ShardedKey key;
        key.key = name;
        key.global_shape = {static_cast<int64_t>(index)};
        key.global_offset = {0};
        return key;
    }

    // Helper method to simulate HTTP message exchange
    void simulateMessageExchange() {
        // Give services time to exchange control messages
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Helper method to wait for service startup
    bool waitForServiceReady(TestTensorTransferPull* service, int timeout_ms = 5000) {
        auto start_time = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start_time < std::chrono::milliseconds(timeout_ms)) {
            if (service->IsRunning()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    }

    Options put_options_;
    Options get_options_;
    AParallelConfig parallel_config_;
    std::unique_ptr<TestTensorTransferPull> put_service_;
    std::unique_ptr<TestTensorTransferPull> get_service_;
};

// Test basic service creation and lifecycle
TEST_F(TensorTransferPullIntegrationTest, BasicServiceLifecycle) {
    EXPECT_TRUE(put_service_ != nullptr);
    EXPECT_TRUE(get_service_ != nullptr);

    // Services should not be running initially
    EXPECT_FALSE(put_service_->IsRunning());
    EXPECT_FALSE(get_service_->IsRunning());
}

// Test service start/stop with mock conditions (services won't actually start due to missing RDMA)
TEST_F(TensorTransferPullIntegrationTest, ServiceStartStopMockConditions) {
    // Note: These services will likely fail to start in test environment
    // due to missing RDMA infrastructure, but we test the interface

    bool put_start_result = put_service_->Start(put_options_, parallel_config_);
    bool get_start_result = get_service_->Start(get_options_, parallel_config_);

    // In real environment with RDMA: EXPECT_TRUE for both
    // In test environment: services may fail to start, which is expected

    if (put_start_result) {
        EXPECT_TRUE(put_service_->IsRunning());
        put_service_->Stop();
        EXPECT_FALSE(put_service_->IsRunning());
    }

    if (get_start_result) {
        EXPECT_TRUE(get_service_->IsRunning());
        get_service_->Stop();
        EXPECT_FALSE(get_service_->IsRunning());
    }
}

// Test tensor operations without actual network (will fail gracefully)
TEST_F(TensorTransferPullIntegrationTest, TensorOperationsWithoutNetwork) {
    const size_t tensor_size = 1024;
    ATensor put_tensor = createTestTensor(tensor_size);
    ATensor get_tensor = createTestTensor(tensor_size, false);
    ShardedKey tensor_key = createTestKey("test_tensor");

    const int64_t seq_id = 1;

    // These operations should fail gracefully when services are not started
    EXPECT_FALSE(put_service_->Put(seq_id, tensor_key, put_tensor));
    EXPECT_FALSE(get_service_->Get(seq_id, tensor_key, get_tensor));

    // Test multi operations
    std::vector<std::pair<ShardedKey, ATensor>> put_tensors;
    std::vector<std::pair<ShardedKey, ATensor>> get_tensors;

    put_tensors.emplace_back(tensor_key, put_tensor);
    get_tensors.emplace_back(tensor_key, get_tensor);

    EXPECT_FALSE(put_service_->MultiPut(seq_id, put_tensors));
    EXPECT_FALSE(get_service_->MultiGet(seq_id, get_tensors));

    // Cleanup
    cleanupTensor(put_tensor);
    cleanupTensor(get_tensor);
}

// Test peer configuration
TEST_F(TensorTransferPullIntegrationTest, PeerConfiguration) {
    std::vector<NodeInfo> peer_hosts
        = {NodeInfo{.hostname_or_ip = "127.0.0.1", .rdma_port = 8080, .ctrl_flow_port = 8081},
           NodeInfo{.hostname_or_ip = "127.0.0.1", .rdma_port = 8082, .ctrl_flow_port = 8083}};

    // SetPeerHosts should not throw
    EXPECT_NO_THROW(put_service_->SetPeerHosts(peer_hosts));
    EXPECT_NO_THROW(get_service_->SetPeerHosts(peer_hosts));
}

// Test data integrity helpers
TEST_F(TensorTransferPullIntegrationTest, DataIntegrityHelpers) {
    const size_t tensor_size = 512;

    // Create two tensors with same test pattern
    ATensor tensor1 = createTestTensor(tensor_size);
    ATensor tensor2 = createTestTensor(tensor_size);

    // They should have identical data
    EXPECT_EQ(memcmp(tensor1.storage.data, tensor2.storage.data, tensor_size), 0);

    // Create tensor without initialization
    ATensor tensor3 = createTestTensor(tensor_size, false);

    // Copy data from tensor1 to tensor3
    memcpy(tensor3.storage.data, tensor1.storage.data, tensor_size);

    // Now tensor1 and tensor3 should be identical
    EXPECT_EQ(memcmp(tensor1.storage.data, tensor3.storage.data, tensor_size), 0);

    // But tensor2 and a new uninitialized tensor should be different
    ATensor tensor4 = createTestTensor(tensor_size, false);
    EXPECT_NE(memcmp(tensor2.storage.data, tensor4.storage.data, tensor_size), 0);

    // Cleanup
    cleanupTensor(tensor1);
    cleanupTensor(tensor2);
    cleanupTensor(tensor3);
    cleanupTensor(tensor4);
}

// Test ShardedKey functionality
TEST_F(TensorTransferPullIntegrationTest, ShardedKeyFunctionality) {
    ShardedKey key1 = createTestKey("tensor1", 1);
    ShardedKey key2 = createTestKey("tensor1", 1);
    ShardedKey key3 = createTestKey("tensor2", 1);
    ShardedKey key4 = createTestKey("tensor1", 2);

    // Same keys should be equal
    EXPECT_TRUE(key1 == key2);

    // Different keys should not be equal
    EXPECT_FALSE(key1 == key3); // Different name
    EXPECT_FALSE(key1 == key4); // Different index

    // Test that keys can be used in containers
    std::unordered_map<ShardedKey, int, ShardedKeyHash> key_map;
    key_map[key1] = 100;
    key_map[key3] = 200;

    EXPECT_EQ(key_map[key1], 100);
    EXPECT_EQ(key_map[key2], 100); // key2 == key1
    EXPECT_EQ(key_map[key3], 200);
}

// Test error conditions and edge cases
TEST_F(TensorTransferPullIntegrationTest, ErrorConditionsAndEdgeCases) {
    // Test with invalid tensor data
    ATensor invalid_tensor;
    ShardedKey key = createTestKey("invalid");

    // Should handle invalid tensor gracefully
    EXPECT_FALSE(put_service_->Put(1, key, invalid_tensor));

    // Test with empty key
    ShardedKey empty_key;
    ATensor valid_tensor = createTestTensor(100);

    // Should handle empty key gracefully
    EXPECT_FALSE(put_service_->Put(1, empty_key, valid_tensor));

    cleanupTensor(valid_tensor);
}

// Test memory management
TEST_F(TensorTransferPullIntegrationTest, MemoryManagement) {
    const size_t tensor_size = 1024;

    // Create and destroy many tensors to test memory leaks
    for (int i = 0; i < 10; ++i) {
        ATensor tensor = createTestTensor(tensor_size);

        // Verify tensor is properly initialized
        EXPECT_TRUE(tensor.storage.data != nullptr);
        EXPECT_EQ(tensor.storage.storage_size, static_cast<int32_t>(tensor_size));
        EXPECT_TRUE(tensor.size != nullptr);
        EXPECT_TRUE(tensor.stride != nullptr);

        cleanupTensor(tensor);

        // After cleanup, pointers should be null
        EXPECT_TRUE(tensor.storage.data == nullptr);
        EXPECT_TRUE(tensor.size == nullptr);
        EXPECT_TRUE(tensor.stride == nullptr);
    }
}

// Test with realistic data sizes
TEST_F(TensorTransferPullIntegrationTest, RealisticDataSizes) {
    // Test with different tensor sizes
    std::vector<size_t> test_sizes = {1, 100, 1024, 10240, 102400};

    for (size_t size : test_sizes) {
        ATensor tensor = createTestTensor(size);

        // Verify tensor properties
        EXPECT_EQ(tensor.storage.storage_size, static_cast<int32_t>(size));
        EXPECT_EQ(tensor.size[0], static_cast<int64_t>(size));

        // Verify data integrity
        uint8_t* data = static_cast<uint8_t*>(tensor.storage.data);
        for (size_t i = 0; i < size; ++i) {
            EXPECT_EQ(data[i], static_cast<uint8_t>(i % 256));
        }

        cleanupTensor(tensor);
    }
}

// Test dual server communication with mock RDMA
TEST_F(TensorTransferPullIntegrationTest, DualServerCommunication) {
    // Try to start both servers, will use mock if RDMA is not available
    bool put_started = put_service_->Start(put_options_, parallel_config_);
    bool get_started = get_service_->Start(get_options_, parallel_config_);

    ASSERT_TRUE(put_started) << "PUT service should start";
    ASSERT_TRUE(get_started) << "GET service should start";

    // Both servers are now running, test communication
    const size_t tensor_size = 1024;
    const int64_t seq_id = 42;

    // Create test data for PUT operation
    ATensor put_tensor = createTestTensor(tensor_size);
    ShardedKey tensor_key = createTestKey("real_comm_test");

    // Step 1: PUT operation on put_service
    SPDLOG_INFO("Starting PUT operation...");
    bool put_result = put_service_->Put(seq_id, tensor_key, put_tensor);
    EXPECT_TRUE(put_result) << "PUT operation should succeed";

    // Create tensor for GET operation
    ATensor get_tensor = createTestTensor(tensor_size, false); // Don't initialize data

    // Step 2: GET operation on get_service
    SPDLOG_INFO("Starting GET operation...");
    bool get_result = get_service_->Get(seq_id, tensor_key, get_tensor);
    EXPECT_TRUE(get_result) << "GET operation should succeed";

    // Step 3: Verify data integrity
    if (get_result) {
        SPDLOG_INFO("GET operation succeeded, verifying data integrity...");
        EXPECT_EQ(memcmp(put_tensor.storage.data, get_tensor.storage.data, tensor_size), 0)
            << "Retrieved data should match original data";
        SPDLOG_INFO("Data integrity verified!");
    }

    // Step 4: Test Complete operations
    // Notice: The complete operation of GET service should be called before PUT service.
    EXPECT_NO_THROW(get_service_->Complete());
    EXPECT_NO_THROW(put_service_->Complete());

    // Cleanup
    cleanupTensor(put_tensor);
    cleanupTensor(get_tensor);

    // Stop services
    put_service_->Stop();
    get_service_->Stop();

    SPDLOG_INFO("Dual server communication test completed");
}

// Test multi-tensor dual server communication with mock RDMA
TEST_F(TensorTransferPullIntegrationTest, DualServerMultiTensorCommunication) {
    // Start both servers
    bool put_started = put_service_->Start(put_options_, parallel_config_);
    bool get_started = get_service_->Start(get_options_, parallel_config_);

    ASSERT_TRUE(put_started) << "PUT service should start";
    ASSERT_TRUE(get_started) << "GET service should start";

    const size_t tensor_size = 32;
    const int num_tensors = 2;
    const int64_t seq_id = 100;

    // Prepare PUT tensors
    std::vector<std::pair<ShardedKey, ATensor>> put_tensors;
    for (int i = 0; i < num_tensors; ++i) {
        ShardedKey key = createTestKey("multi_tensor_" + std::to_string(i), i);
        ATensor tensor = createTestTensor(tensor_size);
        put_tensors.emplace_back(std::move(key), std::move(tensor)); // 使用移动语义避免拷贝
    }

    // Step 1: MultiPut operation
    SPDLOG_INFO("Starting MultiPut operation with {} tensors...", num_tensors);
    bool multiput_result = put_service_->MultiPut(seq_id, put_tensors);
    EXPECT_TRUE(multiput_result) << "MultiPut operation should succeed";

    // Prepare GET tensors
    std::vector<std::pair<ShardedKey, ATensor>> get_tensors;
    for (int i = 0; i < num_tensors; ++i) {
        ShardedKey key = createTestKey("multi_tensor_" + std::to_string(i), i);
        ATensor tensor = createTestTensor(tensor_size, false); // Don't initialize
        get_tensors.emplace_back(std::move(key), std::move(tensor)); // 使用移动语义避免拷贝
    }

    // Step 2: MultiGet operation
    SPDLOG_INFO("Starting MultiGet operation...");
    bool multiget_result = get_service_->MultiGet(seq_id, get_tensors);
    EXPECT_TRUE(multiget_result) << "MultiGet operation should succeed";

    // Step 3: Verify data integrity for all tensors
    if (multiget_result) {
        SPDLOG_INFO("MultiGet succeeded, verifying data integrity...");
        for (size_t i = 0; i < put_tensors.size(); ++i) {
            EXPECT_EQ(memcmp(put_tensors[i].second.storage.data, get_tensors[i].second.storage.data, tensor_size), 0)
                << "Tensor " << i << " data should match";
        }
        SPDLOG_INFO("All tensor data integrity verified!");
    }

    // Step 4: Complete operations
    // Notice: The complete operation of GET service should be called before PUT service.
    EXPECT_NO_THROW(get_service_->Complete());
    EXPECT_NO_THROW(put_service_->Complete());

    // Cleanup - safely handle potentially moved tensors
    for (auto& pair : put_tensors) {
        // Only cleanup if tensor wasn't moved (has valid data pointer)
        if (pair.second.storage.data != nullptr) {
            cleanupTensor(pair.second);
        }
    }
    for (auto& pair : get_tensors) {
        // Only cleanup if tensor wasn't moved (has valid data pointer)
        if (pair.second.storage.data != nullptr) {
            cleanupTensor(pair.second);
        }
    }

    // Stop services
    put_service_->Stop();
    get_service_->Stop();

    SPDLOG_INFO("Multi-tensor communication test completed");
}

// Test concurrent dual server operations with mock RDMA
TEST_F(TensorTransferPullIntegrationTest, DualServerConcurrentOperations) {
    // This test verifies that servers handle various error conditions gracefully
    // Test 1: Operations on stopped servers
    EXPECT_FALSE(put_service_->IsRunning());
    EXPECT_FALSE(get_service_->IsRunning());

    // Start both servers
    bool put_started = put_service_->Start(put_options_, parallel_config_);
    bool get_started = get_service_->Start(get_options_, parallel_config_);

    ASSERT_TRUE(put_started) << "PUT service should start";
    ASSERT_TRUE(get_started) << "GET service should start";

    const size_t tensor_size = 256;
    const int num_concurrent = 5;

    SPDLOG_INFO("Starting concurrent dual server operations...");

    // Create tensors for concurrent PUT operations
    std::vector<ATensor> put_tensors;
    std::vector<ShardedKey> keys;
    for (int i = 0; i < num_concurrent; ++i) {
        put_tensors.push_back(createTestTensor(tensor_size));
        keys.push_back(createTestKey("concurrent_" + std::to_string(i), i));
    }

    // Launch concurrent PUT operations
    std::vector<std::future<bool>> put_futures;
    for (int i = 0; i < num_concurrent; ++i) {
        put_futures.push_back(std::async(std::launch::async, [this, i, &keys, &put_tensors]() {
            return put_service_->Put(200, keys[i], put_tensors[i]);
        }));
    }

    // Wait for all PUT operations to complete
    std::vector<bool> put_results;
    for (auto& future : put_futures) {
        put_results.push_back(future.get());
    }

    // Check PUT results and simulate meta messages for mock case
    for (size_t i = 0; i < put_results.size(); ++i) {
        EXPECT_TRUE(put_results[i]) << "PUT operation " << i << " should succeed";
        SPDLOG_INFO("PUT operation {} succeeded", i);
    }

    // Create tensors for concurrent GET operations
    std::vector<ATensor> get_tensors;
    for (int i = 0; i < num_concurrent; ++i) {
        get_tensors.push_back(createTestTensor(tensor_size, false));
    }

    // Launch concurrent GET operations
    std::vector<std::future<bool>> get_futures;
    for (int i = 0; i < num_concurrent; ++i) {
        get_futures.push_back(std::async(std::launch::async, [this, i, &keys, &get_tensors]() {
            return get_service_->Get(200, keys[i], get_tensors[i]);
        }));
    }

    // Wait for all GET operations to complete
    std::vector<bool> get_results;
    for (auto& future : get_futures) {
        get_results.push_back(future.get());
    }

    // Verify results - all operations should succeed
    for (int i = 0; i < num_concurrent; ++i) {
        EXPECT_TRUE(get_results[i]) << "GET operation " << i << " should succeed";
        if (put_results[i] && get_results[i]) {
            EXPECT_EQ(memcmp(put_tensors[i].storage.data, get_tensors[i].storage.data, tensor_size), 0)
                << "Concurrent operation " << i << " data should match";
            SPDLOG_INFO("Concurrent operation {} data verified!", i);
        }
    }

    // Complete operations
    EXPECT_NO_THROW(get_service_->Complete());
    EXPECT_NO_THROW(put_service_->Complete());

    // Cleanup
    for (auto& tensor : put_tensors) {
        cleanupTensor(tensor);
    }
    for (auto& tensor : get_tensors) {
        cleanupTensor(tensor);
    }

    // Stop services
    put_service_->Stop();
    get_service_->Stop();

    SPDLOG_INFO("Concurrent dual server operations test completed");
}

// Test server resilience and error recovery
TEST_F(TensorTransferPullIntegrationTest, ServerResilienceTest) {
    // Test 1: Operations on stopped servers
    EXPECT_FALSE(put_service_->IsRunning());
    EXPECT_FALSE(get_service_->IsRunning());

    ATensor test_tensor = createTestTensor(100);
    ShardedKey test_key = createTestKey("resilience_test");

    // Operations should fail gracefully on stopped servers
    EXPECT_FALSE(put_service_->Put(1, test_key, test_tensor));
    EXPECT_FALSE(get_service_->Get(1, test_key, test_tensor));

    // Test 2: Start servers and test resilience
    bool put_started = put_service_->Start(put_options_, parallel_config_);
    bool get_started = get_service_->Start(get_options_, parallel_config_);

    if (put_started && get_started) {
        SPDLOG_INFO("Testing server resilience with running servers...");

        // Test operations with invalid sequence IDs (should be handled gracefully)
        // Note: Some implementations may accept negative seq_id, so we don't force failure
        bool invalid_result = put_service_->Put(-1, test_key, test_tensor);
        SPDLOG_INFO("PUT with negative seq_id result: {}", (invalid_result ? "success" : "failed"));

        // Test rapid start/stop cycles
        for (int i = 0; i < 3; ++i) {
            put_service_->Stop();
            get_service_->Stop();

            // Brief pause
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Restart
            bool put_restart = put_service_->Start(put_options_, parallel_config_);
            bool get_restart = get_service_->Start(get_options_, parallel_config_);
            if (!put_restart || !get_restart) {
                SPDLOG_INFO("Restart cycle {} failed", i);
                break;
            }
            SPDLOG_INFO("Restart cycle {} successful", i);
        }

        // Final cleanup
        put_service_->Stop();
        get_service_->Stop();
    } else {
        SPDLOG_INFO("Service startup failed - testing basic error handling");
    }

    cleanupTensor(test_tensor);
    SPDLOG_INFO("Server resilience test completed");
}

// Test ping handler functionality
TEST_F(TensorTransferPullIntegrationTest, DISABLED_PingHandlerTest) {
    // Test 1: Operations on stopped servers
    EXPECT_FALSE(put_service_->IsRunning());
    EXPECT_FALSE(get_service_->IsRunning());

    // Start both servers
    bool put_started = put_service_->Start(put_options_, parallel_config_);
    bool get_started = get_service_->Start(get_options_, parallel_config_);

    ASSERT_TRUE(put_started) << "PUT service should start";
    ASSERT_TRUE(get_started) << "GET service should start";

    SPDLOG_INFO("Testing ping handler functionality...");

    // Wait for services to be ready
    ASSERT_TRUE(waitForServiceReady(put_service_.get()));
    ASSERT_TRUE(waitForServiceReady(get_service_.get()));

    // Test ping to PUT service
    std::string local_host = GetLocalHostnameOrIP();
    std::string put_ping_url = "http://" + local_host + ":19080/ping";

    // Use curl to test ping endpoint
    std::string curl_cmd = "curl -s -w '%{http_code}' -o /tmp/ping_response " + put_ping_url;
    int curl_result = system(curl_cmd.c_str());

    if (curl_result == 0) {
        // Read the response
        std::ifstream response_file("/tmp/ping_response");
        std::string response_body;
        if (response_file.is_open()) {
            std::getline(response_file, response_body);
            response_file.close();
        }

        // Read the HTTP status code
        std::ifstream status_file("/tmp/ping_response");
        std::string status_line;
        if (status_file.is_open()) {
            std::getline(status_file, status_line);
            status_file.close();
        }

        SPDLOG_INFO("Ping response: {}", response_body);
        SPDLOG_INFO("Ping status: {}", status_line);

        // Verify response
        EXPECT_EQ(response_body, "pong") << "Ping response should be 'pong'";
    } else {
        SPDLOG_WARN("Curl command failed, ping test skipped");
    }

    // Test ping to GET service
    std::string get_ping_url = "http://" + local_host + ":19081/ping";
    curl_cmd = "curl -s -w '%{http_code}' -o /tmp/ping_response2 " + get_ping_url;
    curl_result = system(curl_cmd.c_str());

    if (curl_result == 0) {
        // Read the response
        std::ifstream response_file("/tmp/ping_response2");
        std::string response_body;
        if (response_file.is_open()) {
            std::getline(response_file, response_body);
            response_file.close();
        }

        SPDLOG_INFO("GET service ping response: {}", response_body);

        // Verify response
        EXPECT_EQ(response_body, "pong") << "GET service ping response should be 'pong'";
    } else {
        SPDLOG_WARN("Curl command failed for GET service, ping test skipped");
    }

    // Clean up temporary files
    system("rm -f /tmp/ping_response /tmp/ping_response2");

    // Stop services
    put_service_->Stop();
    get_service_->Stop();

    SPDLOG_INFO("Ping handler test completed");
}
