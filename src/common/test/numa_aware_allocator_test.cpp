#include "common/numa_aware_allocator.h"

#include <cstddef>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

namespace astate {
class NumaAwareAllocatorTest : public ::testing::Test {
 protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(NumaAwareAllocatorTest, InitializeWithDefaultNodes) {
    NumaAwareAllocator allocator;
    bool result = allocator.Initialize();

    if (numa_available() >= 0) {
        EXPECT_TRUE(result);
        EXPECT_TRUE(allocator.IsInitialized());
        EXPECT_FALSE(allocator.GetAvailableNodes().empty());
        SPDLOG_INFO("Available NUMA nodes: {}", allocator.GetNodeListString());
    } else {
        EXPECT_FALSE(result);
        EXPECT_FALSE(allocator.IsInitialized());
        SPDLOG_INFO("NUMA not available on this system");
    }
}

TEST_F(NumaAwareAllocatorTest, InitializeWithSpecificNodes) {
    NumaAwareAllocator allocator;
    std::vector<int> preferred_nodes = {0, 1};
    bool result = allocator.Initialize(preferred_nodes);

    if (numa_available() >= 0) {
        if (result) {
            EXPECT_TRUE(allocator.IsInitialized());
            EXPECT_FALSE(allocator.GetAvailableNodes().empty());
            SPDLOG_INFO("Available NUMA nodes: {}", allocator.GetNodeListString());
        } else {
            SPDLOG_INFO("Specified NUMA nodes not available");
        }
    } else {
        EXPECT_FALSE(result);
        EXPECT_FALSE(allocator.IsInitialized());
        SPDLOG_INFO("NUMA not available on this system");
    }
}

TEST_F(NumaAwareAllocatorTest, AllocateMemory) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping allocation test";
    }

    auto test_size = static_cast<size_t>(1024 * 1024); // 1MB
    auto result = allocator.TryAllocateRoundRobinOverGpus(test_size);

    if (result.success) {
        EXPECT_NE(result.ptr, nullptr);
        EXPECT_GE(result.numa_node, 0);
        EXPECT_EQ(result.size, test_size);
        EXPECT_TRUE(result.success);

        SPDLOG_INFO("Allocated {} bytes on NUMA node {}", result.size, result.numa_node);
        allocator.Deallocate(result.ptr, result.size);
    } else {
        SPDLOG_WARN("Memory allocation failed, possibly due to insufficient memory");
    }
}

TEST_F(NumaAwareAllocatorTest, AllocateOnSpecificNode) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping specific node allocation test";
    }

    const auto& available_nodes = allocator.GetAvailableNodes();
    if (available_nodes.empty()) {
        GTEST_SKIP() << "No available NUMA nodes";
    }

    int target_node = available_nodes[0];
    auto test_size = static_cast<size_t>(512 * 1024); // 512KB

    auto result = allocator.TryAllocateNearGpu(test_size, target_node);

    if (result.success) {
        EXPECT_NE(result.ptr, nullptr);
        EXPECT_EQ(result.numa_node, target_node);
        EXPECT_EQ(result.size, test_size);
        EXPECT_TRUE(result.success);

        SPDLOG_INFO("Allocated {} bytes on specific NUMA node {}", result.size, result.numa_node);
        allocator.Deallocate(result.ptr, result.size);
    } else {
        SPDLOG_WARN("Memory allocation on specific node failed");
    }
}

TEST_F(NumaAwareAllocatorTest, TryAllocateOnNode) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping tryAllocateOnNode test";
    }

    const auto& available_nodes = allocator.GetAvailableNodes();
    if (available_nodes.empty()) {
        GTEST_SKIP() << "No available NUMA nodes";
    }

    int target_node = available_nodes[0];
    auto test_size = static_cast<size_t>(256 * 1024); // 256KB

    auto result = allocator.TryAllocateNearGpu(test_size, target_node);

    if (result.success) {
        EXPECT_NE(result.ptr, nullptr);
        EXPECT_EQ(result.numa_node, target_node);
        EXPECT_EQ(result.size, test_size);
        EXPECT_TRUE(result.success);

        SPDLOG_INFO("Try allocated {} bytes on NUMA node {}", result.size, result.numa_node);
        allocator.Deallocate(result.ptr, result.size);
    } else {
        SPDLOG_WARN("TryAllocateOnNode failed, fallback to any node");
    }
}

TEST_F(NumaAwareAllocatorTest, TryAllocateNearCudaDevice) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping CUDA device allocation test";
    }

    int cuda_device_id = 0;
    auto test_size = static_cast<size_t>(128 * 1024); // 128KB

    auto result = allocator.TryAllocateNearGpu(test_size, cuda_device_id);

    if (result.success) {
        EXPECT_NE(result.ptr, nullptr);
        EXPECT_GE(result.numa_node, 0);
        EXPECT_EQ(result.size, test_size);
        EXPECT_TRUE(result.success);

        SPDLOG_INFO(
            "Try allocated {} bytes near CUDA device {} on NUMA node {}",
            result.size,
            cuda_device_id,
            result.numa_node);
        allocator.Deallocate(result.ptr, result.size);
    } else {
        SPDLOG_WARN("TryAllocateNearCudaDevice failed for device {}", cuda_device_id);
    }
}

TEST_F(NumaAwareAllocatorTest, TryAllocateNearCurrentCudaDevice) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping current CUDA device "
                        "allocation test";
    }

    auto test_size = static_cast<size_t>(64 * 1024); // 64KB

    auto result = allocator.TryAllocateNearCurrentDevice(test_size);

    if (result.success) {
        EXPECT_NE(result.ptr, nullptr);
        EXPECT_GE(result.numa_node, 0);
        EXPECT_EQ(result.size, test_size);
        EXPECT_TRUE(result.success);

        SPDLOG_INFO("Try allocated {} bytes near current CUDA device on NUMA node {}", result.size, result.numa_node);
        allocator.Deallocate(result.ptr, result.size);
    } else {
        SPDLOG_WARN("TryAllocateNearCurrentCudaDevice failed");
    }
}

TEST_F(NumaAwareAllocatorTest, GetCurrentProcessNumaId) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping getCurrentProcessNumaId test";
    }

    int numa_id = allocator.GetCurrentProcessNumaId();

    if (numa_id >= 0) {
        EXPECT_GE(numa_id, 0);
        SPDLOG_INFO("Current process NUMA id: {}", numa_id);
    } else {
        SPDLOG_WARN("Failed to get current process NUMA id: {}", numa_id);
    }
}

TEST_F(NumaAwareAllocatorTest, GetNumaNodeForCuda) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping getNumaNodeForCuda test";
    }

    // 测试CUDA设备0
    int cuda_device_id = 0;
    int numa_id = allocator.GetNumaNodeForCuda(cuda_device_id);

    if (numa_id >= 0) {
        EXPECT_GE(numa_id, 0);
        SPDLOG_INFO("CUDA device {} is on NUMA node {}", cuda_device_id, numa_id);
    } else {
        SPDLOG_WARN("CUDA device {} not found or not available", cuda_device_id);
    }
}

TEST_F(NumaAwareAllocatorTest, RoundRobinAllocation) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping round-robin test";
    }

    const auto& available_nodes = allocator.GetAvailableNodes();
    if (available_nodes.size() < 2) {
        GTEST_SKIP() << "Need at least 2 NUMA nodes for round-robin test";
    }

    std::vector<int> allocated_nodes;
    auto test_size = static_cast<size_t>(256 * 1024); // 256KB

    for (int i = 0; i < 5; ++i) {
        auto result = allocator.TryAllocateRoundRobinOverGpus(test_size);
        if (result.success) {
            allocated_nodes.push_back(result.numa_node);
            SPDLOG_INFO("Allocation {} on NUMA node {}", i, result.numa_node);
            allocator.Deallocate(result.ptr, result.size);
        }
    }

    if (allocated_nodes.size() > 1) {
        std::sort(allocated_nodes.begin(), allocated_nodes.end());
        auto unique_end = std::unique(allocated_nodes.begin(), allocated_nodes.end());
        size_t unique_count = std::distance(allocated_nodes.begin(), unique_end);

        SPDLOG_INFO("Allocated on {} different nodes out of {} available", unique_count, available_nodes.size());
    }
}

TEST_F(NumaAwareAllocatorTest, AllocateZeroBytes) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping zero allocation test";
    }

    auto result = allocator.TryAllocateRoundRobinOverGpus(0);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.ptr, nullptr);
}

TEST_F(NumaAwareAllocatorTest, AllocateWithoutInitialization) {
    NumaAwareAllocator allocator;

    auto result = allocator.TryAllocateRoundRobinOverGpus(1024);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.ptr, nullptr);
}

TEST_F(NumaAwareAllocatorTest, AllocateOnInvalidNode) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping invalid node test";
    }

    auto result = allocator.TryAllocateNearGpu(1024, 999); // 假设节点 999 不存在
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.ptr, nullptr);
}

TEST_F(NumaAwareAllocatorTest, MultipleAllocations) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping multiple allocations test";
    }

    std::vector<std::pair<void*, size_t>> allocations;
    size_t test_size = 1024; // 1KB

    // 分配多个内存块
    for (int i = 0; i < 10; ++i) {
        auto result = allocator.TryAllocateRoundRobinOverGpus(test_size);
        if (result.success) {
            allocations.emplace_back(result.ptr, result.size);
            SPDLOG_INFO("Allocation {} on NUMA node {}", i, result.numa_node);
        }
    }

    // 释放所有内存
    for (const auto& alloc : allocations) {
        allocator.Deallocate(alloc.first, alloc.second);
    }

    EXPECT_EQ(allocations.size(), 10);
    SPDLOG_INFO("Successfully allocated and deallocated {} memory blocks", allocations.size());
}

TEST_F(NumaAwareAllocatorTest, LargeAllocation) {
    NumaAwareAllocator allocator;
    if (!allocator.Initialize()) {
        GTEST_SKIP() << "NUMA not available, skipping large allocation test";
    }

    auto large_size = static_cast<size_t>(100 * 1024 * 1024); // 100MB
    auto result = allocator.TryAllocateRoundRobinOverGpus(large_size);

    if (result.success) {
        EXPECT_NE(result.ptr, nullptr);
        EXPECT_EQ(result.size, large_size);
        EXPECT_TRUE(result.success);

        SPDLOG_INFO("Large allocation successful: {} bytes on NUMA node {}", result.size, result.numa_node);
        allocator.Deallocate(result.ptr, result.size);
    } else {
        SPDLOG_WARN("Large allocation failed, possibly due to insufficient memory");
    }
}
} // namespace astate
