#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <numa.h>

#include <glog/logging.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/resource.h>

#include "common/cuda_utils.h"
#include "common/string_utils.h"

namespace astate {

class NumaAwareAllocator {
 public:
    struct AllocationResult {
        void* ptr;
        int numa_node;
        size_t size;
        bool success;
        AllocationResult()
            : ptr(nullptr),
              numa_node(-1),
              size(0),
              success(false) {}
        AllocationResult(void* p, int node, size_t s, bool ok)
            : ptr(p),
              numa_node(node),
              size(s),
              success(ok) {}
    };

    bool Initialize(const std::vector<int>& preferred_nodes = {}) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (initialized_) {
            return true;
        }
        if (numa_available() < 0) {
            SPDLOG_ERROR("NUMA not available on this system");
            return false;
        }

        cuda_available_ = HasNvGpu();
        if (cuda_available_) {
            SPDLOG_INFO("CUDA runtime detected -> GPU-aware NUMA path will be "
                        "used when applicable.");
        } else {
            SPDLOG_INFO("No CUDA runtime/device -> pure CPU NUMA path.");
        }

        if (preferred_nodes.empty()) {
            struct bitmask* bm = numa_all_nodes_ptr;
            for (int node = 0; node <= numa_max_node(); ++node) {
                if (numa_bitmask_isbitset(bm, node) != 0) {
                    long long free_bytes = 0;
                    if (numa_node_size64(node, &free_bytes) > 0) {
                        available_nodes_.push_back(node);
                    }
                }
            }
        } else {
            for (int node : preferred_nodes) {
                if (node >= 0 && node <= numa_max_node()) {
                    void* test_ptr = numa_alloc_onnode(1024, node);
                    if (test_ptr != nullptr) {
                        numa_free(test_ptr, 1024);
                        available_nodes_.push_back(node);
                    }
                }
            }
        }

        if (available_nodes_.empty()) {
            SPDLOG_ERROR("No available NUMA nodes found");
            return false;
        }

        if (cuda_available_) {
            DetectCudaNumaTopology();
        }

        current_node_index_ = 0;
        initialized_ = true;
        SPDLOG_INFO("NumaAwareAllocator initialized with {} nodes: {}", available_nodes_.size(), GetNodeListString());
        return true;
    }

    int GetCurrentProcessNumaId() {
        if (cuda_available_) {
            int dev = -1;
            if (cudaGetDevice(&dev) != cudaSuccess) {
                return -1;
            }
            return GetNumaNodeForCuda(dev);
        }

        // ! if already set numa >=0
        int pref = numa_preferred();
        if (pref >= 0) {
            return pref;
        }

        int cpu = sched_getcpu();
        if (cpu >= 0) {
            int node = numa_node_of_cpu(cpu);
            if (node >= 0) {
                return node;
            }
        }
        return 0;
    }

    AllocationResult TryAllocateNearGpu(size_t size, int cuda_id) {
        int numa_node = GetNumaNodeForCuda(cuda_id);
        if (numa_node == -1) {
            SPDLOG_WARN(
                "No NUMA mapping for CUDA device {}, falling back to "
                "allocate()",
                cuda_id);
            return TryAllocateRoundRobinOverGpus(size);
        }
        return AllocatePinned(size, numa_node);
    }

    AllocationResult TryAllocateNearCpu(size_t size) {
        int node = GetCurrentProcessNumaId();
        node = std::max(node, 0);
        // ! cpu path do not use cuda register
        return AllocatePinned(size, node);
    }

    AllocationResult TryAllocateNearCurrentDevice(size_t size) {
        if (cuda_available_) {
            int dev = -1;
            if (cudaGetDevice(&dev) == cudaSuccess) {
                return TryAllocateNearGpu(size, dev);
            }
            SPDLOG_WARN("cudaGetDevice failed, fallback to TryAllocateRoundRobin");
            return TryAllocateRoundRobinOverGpus(size);
        }
        return TryAllocateNearCpu(size);
    }

    // Round-robin among available nodes, but always pinned
    AllocationResult TryAllocateRoundRobinOverGpus(size_t size) {
        if (!initialized_) {
            SPDLOG_ERROR("NumaAwareAllocator not initialized");
            return {};
        }
        if (size == 0) {
            SPDLOG_WARN("Attempting to allocate 0 bytes");
            return {};
        }
        std::lock_guard<std::mutex> lock(alloc_mutex_);
        for (size_t attempt = 0; attempt < available_nodes_.size(); ++attempt) {
            int node = available_nodes_[current_node_index_];
            current_node_index_ = (current_node_index_ + 1) % available_nodes_.size();
            void* ptr = numa_alloc_onnode(size, node);
            if (ptr == nullptr) {
                continue;
            }

            if (cuda_available_) {
                cudaError_t rc = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
                if (rc != cudaSuccess) {
                    SPDLOG_ERROR("cudaHostRegister failed on node {}: {}", node, cudaGetErrorString(rc));
                    numa_free(ptr, size);
                    continue;
                }
            }

            return {ptr, node, size, true};
        }
        SPDLOG_ERROR("Failed to allocate {} bytes on any NUMA node (pinned)", size);
        return {};
    }

    AllocationResult AllocatePinned(size_t size, int numa_node) {
        if (!initialized_) {
            SPDLOG_ERROR("NumaAwareAllocator not initialized");
            return {};
        }
        if (size == 0) {
            SPDLOG_WARN("Attempting to allocate 0 bytes");
            return {};
        }
        if (std::find(available_nodes_.begin(), available_nodes_.end(), numa_node) == available_nodes_.end()) {
            SPDLOG_ERROR("NUMA node {} is not available", numa_node);
            return {};
        }
        void* ptr = numa_alloc_onnode(size, numa_node);
        if (ptr == nullptr) {
            SPDLOG_ERROR("numa_alloc_onnode failed node={}, size={}", numa_node, size);
            return {};
        }

        if (cuda_available_) {
            cudaError_t rc = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
            if (rc != cudaSuccess) {
                SPDLOG_ERROR("cudaHostRegister failed: {}", cudaGetErrorString(rc));
                numa_free(ptr, size);
                return {};
            }
        } else {
            struct rlimit rl {};
            if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && size > rl.rlim_cur) {
                SPDLOG_WARN(
                    "RLIMIT_MEMLOCK too small: need ~{} bytes, soft limit {} "
                    "bytes. "
                    "Consider raising memlock ulimit.",
                    size,
                    (uint64_t)rl.rlim_cur);
            }

            TryMlock(ptr, size);
        }

        return {ptr, numa_node, size, true};
    }

    // Fallback non-NUMA: pageable or pinned (here choose pageable malloc)
    static AllocationResult AllocateAvoidNuma(size_t size) {
        void* ptr = std::malloc(size);
        if (ptr == nullptr) {
            return {nullptr, -1, size, false};
        }
        return {ptr, -1, size, true};
    }

    void Deallocate(void* ptr, size_t size) {
        if (ptr == nullptr) {
            return;
        }

        if (cuda_available_) {
            // Try to unregister; ignore "not registered" error
            cudaError_t rc = cudaHostUnregister(ptr);
            if (rc != cudaSuccess && rc != cudaErrorHostMemoryNotRegistered) {
                SPDLOG_WARN("cudaHostUnregister returned {}", cudaGetErrorString(rc));
                cudaGetLastError(); // clear sticky
            }
        } else {
            if (munlock(ptr, size) != 0) {
                SPDLOG_WARN("Try munlock failed under deallocate, but we still "
                            "continue");
            }
        }

        if (numa_available() != -1) {
            numa_free(ptr, size);
        } else {
            std::free(ptr);
        }
    }

    const std::vector<int>& GetAvailableNodes() const { return available_nodes_; }
    bool IsInitialized() const { return initialized_; }
    std::string GetNodeListString() const {
        std::string result = "[";
        for (size_t i = 0; i < available_nodes_.size(); ++i) {
            if (i > 0) {
                result += ", ";
            }
            result += std::to_string(available_nodes_[i]);
        }
        result += "]";
        return result;
    }
    int GetNumaNodeForCuda(int cuda_device_id) const {
        auto it = cuda_to_numa_.find(cuda_device_id);
        if (it != cuda_to_numa_.end()) {
            return it->second;
        }
        return -1;
    }

    void DetectCudaNumaTopology() {
        int cuda_count = 0;
        cudaError_t err = cudaGetDeviceCount(&cuda_count);
        if (err != cudaSuccess) {
            SPDLOG_WARN("cudaGetDeviceCount failed: {}", cudaGetErrorString(err));
            return;
        }
        SPDLOG_INFO("CUDA device count: {}", cuda_count);
        for (int cuda_id = 0; cuda_id < cuda_count; ++cuda_id) {
            std::string pci_bus_id(64, 0);
            cudaError_t err2 = cudaDeviceGetPCIBusId(pci_bus_id.data(), 64, cuda_id);
            if (err2 != cudaSuccess) {
                SPDLOG_WARN("CudaDeviceGetPCIBusId failed for device {}: {}", cuda_id, cudaGetErrorString(err2));
                continue;
            }
            std::string busid = ToLower(std::string(pci_bus_id)); // e.g., "0000:3b:00.0"
            SPDLOG_INFO("TOPOLOGY(Fixed): CUDA device {} PCI bus ID: {}", cuda_id, busid);

            // Prefer direct path
            std::vector<std::string> paths
                = {"/sys/bus/pci/devices/" + busid + "/numa_node",
                   "/sys/devices/pci0000:00/" + busid + "/numa_node",
                   "/sys/devices/pci0000:00/0000:" + busid + "/numa_node"};
            int numa_node = -1;
            for (const auto& p : paths) {
                std::ifstream f(p);
                if (!f.is_open()) {
                    continue;
                }
                std::string line;
                if (std::getline(f, line)) {
                    std::istringstream iss(line);
                    iss >> numa_node;
                    if (numa_node >= 0) {
                        SPDLOG_INFO(
                            "TOPOLOGY(Fixed): Found NUMA node {} for CUDA "
                            "device {} at path: {}",
                            numa_node,
                            cuda_id,
                            p);
                        break;
                    }
                }
            }
            if (numa_node < 0) {
                SPDLOG_WARN(
                    "TOPOLOGY(Fixed): Could not determine NUMA node for device "
                    "{}; defaulting to node 0",
                    cuda_id);
                numa_node = 0;
            }
            cuda_to_numa_[cuda_id] = numa_node;
        }
    }

 private:
    static void TryMlock(void* ptr, size_t size) {
        if (mlock(ptr, size) != 0) {
            SPDLOG_WARN("mlock failed: errno={}({})", errno, strerror(errno));
        }
    }

    std::vector<int> available_nodes_;
    std::atomic<size_t> current_node_index_{0};
    std::atomic<bool> initialized_{false};
    std::atomic_bool cuda_available_{true};
    mutable std::mutex init_mutex_;
    mutable std::mutex alloc_mutex_;
    std::unordered_map<int, int> cuda_to_numa_;
};

} // namespace astate
