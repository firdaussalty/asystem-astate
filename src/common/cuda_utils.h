#pragma once

#include <sstream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <numa.h>
#include <numaif.h>

#include <spdlog/spdlog.h>

namespace astate {

inline size_t GetCudaMemoryUsageMb() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        return 0;
    }

    size_t total_memory = 0;
    size_t free_memory = 0;
    auto err = cudaMemGetInfo(&free_memory, &total_memory);
    if (err != cudaSuccess) {
        return 0;
    }

    return (total_memory - free_memory) / 1024 / 1024;
}

inline bool HasNvGpu() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        return false;
    }
    return device_count > 0;
}

inline std::string CpuMaskStr() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    sched_getaffinity(0, sizeof(mask), &mask);
    std::ostringstream oss;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) {
            oss << i << ",";
        }
    }
    return oss.str();
}

inline std::string MemPolicyStr() {
    int policy = 0;
    uint64_t maxnode = 8 * sizeof(uint64_t);
    std::vector<uint64_t> nodemask((numa_max_node() + sizeof(uint64_t) * 8) / (sizeof(uint64_t) * 8), 0);
    if (get_mempolicy(&policy, nodemask.data(), maxnode, nullptr, 0) == 0) {
        // MPOL_DEFAULT/INTERLEAVE/BIND/PREFERRED
        return "policy=" + std::to_string(policy);
    }
    return "policy=?";
}

} // namespace astate
