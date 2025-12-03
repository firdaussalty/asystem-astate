#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace astate {

// Topology distance types
enum class TopoDistance : uint8_t {
    SELF, // Same device
    PIX, // Same PCIe switch
    NODE, // Same NUMA node
    SYS, // Cross NUMA nodes
    NV18, // NVLink connected
    UNKNOWN // Unknown distance
};

// Device types
enum class DeviceType : uint8_t { GPU, NIC };

// Device information
struct DeviceInfo {
    std::string name; // Device name (GPU0, GPU1, etc.)
    std::string nic_name; // Actual NIC name (mlx5_bond_0, etc.)
    DeviceType type; // Device type
    int numa_node; // NUMA node ID
    std::vector<int> cpu_affinity; // CPU affinity
};

// Topology relationship
struct TopoRelation {
    std::string from_device; // Source device
    std::string to_device; // Target device
    TopoDistance distance; // Topology distance
};

// GPU topology manager for device selection and topology analysis
class GpuTopologyManager {
 public:
    GpuTopologyManager() = default;
    ~GpuTopologyManager() = default;

    // Disable copy
    GpuTopologyManager(const GpuTopologyManager&) = delete;
    GpuTopologyManager& operator=(const GpuTopologyManager&) = delete;

    // Initialize topology information
    bool Initialize();

    // Select optimal RDMA devices for given CUDA device
    std::string SelectRdmaDevices(int cuda_device_id, int max_devices = 2);

    // Select RDMA devices by rank ID (GPU-less mode)
    std::string SelectRdmaDevicesByRank(int rank_id, int max_devices = 2);

    // Getters
    const std::vector<DeviceInfo>& GetGpuDevices() const { return gpu_devices_; }
    const std::vector<DeviceInfo>& GetNicDevices() const { return nic_devices_; }
    const std::vector<TopoRelation>& GetTopoRelations() const { return topo_relations_; }
    bool IsInitialized() const { return initialized_; }

    // Debug: print topology information
    void PrintTopology() const;

 private:
    // Parse nvidia-smi topology output
    bool ParseNvidiaSmiTopo();

    // Execute nvidia-smi topo command
    static std::string ExecuteNvidiaSmiTopo();

    // Get RDMA devices from system (non-GPU environment)
    bool GetRdmaDevicesFromSystem();

    // Get network devices from system
    bool GetNetworkDevicesFromSystem();

    // Get NIC name from PCI address
    static std::string GetNicNameFromPci(const std::string& pci_addr);

    // Get NUMA node from PCI address
    static int GetNumaNodeFromPci(const std::string& pci_addr);

    // Fallback device selection from NIC list
    std::string SelectDevicesFromNicList(int max_devices);

    // Parse topology matrix
    bool ParseTopoMatrix(const std::vector<std::string>& lines);

    // Parse device information line
    bool ParseDeviceInfoLine(const std::string& line);

    // Parse NIC legend
    bool ParseNicLegend(const std::vector<std::string>& lines);

    // Convert distance string to enum
    static TopoDistance StringToDistance(const std::string& distance_str);

    // Get topology distance between devices
    TopoDistance GetTopoDistance(const std::string& from_device, const std::string& to_device) const;

    // Select devices by distance priority
    std::vector<std::string> SelectDevicesByDistance(
        const std::string& source_device, int max_devices, const std::vector<TopoDistance>& distance_priority);

    // Convert device name to actual NIC name
    std::string DeviceNameToNicName(const std::string& device_name) const;

    // Validate device name format
    static bool IsValidDeviceName(const std::string& device_name);

    std::vector<DeviceInfo> gpu_devices_; // GPU devices
    std::vector<DeviceInfo> nic_devices_; // NIC devices
    std::vector<TopoRelation> topo_relations_; // Topology relations
    std::unordered_map<std::string, std::string> nic_name_mapping_; // NIC name mapping

    bool initialized_{false}; // Initialization status

    // Distance priority (nearest to farthest)
    static const std::vector<TopoDistance> DISTANCE_PRIORITY;
};

// Distance priority definition
inline const std::vector<TopoDistance> GpuTopologyManager::DISTANCE_PRIORITY = {
    TopoDistance::PIX, // Nearest
    TopoDistance::NODE, // Medium
    TopoDistance::SYS // Farthest
};

} // namespace astate
