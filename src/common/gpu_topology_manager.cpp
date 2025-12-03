#include "common/gpu_topology_manager.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <cuda_runtime.h>
#include <dirent.h>

#include <boost/algorithm/string/predicate.hpp>
#include <spdlog/spdlog.h>

namespace astate {

bool GpuTopologyManager::Initialize() {
    if (initialized_) {
        SPDLOG_INFO("GpuTopologyManager already initialized");
        return true;
    }

    SPDLOG_INFO("=== Starting GpuTopologyManager initialization ===");

    SPDLOG_INFO("Step 1: Attempting to parse nvidia-smi topo output");
    if (ParseNvidiaSmiTopo()) {
        SPDLOG_INFO("Step 1 SUCCESS: nvidia-smi topo parsed successfully");
        initialized_ = true;
        SPDLOG_INFO("Step 2: GpuTopologyManager initialization completed successfully");
        PrintTopology();
        SPDLOG_INFO("=== GpuTopologyManager initialization completed ===");
        return true;
    }

    SPDLOG_INFO("Step 1 FAILED: nvidia-smi topo failed, trying to get RDMA "
                "devices from system");
    if (GetRdmaDevicesFromSystem()) {
        SPDLOG_INFO("Step 1 SUCCESS: RDMA devices retrieved from system successfully");
        initialized_ = true;
        SPDLOG_INFO("Step 2: GpuTopologyManager initialization completed successfully");
        PrintTopology();
        SPDLOG_INFO("=== GpuTopologyManager initialization completed ===");
        return true;
    }

    SPDLOG_WARN("Step 1 FAILED: Both nvidia-smi and system API failed, will "
                "use fallback strategy");
    initialized_ = true;
    SPDLOG_INFO("Step 1 FALLBACK: Marked as initialized for fallback strategy");
    return true;
}

std::string GpuTopologyManager::SelectRdmaDevices(int cuda_device_id, int max_devices) {
    SPDLOG_INFO("=== Starting RDMA device selection for CUDA device ===");

    if (!initialized_) {
        SPDLOG_ERROR("Step 1 FAILED: GpuTopologyManager not initialized");
        return "";
    }
    SPDLOG_INFO("Step 1 SUCCESS: GpuTopologyManager is initialized");

    SPDLOG_INFO("Step 2: Checking CUDA device availability");
    int device_count = 0;
    cudaError_t cuda_result = cudaGetDeviceCount(&device_count);

    if (cuda_result != cudaSuccess || device_count == 0) {
        SPDLOG_WARN("Step 2 FAILED: No CUDA devices available, using "
                    "rank-based selection");
        return SelectRdmaDevicesByRank(0, max_devices);
    }
    SPDLOG_INFO("Step 2 SUCCESS: CUDA devices available");

    if (cuda_device_id < 0 || cuda_device_id >= device_count) {
        SPDLOG_ERROR("Step 3 FAILED: Invalid CUDA device ID: {}, device count: {}", cuda_device_id, device_count);
        return "";
    }
    SPDLOG_INFO("Step 3 SUCCESS: CUDA device ID {} is valid", cuda_device_id);

    std::string source_device = "GPU" + std::to_string(cuda_device_id);
    SPDLOG_INFO("Step 4: Source device name: {}", source_device);

    SPDLOG_INFO(
        "Step 5: Selecting RDMA devices for CUDA device {} (source: {}), max "
        "devices: {}",
        cuda_device_id,
        source_device,
        max_devices);

    SPDLOG_INFO("Step 6: Calling selectDevicesByDistance with distance priority");
    std::vector<std::string> selected_devices = SelectDevicesByDistance(source_device, max_devices, DISTANCE_PRIORITY);
    SPDLOG_INFO("Step 6.1: selectDevicesByDistance returned {} devices", selected_devices.size());

    if (selected_devices.empty()) {
        SPDLOG_WARN("Step 6 FAILED: No RDMA devices found for {}, trying fallback", source_device);
        return SelectDevicesFromNicList(max_devices);
    }
    SPDLOG_INFO("Step 6 SUCCESS: Found {} RDMA devices", selected_devices.size());

    SPDLOG_INFO("Step 7: Converting device names to NIC names");
    std::vector<std::string> nic_names;
    for (const auto& device : selected_devices) {
        std::string nic_name = DeviceNameToNicName(device);
        SPDLOG_INFO("Step 7.1: Converting {} -> {}", device, nic_name);
        if (!nic_name.empty()) {
            nic_names.push_back(nic_name);
        }
    }
    SPDLOG_INFO("Step 7 SUCCESS: Converted {} device names to NIC names", nic_names.size());

    SPDLOG_INFO("Step 8: Building final result string");
    std::string result;
    for (size_t i = 0; i < nic_names.size(); ++i) {
        if (i > 0) {
            result += ",";
        }
        result += nic_names[i];
    }

    SPDLOG_INFO("Step 8 SUCCESS: Final result: {}", result);
    SPDLOG_INFO("=== RDMA device selection completed ===");
    return result;
}

std::string GpuTopologyManager::SelectRdmaDevicesByRank(int rank_id, int max_devices) {
    SPDLOG_INFO("=== Starting RDMA device selection by rank ===");
    SPDLOG_INFO("DEBUG: SelectRdmaDevicesByRank called with rank_id={}, max_devices={}", rank_id, max_devices);

    if (!initialized_) {
        SPDLOG_ERROR("Step 1 FAILED: GpuTopologyManager not initialized");
        return "";
    }
    SPDLOG_INFO("Step 1 SUCCESS: GpuTopologyManager is initialized");

    SPDLOG_INFO("Step 2: Selecting RDMA devices by rank {}, max devices: {}", rank_id, max_devices);

    SPDLOG_INFO("Step 3: Getting available NIC device count");
    int nic_count = static_cast<int>(nic_devices_.size());
    SPDLOG_INFO("Step 3.1: Total NIC devices available: {}", nic_count);

    if (nic_count == 0) {
        SPDLOG_WARN("Step 3 FAILED: No NIC devices available, trying fallback");
        SPDLOG_INFO("Step 3.2: Calling SelectDevicesFromNicList with max_devices={}", max_devices);
        std::string result = SelectDevicesFromNicList(max_devices);
        SPDLOG_INFO("Step 3.3: SelectDevicesFromNicList returned: '{}'", result);
        return result;
    }
    SPDLOG_INFO("Step 3 SUCCESS: NIC devices are available");

    SPDLOG_INFO("Step 4: Calculating first NIC index using hash");
    int first_nic_index = rank_id % nic_count;
    SPDLOG_INFO("Step 4.1: rank_id: {}, nic_count: {}, first_nic_index: {}", rank_id, nic_count, first_nic_index);

    SPDLOG_INFO("Step 5: Selecting NIC devices");
    std::vector<std::string> selected_nic_names;
    for (int i = 0; i < max_devices && i < nic_count; ++i) {
        int nic_index = (first_nic_index + i) % nic_count;
        const auto& nic_device = nic_devices_[nic_index];
        SPDLOG_INFO(
            "Step 5.{}: Selecting NIC index {} (device: {}, nic_name: {})",
            (i + 1),
            nic_index,
            nic_device.name,
            nic_device.nic_name);
        if (!nic_device.nic_name.empty()) {
            selected_nic_names.push_back(nic_device.nic_name);
        }
    }
    SPDLOG_INFO("Step 5 SUCCESS: Selected {} NIC devices", selected_nic_names.size());

    SPDLOG_INFO("Step 6: Building final result string");
    std::string result;
    for (size_t i = 0; i < selected_nic_names.size(); ++i) {
        if (i > 0) {
            result += ",";
        }
        result += selected_nic_names[i];
    }

    SPDLOG_INFO("Step 6 SUCCESS: Final result: {}", result);
    SPDLOG_INFO("=== RDMA device selection by rank completed ===");
    return result;
}

bool GpuTopologyManager::GetRdmaDevicesFromSystem() {
    SPDLOG_INFO("  Getting RDMA devices from system (non-GPU environment)");

    if (!GetNetworkDevicesFromSystem()) {
        SPDLOG_ERROR("  FAILED: Failed to get network devices from system");
        return false;
    }

    SPDLOG_INFO("  RDMA devices retrieved from system: {} devices", nic_devices_.size());
    return !nic_devices_.empty();
}

bool GpuTopologyManager::GetNetworkDevicesFromSystem() {
    SPDLOG_INFO("  Getting network devices from system");

    FILE* pipe = popen("lspci | grep -i network", "r");
    if (pipe == nullptr) {
        SPDLOG_ERROR("  FAILED: Failed to execute lspci command");
        return false;
    }

    char buffer[256];
    int nic_index = 0;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);
        SPDLOG_INFO("  Found network device: {}", line.substr(0, line.length() - 1));

        // 00:01.0 Network controller: Mellanox Technologies MT27800 Family [ConnectX-5]
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string pci_addr = line.substr(0, colon_pos);
            SPDLOG_INFO("  PCI address: {}", pci_addr);

            DeviceInfo nic_info;
            nic_info.name = "NIC" + std::to_string(nic_index);
            nic_info.type = DeviceType::NIC;
            nic_info.numa_node = -1;

            std::string nic_name = GetNicNameFromPci(pci_addr);
            if (!nic_name.empty()) {
                nic_info.nic_name = nic_name;
                nic_name_mapping_[nic_info.name] = nic_name;
                SPDLOG_INFO("  NIC {} -> {}", nic_index, nic_name);
            } else {
                nic_info.nic_name = "mlx5_bond_" + std::to_string(nic_index);
                nic_name_mapping_[nic_info.name] = nic_info.nic_name;
                SPDLOG_INFO("  NIC {} -> {} (default)", nic_index, nic_info.nic_name);
            }

            int numa_node = GetNumaNodeFromPci(pci_addr);
            if (numa_node >= 0) {
                nic_info.numa_node = numa_node;
                SPDLOG_INFO("  NIC {} NUMA node: {}", nic_index, numa_node);
            }

            nic_devices_.push_back(nic_info);
            nic_index++;
        }
    }

    pclose(pipe);

    SPDLOG_INFO("  Network devices collection completed: {} devices found", nic_devices_.size());
    return !nic_devices_.empty();
}

std::string GpuTopologyManager::GetNicNameFromPci(const std::string& pci_addr) {
    std::string sysfs_path = "/sys/bus/pci/devices/" + pci_addr + "/net";

    DIR* dir = opendir(sysfs_path.c_str());
    if (dir != nullptr) {
        struct dirent* entry = nullptr;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                closedir(dir);
                return entry->d_name;
            }
        }
        closedir(dir);
    }

    return "";
}

int GpuTopologyManager::GetNumaNodeFromPci(const std::string& pci_addr) {
    std::vector<std::string> pci_variants = {pci_addr};

    if (std::any_of(pci_addr.begin(), pci_addr.end(), ::islower)) {
        std::string upper_pci = pci_addr;
        std::transform(upper_pci.begin(), upper_pci.end(), upper_pci.begin(), ::toupper);
        pci_variants.push_back(upper_pci);
    }

    if (std::any_of(pci_addr.begin(), pci_addr.end(), ::isupper)) {
        std::string lower_pci = pci_addr;
        std::transform(lower_pci.begin(), lower_pci.end(), lower_pci.begin(), ::tolower);
        pci_variants.push_back(lower_pci);
    }

    for (const auto& variant : pci_variants) {
        std::string numa_path = "/sys/bus/pci/devices/" + variant + "/numa_node";
        SPDLOG_INFO("  Trying NUMA path: {}", numa_path);

        std::ifstream file(numa_path);
        if (file.is_open()) {
            int numa_node = 0;
            if (file >> numa_node) {
                SPDLOG_INFO("  SUCCESS: Found NUMA node {} for PCI {}", numa_node, variant);
                return numa_node;
            }
            file.close();
        }
    }

    SPDLOG_WARN("  FAILED: No NUMA node found for PCI {} (tried {} variants)", pci_addr, pci_variants.size());
    return -1;
}

std::string GpuTopologyManager::SelectDevicesFromNicList(int max_devices) {
    SPDLOG_INFO("  FALLBACK: Selecting devices from NIC list - ENTERED FUNCTION");

    if (nic_devices_.empty()) {
        SPDLOG_INFO("  NIC list is empty, trying to get devices from system");
        if (!GetNetworkDevicesFromSystem()) {
            SPDLOG_WARN("  FAILED: Could not get devices from system, using "
                        "default device names");
            std::string result;
            for (int i = 0; i < max_devices; ++i) {
                if (i > 0) {
                    result += ",";
                }
                result += "mlx5_bond_" + std::to_string(i);
            }
            SPDLOG_INFO("  FALLBACK result (default): {}", result);
            return result;
        }
    }

    int nic_count = static_cast<int>(nic_devices_.size());
    if (nic_count == 0) {
        SPDLOG_WARN("  FAILED: No NIC devices available even after system "
                    "query, using default device names");
        std::string result;
        for (int i = 0; i < max_devices; ++i) {
            if (i > 0) {
                result += ",";
            }
            result += "mlx5_bond_" + std::to_string(i);
        }
        SPDLOG_INFO("  FALLBACK result (default): {}", result);
        return result;
    }

    SPDLOG_INFO("  Found {} NIC devices for fallback selection", nic_count);

    std::vector<std::string> selected_nic_names;
    int devices_to_select = std::min(max_devices, nic_count);

    for (int i = 0; i < devices_to_select; ++i) {
        const auto& nic_device = nic_devices_[i];
        SPDLOG_INFO("  Selecting NIC {}: {} -> {}", i, nic_device.name, nic_device.nic_name);
        if (!nic_device.nic_name.empty()) {
            selected_nic_names.push_back(nic_device.nic_name);
        }
    }

    std::string result;
    for (size_t i = 0; i < selected_nic_names.size(); ++i) {
        if (i > 0) {
            result += ",";
        }
        result += selected_nic_names[i];
    }

    SPDLOG_INFO("  FALLBACK result: {}", result);
    return result;
}

bool GpuTopologyManager::ParseNvidiaSmiTopo() {
    SPDLOG_INFO("=== Starting nvidia-smi topo parsing ===");

    SPDLOG_INFO("Step 1: Executing nvidia-smi topo command");
    std::string topo_output = ExecuteNvidiaSmiTopo();
    if (topo_output.empty()) {
        SPDLOG_WARN("Step 1 FAILED: Failed to execute nvidia-smi topo command");
        return false;
    }
    SPDLOG_INFO("Step 1 SUCCESS: nvidia-smi topo command executed, output length: {}", topo_output.length());

    SPDLOG_INFO("Step 2: Splitting output into lines");
    std::istringstream iss(topo_output);
    std::vector<std::string> lines;
    std::string line;
    int line_count = 0;
    while (std::getline(iss, line)) {
        if (!line.empty()) {
            lines.push_back(line);
            line_count++;
        }
    }
    SPDLOG_INFO("Step 2 SUCCESS: Split into {} non-empty lines", line_count);

    if (lines.empty()) {
        SPDLOG_WARN("Step 2 FAILED: Empty output from nvidia-smi topo");
        return false;
    }

    SPDLOG_INFO("Step 3: Parsing topology matrix");
    if (!ParseTopoMatrix(lines)) {
        SPDLOG_WARN("Step 3 FAILED: Failed to parse topology matrix");
        return false;
    }
    SPDLOG_INFO("Step 3 SUCCESS: Topology matrix parsed successfully");

    SPDLOG_INFO("Step 4: Parsing NIC legend");
    if (!ParseNicLegend(lines)) {
        SPDLOG_WARN("Step 4 FAILED: Failed to parse NIC legend");
        return false;
    }
    SPDLOG_INFO("Step 4 SUCCESS: NIC legend parsed successfully");

    SPDLOG_INFO("=== nvidia-smi topo parsing completed successfully ===");
    return true;
}

std::string GpuTopologyManager::ExecuteNvidiaSmiTopo() {
    SPDLOG_INFO("Step 1.1: Opening pipe to nvidia-smi topo -m");
    FILE* pipe = popen("nvidia-smi topo -m", "r");
    if (pipe == nullptr) {
        SPDLOG_ERROR("Step 1.1 FAILED: Failed to execute nvidia-smi topo "
                     "command (popen failed)");
        return "";
    }
    SPDLOG_INFO("Step 1.1 SUCCESS: Pipe opened successfully");

    SPDLOG_INFO("Step 1.2: Reading command output");
    std::string result;
    char buffer[128];
    int read_count = 0;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
        read_count++;
    }
    SPDLOG_INFO("Step 1.2 SUCCESS: Read {} buffer chunks, total size: {}", read_count, result.length());

    SPDLOG_INFO("Step 1.3: Closing pipe");
    int status = pclose(pipe);
    if (status != 0) {
        SPDLOG_WARN("Step 1.3 WARNING: nvidia-smi topo command exited with status {}", status);
    } else {
        SPDLOG_INFO("Step 1.3 SUCCESS: Pipe closed successfully");
    }

    return result;
}

bool GpuTopologyManager::ParseTopoMatrix(const std::vector<std::string>& lines) {
    SPDLOG_INFO("Step 3.1: Starting topology matrix parsing");
    bool found_matrix = false;
    bool found_legend = false;
    int processed_lines = 0;
    int device_lines = 0;

    SPDLOG_INFO("Step 3.2: Processing {} lines", lines.size());
    for (const auto& line : lines) {
        processed_lines++;

        if (line.empty() || line.find("Legend:") != std::string::npos) {
            SPDLOG_INFO("Step 3.2.{}: Found legend marker, stopping matrix parsing", processed_lines);
            found_legend = true;
            continue;
        }

        if (found_legend) {
            SPDLOG_INFO("Step 3.2.{}: Skipping line after legend", processed_lines);
            break;
        }

        if (line.find("GPU") != std::string::npos || line.find("NIC") != std::string::npos) {
            SPDLOG_INFO("Step 3.2.{}: Processing device line: {}...", processed_lines, line.substr(0, 50));
            if (!ParseDeviceInfoLine(line)) {
                SPDLOG_WARN("Step 3.2.{} FAILED: Failed to parse device info line: {}", processed_lines, line);
            } else {
                device_lines++;
                SPDLOG_INFO("Step 3.2.{} SUCCESS: Device line parsed successfully", processed_lines);
            }
            found_matrix = true;
        } else {
            SPDLOG_INFO("Step 3.2.{}: Skipping non-device line", processed_lines);
        }
    }

    SPDLOG_INFO(
        "Step 3.3: Matrix parsing summary - processed lines: {}, device lines: "
        "{}, found_matrix: {}",
        processed_lines,
        device_lines,
        found_matrix);
    return found_matrix;
}

bool GpuTopologyManager::ParseDeviceInfoLine(const std::string& line) {
    SPDLOG_INFO("  Parsing device info line: {}...", line.substr(0, 80));

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;

    SPDLOG_INFO("  Step 1: Tokenizing line");
    while (iss >> token) {
        tokens.push_back(token);
    }
    SPDLOG_INFO("  Step 1 SUCCESS: Found {} tokens", tokens.size());

    if (tokens.size() < 3) {
        SPDLOG_WARN("  Step 1 FAILED: Too few tokens ({})", tokens.size());
        return false;
    }

    std::string device_name = tokens[0];
    SPDLOG_INFO("  Step 2: Device name: {}", device_name);

    DeviceType device_type{};
    if (boost::algorithm::starts_with(device_name, "GPU")) {
        device_type = DeviceType::GPU;
        SPDLOG_INFO("  Step 2 SUCCESS: Device type is GPU");
    } else if (boost::algorithm::starts_with(device_name, "NIC")) {
        device_type = DeviceType::NIC;
        SPDLOG_INFO("  Step 2 SUCCESS: Device type is NIC");
    } else {
        SPDLOG_WARN("  Step 2 FAILED: Unknown device type: {}", device_name);
        return false;
    }

    SPDLOG_INFO("  Step 3: Creating device info");
    DeviceInfo device_info;
    device_info.name = device_name;
    device_info.type = device_type;
    device_info.numa_node = -1;

    SPDLOG_INFO("  Step 4: Parsing CPU affinity and NUMA info");
    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "CPU" && i + 1 < tokens.size()) {
            std::string cpu_affinity = tokens[i + 1];
            SPDLOG_INFO("  Step 4.1: Found CPU affinity: {}", cpu_affinity);
        } else if (tokens[i] == "NUMA" && i + 1 < tokens.size()) {
            try {
                device_info.numa_node = std::stoi(tokens[i + 1]);
                SPDLOG_INFO("  Step 4.2: Found NUMA node: {}", device_info.numa_node);
            } catch (const std::exception& e) {
                SPDLOG_WARN("  Step 4.2 FAILED: Failed to parse NUMA node: {}", tokens[i + 1]);
            }
        }
    }

    SPDLOG_INFO("  Step 5: Adding device to list");
    if (device_type == DeviceType::GPU) {
        gpu_devices_.push_back(device_info);
        SPDLOG_INFO("  Step 5 SUCCESS: Added GPU device, total GPUs: {}", gpu_devices_.size());
    } else {
        nic_devices_.push_back(device_info);
        SPDLOG_INFO("  Step 5 SUCCESS: Added NIC device, total NICs: {}", nic_devices_.size());
    }

    SPDLOG_INFO("  Step 6: Parsing topology relations");
    int relations_added = 0;
    for (size_t i = 1; i < tokens.size(); ++i) {
        std::string target_device;
        if (i - 1 < gpu_devices_.size()) {
            target_device = gpu_devices_[i - 1].name;
        } else if (i - 1 - gpu_devices_.size() < nic_devices_.size()) {
            target_device = nic_devices_[i - 1 - gpu_devices_.size()].name;
        } else {
            continue;
        }

        TopoDistance distance = StringToDistance(tokens[i]);
        if (distance != TopoDistance::UNKNOWN) {
            TopoRelation relation;
            relation.from_device = device_name;
            relation.to_device = target_device;
            relation.distance = distance;
            topo_relations_.push_back(relation);
            relations_added++;
            SPDLOG_INFO(
                "  Step 6.{}: Added relation {} -> {} (distance: {})",
                relations_added,
                device_name,
                target_device,
                static_cast<int>(distance));
        }
    }
    SPDLOG_INFO("  Step 6 SUCCESS: Added {} topology relations", relations_added);

    SPDLOG_INFO("  Device info line parsing completed successfully");
    return true;
}

bool GpuTopologyManager::ParseNicLegend(const std::vector<std::string>& lines) {
    SPDLOG_INFO("Step 4.1: Starting NIC legend parsing");
    bool found_legend = false;
    int mappings_found = 0;

    SPDLOG_INFO("Step 4.2: Searching for NIC Legend section");
    for (const auto& line : lines) {
        if (line.find("NIC Legend:") != std::string::npos) {
            SPDLOG_INFO("Step 4.2 SUCCESS: Found NIC Legend marker");
            found_legend = true;
            continue;
        }

        if (found_legend) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string nic_name = line.substr(0, colon_pos);
                std::string actual_name = line.substr(colon_pos + 1);

                actual_name.erase(0, actual_name.find_first_not_of(" \t"));
                actual_name.erase(actual_name.find_last_not_of(" \t") + 1);

                if (!nic_name.empty() && !actual_name.empty()) {
                    nic_name_mapping_[nic_name] = actual_name;
                    mappings_found++;
                    SPDLOG_INFO("Step 4.3.{}: NIC mapping: {} -> {}", mappings_found, nic_name, actual_name);
                } else {
                    SPDLOG_WARN("Step 4.3: Skipping invalid mapping line: {}", line);
                }
            } else if (!line.empty() && line.find("Legend:") == std::string::npos) {
                SPDLOG_INFO("Step 4.3: Skipping non-mapping line: {}", line);
            }
        }
    }

    SPDLOG_INFO("Step 4.4: NIC legend parsing summary - mappings found: {}", mappings_found);
    if (!nic_name_mapping_.empty()) {
        SPDLOG_INFO("Step 4.4 SUCCESS: NIC legend parsed successfully");
        return true;
    }
    SPDLOG_WARN("Step 4.4 FAILED: No NIC mappings found");
    return false;
}

TopoDistance GpuTopologyManager::StringToDistance(const std::string& distance_str) {
    if (distance_str == "X") {
        return TopoDistance::SELF;
    }
    if (distance_str == "PIX") {
        return TopoDistance::PIX;
    }
    if (distance_str == "NODE") {
        return TopoDistance::NODE;
    }
    if (distance_str == "SYS") {
        return TopoDistance::SYS;
    }
    if (boost::algorithm::starts_with(distance_str, "NV")) {
        return TopoDistance::NV18;
    }

    return TopoDistance::UNKNOWN;
}

TopoDistance GpuTopologyManager::GetTopoDistance(const std::string& from_device, const std::string& to_device) const {
    for (const auto& relation : topo_relations_) {
        if (relation.from_device == from_device && relation.to_device == to_device) {
            return relation.distance;
        }
    }

    for (const auto& relation : topo_relations_) {
        if (relation.from_device == to_device && relation.to_device == from_device) {
            return relation.distance;
        }
    }

    return TopoDistance::UNKNOWN;
}

std::vector<std::string> GpuTopologyManager::SelectDevicesByDistance(
    const std::string& source_device, int max_devices, const std::vector<TopoDistance>& distance_priority) {
    SPDLOG_INFO("    Starting device selection by distance for {}, max devices: {}", source_device, max_devices);

    std::vector<std::string> selected_devices;

    std::unordered_map<TopoDistance, std::vector<std::string>> devices_by_distance;

    SPDLOG_INFO("    Step 1: Grouping devices by distance");
    for (const auto& nic_device : nic_devices_) {
        TopoDistance distance = GetTopoDistance(source_device, nic_device.name);
        SPDLOG_INFO(
            "    Device {} has distance {} from {}", nic_device.name, static_cast<int>(distance), source_device);

        if (distance != TopoDistance::UNKNOWN) {
            devices_by_distance[distance].push_back(nic_device.name);
        }
    }

    for (size_t priority_idx = 0; priority_idx < distance_priority.size(); ++priority_idx) {
        TopoDistance target_distance = distance_priority[priority_idx];
        SPDLOG_INFO(
            "    Step {}: Processing distance {} (priority {})",
            (priority_idx + 2),
            static_cast<int>(target_distance),
            (priority_idx + 1));

        if (selected_devices.size() >= static_cast<size_t>(max_devices)) {
            SPDLOG_INFO(
                "    Step {} SKIP: Already selected {} devices, max is {}",
                (priority_idx + 2),
                selected_devices.size(),
                max_devices);
            break;
        }

        auto it = devices_by_distance.find(target_distance);
        if (it == devices_by_distance.end()) {
            SPDLOG_INFO(
                "    Step {} SKIP: No devices with distance {}", (priority_idx + 2), static_cast<int>(target_distance));
            continue;
        }

        std::vector<std::string>& available_devices = it->second;
        SPDLOG_INFO(
            "    Step {}: Found {} devices with distance {}",
            (priority_idx + 2),
            available_devices.size(),
            static_cast<int>(target_distance));

        std::vector<std::string> unselected_devices;
        for (const auto& device : available_devices) {
            if (std::find(selected_devices.begin(), selected_devices.end(), device) == selected_devices.end()) {
                unselected_devices.push_back(device);
            }
        }

        if (unselected_devices.empty()) {
            SPDLOG_INFO(
                "    Step {} SKIP: All devices with distance {} already "
                "selected",
                (priority_idx + 2),
                static_cast<int>(target_distance));
            continue;
        }

        SPDLOG_INFO("    Step {}: {} unselected devices available", (priority_idx + 2), unselected_devices.size());

        std::random_device random_device;
        std::mt19937 gen(random_device());
        std::shuffle(unselected_devices.begin(), unselected_devices.end(), gen);

        int devices_to_select = std::min(
            static_cast<int>(unselected_devices.size()), max_devices - static_cast<int>(selected_devices.size()));

        for (int i = 0; i < devices_to_select; ++i) {
            selected_devices.push_back(unselected_devices[i]);
            SPDLOG_INFO(
                "    Step {}.{} SUCCESS: Selected {} with distance {} from {}",
                (priority_idx + 2),
                (i + 1),
                unselected_devices[i],
                static_cast<int>(target_distance),
                source_device);
        }

        SPDLOG_INFO(
            "    Step {}: Distance {} processing completed", (priority_idx + 2), static_cast<int>(target_distance));
    }

    SPDLOG_INFO("    Device selection completed: selected {} devices", selected_devices.size());
    return selected_devices;
}

std::string GpuTopologyManager::DeviceNameToNicName(const std::string& device_name) const {
    auto iter = nic_name_mapping_.find(device_name);
    if (iter != nic_name_mapping_.end()) {
        return iter->second;
    }

    if (boost::algorithm::starts_with(device_name, "NIC")) {
        std::string nic_index = device_name.substr(3);
        return "mlx5_bond_" + nic_index;
    }

    return "";
}

bool GpuTopologyManager::IsValidDeviceName(const std::string& device_name) {
    return boost::algorithm::starts_with(device_name, "GPU") || boost::algorithm::starts_with(device_name, "NIC");
}

void GpuTopologyManager::PrintTopology() const {
    SPDLOG_INFO("=== GPU Topology Information ===");
    SPDLOG_INFO("GPU Devices ({}):", gpu_devices_.size());
    for (const auto& gpu : gpu_devices_) {
        SPDLOG_INFO("  {} (NUMA: {})", gpu.name, gpu.numa_node);
    }

    SPDLOG_INFO("NIC Devices ({}):", nic_devices_.size());
    for (const auto& nic : nic_devices_) {
        std::string actual_name = DeviceNameToNicName(nic.name);
        SPDLOG_INFO("  {} -> {} (NUMA: {})", nic.name, actual_name, nic.numa_node);
    }

    SPDLOG_INFO("Topology Relations ({}):", topo_relations_.size());
    for (const auto& relation : topo_relations_) {
        SPDLOG_INFO(
            "  {} -> {} (distance: {})", relation.from_device, relation.to_device, static_cast<int>(relation.distance));
    }
    SPDLOG_INFO("=== End GPU Topology Information ===");
}

} // namespace astate
