#pragma once

#include <any>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>

#include "common/string_utils.h"

namespace astate {

// Options is a map of config items <config_name, config_value>
using Options = std::unordered_map<std::string, std::string>;

inline std::ostream& operator<<(std::ostream& os, const Options& options) {
    os << "{";
    for (auto it = options.begin(); it != options.end(); ++it) {
        if (it != options.begin()) {
            os << ", ";
        }
        os << it->first << ": " << it->second;
    }
    os << "}";
    return os;
}

inline std::string ToString(const Options& options) {
    std::ostringstream oss;
    oss << options;
    return oss.str();
}

// Option definition: config name, config value type, default value.....
using ValueType = enum : uint8_t {
    INT = 0,
    STRING = 1,
    STRING_LIST = 2,
    BOOL = 3,
    FLOAT = 4,
    DOUBLE = 5,
    INT64 = 6,
    UNKNOWN = 255
};

using OptionDef = struct {
    ValueType value_type;
    std::string default_value;
};

using OptionDefinition = std::unordered_map<std::string, OptionDef>;

inline OptionDefinition& GetOptionDefinitions() {
    static OptionDefinition defs;
    return defs;
}

inline std::mutex& GetOptionDefinitionsMutex() {
    static std::mutex m;
    return m;
}

struct CreatOption {
    CreatOption(const std::string& name, ValueType type, const std::string& default_value) {
        std::lock_guard<std::mutex> lock(GetOptionDefinitionsMutex());
        auto& defs = GetOptionDefinitions();

        auto it = defs.find(name);
        if (it == defs.end()) {
            defs.emplace(name, OptionDef{type, default_value});
        } else {
            const auto& old = it->second;
            if (old.value_type != type || old.default_value != default_value) {
                SPDLOG_ERROR(
                    "Option {} already defined with different definition: "
                    "old(type={}, default={}), new(type={}, default={})",
                    name,
                    static_cast<int>(old.value_type),
                    old.default_value,
                    static_cast<int>(type),
                    default_value);
            } else {
                SPDLOG_WARN("Option {} already defined with identical definition, ignoring duplicate", name);
            }
        }
    }
};

#define OPTION(name, type, default_value) \
    inline const std::string name = #name; \
    inline const ::astate::CreatOption option_reg_##name(name, type, default_value);

/********** Option definition: [Option Name, Option Type, Default Value] ***********/
OPTION(ASTATE_OPTIONS_LOAD_MODE, STRING, "ENV") // ENV, FILE...
OPTION(ASTATE_OPTIONS_FILE_PATH, STRING, "") // Config file path
OPTION(TENSOR_TRANSFER_SERVICE_TYPE, STRING, "PULL") // PUSH, PULL, AYSNC...
OPTION(ASTATE_DEBUG_MODE, BOOL, "false")

// Transfer Engine Global Options
OPTION(TRANSFER_ENGINE_META_SERVICE_ADDRESS, STRING, "")
OPTION(TRANSFER_ENGINE_PEERS_HOST, STRING_LIST, "")
OPTION(TRANSFER_ENGINE_GROUP_HOST, STRING_LIST, "")

// Transfer Engine Data Transport Options
OPTION(TRANSFER_ENGINE_TYPE, STRING, "")
OPTION(TRANSFER_ENGINE_LOCAL_ADDRESS, STRING, "")
OPTION(TRANSFER_ENGINE_LOCAL_PORT, INT, "0")
OPTION(TRANSFER_ENGINE_READ_TIMEOUT_MS, INT, "120000") // 120s
OPTION(TRANSFER_ENGINE_WRITE_TIMEOUT_MS, INT, "120000") // 120s
OPTION(TRANSFER_ENGINE_READ_THREAD_NUM, INT, "32")
OPTION(TRANSFER_ENGINE_COPY_THREAD_NUM, INT, "32")
OPTION(TRANSFER_ENGINE_COPY_BUCKET_MEM_SIZE, INT64, "524288000") // 500MB
OPTION(TRANSFER_ENGINE_COPY_LARGE_THREAD_NUM, INT, "2")
OPTION(TRANSFER_ENGINE_COPY_LARGE_BUCKET_MEM_SIZE, INT64, "3145728000") // 3000MB
OPTION(TRANSFER_ENGINE_COPY_SMALL_THREAD_NUM, INT, "8")
OPTION(TRANSFER_ENGINE_SMALL_TENSOR_COMPACT_CACHE_SIZE, INT64, "2097152") // 2M
OPTION(TRANSFER_ENGINE_SMALL_TENSOR_SIZE, INT64, "524288") // 512KB
OPTION(TRANSFER_ENGINE_ENABLE_PERF_METRICS, BOOL, "false")
OPTION(TRANSFER_ENGINE_PERF_STATS_INTERVAL_MS, INT64, "200") // 200ms
OPTION(TRANSFER_ENGINE_LOCAL_CACHE_TENSORS_SIZE, INT64, "20971520000") // 20GB
OPTION(TRANSFER_ENGINE_ENABLE_LOCAL_CACHE_PREFETCH, BOOL, "false")
OPTION(TRANSFER_ENGINE_ENABLE_WRITE_GPU_ASYNC_COPY, BOOL, "false")
OPTION(TRANSFER_ENGINE_ENABLE_READ_GPU_ASYNC_COPY, BOOL, "false")
OPTION(TRANSFER_ENGINE_SAMPLE_RATE, INT, "100") // 1/100 requests for perf metrics
OPTION(TRANSFER_ENGINE_SAMPLE_STEP, INT, "-2") // sample Nth step

OPTION(TRANSFER_ENGINE_TRAINING_PARALLEL_CONFIG, STRING_LIST,
       "") // Training ParallelConfig, e.g., "dp=4,pp=2,tp=2"
OPTION(TRANSFER_ENGINE_INFERENCE_PARALLEL_CONFIG, STRING_LIST,
       "") // Inference ParallelConfig
OPTION(TRANSFER_ENGINE_TENSOR_RESHARDING_KEYS, STRING_LIST,
       "") // e.g., "down_proj,gate_proj,up_proj"

// Transfer Engine Control Service Options
OPTION(TRANSFER_ENGINE_SERVICE_TYPE, STRING, "")
OPTION(TRANSFER_ENGINE_SERVICE_ADDRESS, STRING, "")
OPTION(TRANSFER_ENGINE_SERVICE_PORT, INT, "0")
OPTION(TRANSFER_ENGINE_SERVICE_TENSOR_READY_TIMEOUT_MS, INT, "1200000") // 1200s
OPTION(TRANSFER_ENGINE_SERVICE_SKIP_DISCOVERY, BOOL, "false")
OPTION(TRANSFER_ENGINE_SERVICE_FIXED_PORT, BOOL, "false")
OPTION(DISCOVERY_WORLD_SIZE_RETRY_COUNT, INT,
       "270") // sleep interval 10s, total 2700s
OPTION(DISCOVERY_CONFIG_CENTER_TYPE, STRING, "FILE") // TCPStore, HTTP, FILE
OPTION(TRANSFER_ENGINE_LOG_TENSOR_META, BOOL, "true")

// skip rdma exception for test environment when rdma not working
OPTION(TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION, BOOL, "false")
OPTION(TRANSFER_ENGINE_MAX_RDMA_DEVICES, INT, "2")
OPTION(TRANSFER_ENGINE_RDMA_NUM_POLLERS, INT, "1")
OPTION(TRANSPORT_RECEIVE_RETRY_COUNT, INT, "30")
OPTION(TRANSPORT_RECEIVE_RETRY_SLEEP_MS, INT, "3000")
OPTION(TRANSPORT_SEND_RETRY_COUNT, INT, "30")
OPTION(TRANSPORT_SEND_RETRY_SLEEP_MS, INT, "5000")

// NUMA Options
OPTION(TRANSFER_ENGINE_ENABLE_NUMA_RUN_BINDING, BOOL, "true") // cpu affinity
OPTION(TRANSFER_ENGINE_ENABLE_NUMA_ALLOCATION, BOOL,
       "true") // numa allocation affinity

// BRPC Transport Options
OPTION(BRPC_TRANSPORT_MAX_RETRIES, INT, "90")
OPTION(BRPC_TRANSPORT_TIMEOUT_MS, INT, "10000")

OPTION(DISCOVERY_USE_BATCH_API, BOOL, "true")

// Log Options
OPTION(ASTATE_LOG_BACKEND, STRING, "SPDLOG") // SPDLOG, GLOG
OPTION(ASTATE_LOG_DIR, STRING, "/tmp/astate")
OPTION(ASTATE_LOG_LEVEL, STRING, "INFO") // DEBUG, INFO, WARNING, ERROR
OPTION(ASTATE_LOG_TO_CONSOLE, BOOL, "false")
OPTION(ASTATE_LOG_TO_FILE, BOOL, "true")
OPTION(ASTATE_ENABLE_GLOG_EXT, BOOL, "true")
OPTION(ASTATE_LOG_MAX_FILE_DAYS, INT, "5") // 5 days
/********** Option definition: [Option Name, Option Type, Default Value] ***********/

#define PEERS_HOST_STR_LEN 3

// Value Parser Utils
static int ParseInt(std::any value) {
    if (value.type() == typeid(int)) {
        return std::any_cast<int>(value);
    }
    if (value.type() == typeid(std::string)) {
        return std::stoi(std::any_cast<std::string>(value));
    }
    throw std::invalid_argument("Invalid type for int");
}

static std::string ParseString(const std::any& value) {
    if (value.type() == typeid(std::string)) {
        return std::any_cast<std::string>(value);
    }
    if (value.type() == typeid(const char*)) {
        return std::any_cast<const char*>(value);
    }
    if (value.type() == typeid(char*)) {
        return std::any_cast<char*>(value);
    }
    if (value.type() == typeid(std::string_view)) {
        return std::string(std::any_cast<std::string_view>(value));
    }

    SPDLOG_ERROR("Invalid type for string, got type: {}", value.type().name());
    throw std::invalid_argument("Invalid type for string");
}

static std::vector<std::string> ParseStringList(std::any value, const char& delimiter = DELIMITER) {
    if (value.type() == typeid(std::string)) {
        auto str = std::any_cast<std::string>(value);
        return SplitString(str, delimiter);
    }
    if (value.type() == typeid(std::vector<std::string>)) {
        return std::any_cast<std::vector<std::string>>(value);
    }
    throw std::invalid_argument("Invalid type for string list");
}

static bool ParseBool(std::any value) {
    if (value.type() == typeid(bool)) {
        return std::any_cast<bool>(value);
    }
    if (value.type() == typeid(std::string)) {
        auto str = std::any_cast<std::string>(value);
        if (str == "true" || str == "True" || str == "TRUE" || str == "1") {
            return true;
        }
        if (str == "false" || str == "False" || str == "FALSE" || str == "0") {
            return false;
        }
        throw std::invalid_argument("Invalid type for bool: " + str);
    }
    throw std::invalid_argument("Invalid type for bool: " + std::string(value.type().name()));
}

static float ParseFloat(std::any value) {
    if (value.type() == typeid(float)) {
        return std::any_cast<float>(value);
    }
    if (value.type() == typeid(std::string)) {
        return std::stof(std::any_cast<std::string>(value));
    }
    throw std::invalid_argument("Invalid type for float");
}

static double ParseDouble(std::any value) {
    if (value.type() == typeid(double)) {
        return std::any_cast<double>(value);
    }
    if (value.type() == typeid(std::string)) {
        return std::stod(std::any_cast<std::string>(value));
    }
    throw std::invalid_argument("Invalid type for double");
}

static int64_t ParseLong(std::any value) {
    if (value.type() == typeid(int64_t)) {
        return std::any_cast<int64_t>(value);
    }
    if (value.type() == typeid(std::string)) {
        return std::stol(std::any_cast<std::string>(value));
    }
    throw std::invalid_argument("Invalid type for INT64");
}

static std::any ParseValue(ValueType type, const std::any& value) {
    switch (type) {
        case ValueType::INT:
            return ParseInt(value);
        case ValueType::STRING:
            return ParseString(value);
        case ValueType::STRING_LIST:
            return ParseStringList(value);
        case ValueType::BOOL:
            return ParseBool(value);
        case ValueType::FLOAT:
            return ParseFloat(value);
        case ValueType::DOUBLE:
            return ParseDouble(value);
        case ValueType::INT64:
            return ParseLong(value);
        default:
            throw std::invalid_argument("Invalid type for value: " + std::to_string(type));
    }
}

// Get & Put Option Value from Options
template <typename T>
T GetOptionValue(const Options& options, const std::string& name) {
    auto& defs = GetOptionDefinitions();
    auto def_iter = defs.find(name);
    if (def_iter == defs.end()) {
        SPDLOG_ERROR("Option definition for {} not found", name);
        return T{};
    }

    const auto& def = def_iter->second;

    auto opt_iter = options.find(name);
    if (opt_iter != options.end()) {
        return std::any_cast<T>(ParseValue(def.value_type, opt_iter->second));
    }

    if (def.default_value.empty()) {
        SPDLOG_ERROR("Option {} not set and no default value", name);
        return T{};
    }

    return std::any_cast<T>(ParseValue(def.value_type, def.default_value));
}

void PutOptionValue(Options& options, const std::string& name, const std::string& value);

// Utils for config
std::string GetOptionFromEnv(const std::string& name);
void LoadOptions(Options& options);
void LoadOptionsFromEnv(Options& options);
void LoadOptionsFromFile(Options& options, std::string file_path = "");

} // namespace astate
