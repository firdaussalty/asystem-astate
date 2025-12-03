#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <json/json.h>
#include <json/reader.h>
#include <json/value.h>
#include <json/writer.h>
#include <spdlog/spdlog.h>

#include "common/option.h"
#include "common/string_utils.h"
#include "core/atensor.h"
#include "core/shardedkey.h"

namespace astate {

// Serialize and deserialize functions for Json
inline std::string SerializeToJsonStr(const Json::Value& json) {
    Json::StreamWriterBuilder writer;
    return Json::writeString(writer, json);
}

inline Json::Value DeserializeFromJsonStr(const std::string& str) {
    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string errors;
    std::istringstream iss(str);
    if (!Json::parseFromStream(reader, iss, &root, &errors)) {
        SPDLOG_ERROR("Failed to parse JSON: {}", errors);
        throw std::runtime_error("JSON parsing failed");
    }
    return root;
}

// 节点信息
struct NodeInfo {
    std::string hostname_or_ip;
    int rdma_port;
    int ctrl_flow_port;

    bool operator==(const NodeInfo& other) const {
        return hostname_or_ip == other.hostname_or_ip && rdma_port == other.rdma_port
            && ctrl_flow_port == other.ctrl_flow_port;
    }

    [[nodiscard]] std::string GetHostWithRdmaPort() const { return hostname_or_ip + ":" + std::to_string(rdma_port); }

    [[nodiscard]] std::string ToString() const {
        return "host=" + hostname_or_ip + ", rdma_port=" + std::to_string(rdma_port)
            + ", ctrl_port=" + std::to_string(ctrl_flow_port);
    }

    static std::vector<NodeInfo> GetNodeInfos(const std::vector<std::string>& peers_host) {
        std::vector<NodeInfo> nodes;
        for (const auto& host : peers_host) {
            // host format: "hostname:rdma_port:ctrl_flow_port"
            std::vector<std::string> host_info = SplitString(host, COLON);
            if (host_info.size() == PEERS_HOST_STR_LEN) {
                NodeInfo node_info{host_info[0], std::stoi(host_info[1]), std::stoi(host_info[2])};
                nodes.push_back(node_info);
            } else {
                SPDLOG_ERROR("Invalid peer host format: {}", host);
            }
        }
        return nodes;
    }
};

// Tensor 内存 RDMA 信息
struct TensorMemoryRDMAInfo {
    void* addr;
    size_t size;
    std::string rkey;

    // NOTE: only the fields of ATensor are serialized, the data is not serialized
    ATensor atensor_meta;

 public:
    TensorMemoryRDMAInfo()
        : addr(nullptr),
          size(0) {}

    TensorMemoryRDMAInfo(void* addr, size_t size, std::string rkey, const ATensor& atensor_meta)
        : addr(addr),
          size(size),
          rkey(std::move(rkey)),
          atensor_meta(atensor_meta) {}

    TensorMemoryRDMAInfo(void* addr, size_t size, std::string rkey, ATensor&& atensor_meta)
        : addr(addr),
          size(size),
          rkey(std::move(rkey)),
          atensor_meta(std::move(atensor_meta)) {}

    TensorMemoryRDMAInfo(const TensorMemoryRDMAInfo& other) = default;
    TensorMemoryRDMAInfo& operator=(const TensorMemoryRDMAInfo& other) = default;

    TensorMemoryRDMAInfo(TensorMemoryRDMAInfo&& other) noexcept = default;
    TensorMemoryRDMAInfo& operator=(TensorMemoryRDMAInfo&& other) noexcept = default;
};

struct TensorRDMAMetaPublishMessage {
    int64_t seq_id;
    NodeInfo node_info;
    std::unordered_map<ShardedKey, TensorMemoryRDMAInfo, ShardedKeyHash> tensor_rdma_metas{};
};

struct WeightReadyMessage {
    int64_t seq_id{};
    NodeInfo node_info;
};

// 权重消费完成消息
struct WeightConsumedMessage {
    int64_t seq_id{};
    NodeInfo node_info;
};

// 辅助函数声明
void CheckRequiredField(const Json::Value& root, const std::string& field);
void CheckFieldType(const Json::Value& root, const std::string& field, Json::ValueType type);

std::string Serialize(const Json::Value& json);
Json::Value Deserialize(const std::string& str);

// ATensor相关类型的JSON序列化函数声明
Json::Value ToJson(const ATDeviceType& device_type);
ATDeviceType FromJsonDeviceType(const Json::Value& json);
Json::Value ToJson(const ATDevice& device);
ATDevice FromJsonDevice(const Json::Value& root);
Json::Value ToJson(const ATStorage& storage);
ATStorage FromJsonStorage(const Json::Value& root);
Json::Value ToJson(const ATensor& atensor);
ATensor FromJsonATensor(const Json::Value& root);

// 序列化函数声明
Json::Value ToJson(const NodeInfo& info);
Json::Value ToJson(const TensorMemoryRDMAInfo& info);
Json::Value ToJson(const ShardedKey& key);
Json::Value ToJson(const TensorRDMAMetaPublishMessage& msg);
Json::Value ToJson(const WeightReadyMessage& msg);
Json::Value ToJson(const WeightConsumedMessage& msg);

// 反序列化函数声明
NodeInfo FromJson(const Json::Value& root, const NodeInfo&);
TensorMemoryRDMAInfo FromJson(const Json::Value& root, const TensorMemoryRDMAInfo&);
ShardedKey FromJson(const Json::Value& root, const ShardedKey&);
TensorRDMAMetaPublishMessage FromJson(const Json::Value& root, const TensorRDMAMetaPublishMessage&);
WeightReadyMessage FromJson(const Json::Value& root, const WeightReadyMessage&);
WeightConsumedMessage FromJson(const Json::Value& root, const WeightConsumedMessage&);

} // namespace astate
