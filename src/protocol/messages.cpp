#include "protocol/messages.h"

#include <stdexcept>

#include <json/json.h>
#include <spdlog/spdlog.h>

#include "core/shardedkey.h"
#include "transport/atensor_serializer.h"

namespace astate {

// 辅助函数：检查必需的JSON字段
void checkRequiredField(const Json::Value& root, const std::string& field) {
    if (!root.isMember(field)) {
        throw std::runtime_error("Missing required field: " + field);
    }
}

// 辅助函数：检查字段类型
void checkFieldType(const Json::Value& root, const std::string& field, Json::ValueType type) {
    if (root[field].type() != type) {
        throw std::runtime_error("Invalid type for field: " + field);
    }
}

// NodeInfo 序列化
Json::Value toJson(const NodeInfo& info) {
    Json::Value root;
    root["hostname_or_ip"] = info.hostname_or_ip;
    root["rdma_port"] = info.rdma_port;
    root["ctrl_flow_port"] = info.ctrl_flow_port;
    return root;
}

NodeInfo fromJson(const Json::Value& root, const NodeInfo&) {
    try {
        checkRequiredField(root, "hostname_or_ip");
        checkRequiredField(root, "rdma_port");
        checkRequiredField(root, "ctrl_flow_port");

        checkFieldType(root, "hostname_or_ip", Json::stringValue);
        checkFieldType(root, "rdma_port", Json::intValue);
        checkFieldType(root, "ctrl_flow_port", Json::intValue);

        NodeInfo info;
        info.hostname_or_ip = root["hostname_or_ip"].asString();

        // 验证端口范围
        int rdma_port = root["rdma_port"].asInt();
        int ctrl_flow_port = root["ctrl_flow_port"].asInt();
        if (rdma_port <= 0 || rdma_port > 65535) {
            throw std::runtime_error("Invalid RDMA port number: " + std::to_string(rdma_port));
        }
        if (ctrl_flow_port <= 0 || ctrl_flow_port > 65535) {
            throw std::runtime_error("Invalid control flow port number: " + std::to_string(ctrl_flow_port));
        }
        info.rdma_port = rdma_port;
        info.ctrl_flow_port = ctrl_flow_port;

        return info;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to deserialize NodeInfo: {}", e.what());
        throw;
    }
}

// ATDeviceType JSON序列化辅助函数
Json::Value toJson(const ATDeviceType& device_type) {
    return Json::Value(static_cast<int>(device_type));
}

ATDeviceType fromJsonDeviceType(const Json::Value& json) {
    if (!json.isInt()) {
        throw std::runtime_error("Invalid ATDeviceType JSON format");
    }
    return static_cast<ATDeviceType>(json.asInt());
}

// ATDevice JSON序列化
Json::Value toJson(const ATDevice& device) {
    Json::Value root;
    root["device_type"] = toJson(device.device_type);
    root["device_index"] = device.device_index;
    return root;
}

ATDevice fromJsonDevice(const Json::Value& root) {
    checkRequiredField(root, "device_type");
    checkRequiredField(root, "device_index");
    checkFieldType(root, "device_type", Json::intValue);
    checkFieldType(root, "device_index", Json::intValue);

    ATDevice device;
    device.device_type = fromJsonDeviceType(root["device_type"]);
    device.device_index = static_cast<ATDeviceIndex>(root["device_index"].asInt());
    return device;
}

// ATStorage JSON序列化
Json::Value toJson(const ATStorage& storage) {
    Json::Value root;
    root["storage_size"] = storage.storage_size;
    root["device"] = toJson(storage.device);
    // 注意：不序列化data指针，因为它是运行时特定的内存地址
    return root;
}

ATStorage fromJsonStorage(const Json::Value& root) {
    checkRequiredField(root, "storage_size");
    checkRequiredField(root, "device");
    // checkFieldType(root, "storage_size", Json::uintValue);
    checkFieldType(root, "device", Json::objectValue);

    ATStorage storage;
    storage.storage_size = root["storage_size"].asUInt64();
    storage.device = fromJsonDevice(root["device"]);
    storage.data = nullptr; // 运行时设置
    return storage;
}

// ATensor JSON序列化
Json::Value toJson(const ATensor& atensor) {
    Json::Value root;

    // 序列化基本字段
    root["storage_offset"] = Json::Value::Int64(atensor.storage_offset);
    root["dim_num"] = atensor.dim_num;
    root["dtype"] = static_cast<int>(atensor.dtype);
    root["conj"] = atensor.conj;
    root["neg"] = atensor.neg;
    root["requires_grad"] = atensor.requires_grad;

    // 序列化size数组
    Json::Value sizeArray(Json::arrayValue);
    if (atensor.size != nullptr) {
        for (int32_t i = 0; i < atensor.dim_num; ++i) {
            sizeArray.append(Json::Value::Int64(atensor.size[i]));
        }
    }
    root["size"] = sizeArray;

    // 序列化stride数组
    Json::Value strideArray(Json::arrayValue);
    if (atensor.stride != nullptr) {
        for (int32_t i = 0; i < atensor.dim_num; ++i) {
            strideArray.append(Json::Value::Int64(atensor.stride[i]));
        }
    }
    root["stride"] = strideArray;

    // 序列化storage
    root["storage"] = toJson(atensor.storage);

    return root;
}

ATensor fromJsonATensor(const Json::Value& root) {
    checkRequiredField(root, "storage_offset");
    checkRequiredField(root, "dim_num");
    checkRequiredField(root, "dtype");
    checkRequiredField(root, "conj");
    checkRequiredField(root, "neg");
    checkRequiredField(root, "requires_grad");
    checkRequiredField(root, "size");
    checkRequiredField(root, "stride");
    checkRequiredField(root, "storage");

    ATensor atensor;
    atensor.storage_offset = root["storage_offset"].asInt64();
    atensor.dim_num = root["dim_num"].asInt();
    atensor.dtype = static_cast<ATDtype>(root["dtype"].asInt());
    atensor.conj = root["conj"].asBool();
    atensor.neg = root["neg"].asBool();
    atensor.requires_grad = root["requires_grad"].asBool();

    // 反序列化size数组
    const Json::Value& sizeArray = root["size"];
    if (sizeArray.isArray() && sizeArray.size() == static_cast<Json::ArrayIndex>(atensor.dim_num)) {
        atensor.size = new int64_t[atensor.dim_num];
        for (int32_t i = 0; i < atensor.dim_num; ++i) {
            atensor.size[i] = sizeArray[i].asInt64();
        }
    } else {
        throw std::runtime_error("Invalid size array in ATensor JSON");
    }

    // 反序列化stride数组
    const Json::Value& strideArray = root["stride"];
    if (strideArray.isArray() && strideArray.size() == static_cast<Json::ArrayIndex>(atensor.dim_num)) {
        atensor.stride = new int64_t[atensor.dim_num];
        for (int32_t i = 0; i < atensor.dim_num; ++i) {
            atensor.stride[i] = strideArray[i].asInt64();
        }
    } else {
        // 清理已分配的size内存
        if (atensor.size != nullptr) {
            delete[] atensor.size;
            atensor.size = nullptr;
        }
        throw std::runtime_error("Invalid stride array in ATensor JSON");
    }

    // 反序列化storage
    atensor.storage = fromJsonStorage(root["storage"]);

    return atensor;
}

// Tensor 内存 RDMA 信息序列化
Json::Value toJson(const TensorMemoryRDMAInfo& info) {
    Json::Value root;
    // 将指针地址转换为字符串，避免直接序列化指针值
    root["addr"] = std::to_string(reinterpret_cast<uintptr_t>(info.addr));
    root["size"] = info.size;
    root["rkey"] = info.rkey;
    // 使用JSON风格序列化tensor meta，而不是二进制序列化
    root["tmeta"] = toJson(info.atensor_meta);
    return root;
}

TensorMemoryRDMAInfo fromJson(const Json::Value& root, const TensorMemoryRDMAInfo&) {
    try {
        checkRequiredField(root, "addr");
        checkRequiredField(root, "size");
        checkRequiredField(root, "rkey");
        checkRequiredField(root, "tmeta");

        checkFieldType(root, "addr", Json::stringValue);
        // checkFieldType(root, "size", Json::uintValue);
        checkFieldType(root, "rkey", Json::stringValue);
        // 修改：tmeta现在是JSON对象而不是字符串
        checkFieldType(root, "tmeta", Json::objectValue);

        // 从字符串解析回指针地址
        uintptr_t addr = std::stoull(root["addr"].asString());

        // 验证大小
        size_t size = root["size"].asUInt64();
        if (size <= 0) {
            throw std::runtime_error("Invalid memory size: " + std::to_string(size));
        }

        // 使用JSON风格反序列化tensor meta
        ATensor atensor_meta = fromJsonATensor(root["tmeta"]);

        return TensorMemoryRDMAInfo{
            reinterpret_cast<void*>(addr), size, root["rkey"].asString(), std::move(atensor_meta)};
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to deserialize TensorMemoryRDMAInfo: {}", e.what());
        throw;
    }
}

// ShardedKey 序列化
Json::Value ToJson(const ShardedKey& key) {
    try {
        Json::Value root;
        root["key"] = key.key;

        // 序列化 global_shape
        Json::Value shape_array(Json::arrayValue);
        for (const auto& dim : key.global_shape) {
            shape_array.append(Json::Value::Int64(dim));
        }
        root["global_shape"] = shape_array;

        // 序列化 global_offset
        Json::Value offset_array(Json::arrayValue);
        for (const auto& dim : key.global_offset) {
            offset_array.append(Json::Value::Int64(dim));
        }
        root["global_offset"] = offset_array;

        return root;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to serialize ShardedKey: {}", e.what());
        throw;
    }
}

ShardedKey FromJson(const Json::Value& root, const ShardedKey&) {
    try {
        checkRequiredField(root, "key");
        checkRequiredField(root, "global_shape");
        checkRequiredField(root, "global_offset");

        checkFieldType(root, "key", Json::stringValue);
        checkFieldType(root, "global_shape", Json::arrayValue);
        checkFieldType(root, "global_offset", Json::arrayValue);

        ShardedKey key;
        key.key = root["key"].asString();

        // 反序列化 global_shape
        const Json::Value& shape_array = root["global_shape"];
        for (const auto& dim : shape_array) {
            if (!dim.isInt64()) {
                throw std::runtime_error("Invalid shape dimension type");
            }
            key.global_shape.push_back(dim.asInt64());
        }

        // 反序列化 global_offset
        const Json::Value& offset_array = root["global_offset"];
        for (const auto& dim : offset_array) {
            if (!dim.isInt64()) {
                throw std::runtime_error("Invalid offset dimension type");
            }
            key.global_offset.push_back(dim.asInt64());
        }

        // 验证 shape 和 offset 长度是否匹配
        if (key.global_shape.size() != key.global_offset.size()) {
            throw std::runtime_error("Shape and offset dimensions do not match");
        }

        return key;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to deserialize ShardedKey: {}", e.what());
        throw;
    }
}

// Tensor RDMA 元数据发布消息序列化
Json::Value ToJson(const TensorRDMAMetaPublishMessage& msg) {
    try {
        Json::Value root;
        root["seq_id"] = Json::Value::Int64(msg.seq_id);
        root["node_info"] = toJson(msg.node_info);

        Json::Value metas(Json::objectValue);
        for (const auto& pair : msg.tensor_rdma_metas) {
            Json::Value keyJson = ToJson(pair.first);
            Json::Value valueJson = toJson(pair.second);
            metas[Serialize(keyJson)] = valueJson;
        }
        root["tensor_rdma_metas"] = metas;

        return root;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to serialize TensorRDMAMetaPublishMessage: {}", e.what());
        throw;
    }
}

TensorRDMAMetaPublishMessage FromJson(const Json::Value& root, const TensorRDMAMetaPublishMessage&) {
    try {
        checkRequiredField(root, "seq_id");
        checkRequiredField(root, "node_info");
        checkRequiredField(root, "tensor_rdma_metas");

        checkFieldType(root, "seq_id", Json::intValue);
        checkFieldType(root, "node_info", Json::objectValue);
        checkFieldType(root, "tensor_rdma_metas", Json::objectValue);

        TensorRDMAMetaPublishMessage msg;
        msg.seq_id = root["seq_id"].asInt64();
        msg.node_info = fromJson(root["node_info"], NodeInfo{});

        const Json::Value& metas = root["tensor_rdma_metas"];
        for (const auto& key : metas.getMemberNames()) {
            Json::Value keyJson = Deserialize(key);
            ShardedKey shardKey = FromJson(keyJson, ShardedKey{});

            msg.tensor_rdma_metas[shardKey] = fromJson(metas[key], TensorMemoryRDMAInfo{});
        }

        return msg;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to deserialize TensorRDMAMetaPublishMessage: {}", e.what());
        throw;
    }
}

// WeightReadyMessage 序列化
Json::Value ToJson(const WeightReadyMessage& msg) {
    try {
        Json::Value root;
        root["seq_id"] = Json::Value::Int64(msg.seq_id);
        root["node_info"] = toJson(msg.node_info);
        return root;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to serialize WeightReadyMessage: {}", e.what());
        throw;
    }
}

WeightReadyMessage FromJson(const Json::Value& root, const WeightReadyMessage&) {
    try {
        checkRequiredField(root, "seq_id");
        checkRequiredField(root, "node_info");

        checkFieldType(root, "seq_id", Json::intValue);
        checkFieldType(root, "node_info", Json::objectValue);

        WeightReadyMessage msg;
        msg.seq_id = root["seq_id"].asInt64();
        msg.node_info = fromJson(root["node_info"], NodeInfo{});
        return msg;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to deserialize WeightReadyMessage: {}", e.what());
        throw;
    }
}

// WeightConsumedMessage 序列化
Json::Value ToJson(const WeightConsumedMessage& msg) {
    try {
        Json::Value root;
        root["seq_id"] = Json::Value::Int64(msg.seq_id);
        root["node_info"] = toJson(msg.node_info);
        return root;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to serialize WeightConsumedMessage: {}", e.what());
        throw;
    }
}

WeightConsumedMessage FromJson(const Json::Value& root, const WeightConsumedMessage&) {
    try {
        checkRequiredField(root, "seq_id");
        checkRequiredField(root, "node_info");

        checkFieldType(root, "seq_id", Json::intValue);
        checkFieldType(root, "node_info", Json::objectValue);

        WeightConsumedMessage msg;
        msg.seq_id = root["seq_id"].asInt64();
        msg.node_info = fromJson(root["node_info"], NodeInfo{});
        return msg;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to deserialize WeightConsumedMessage: {}", e.what());
        throw;
    }
}

// 通用序列化/反序列化函数
std::string Serialize(const Json::Value& json) {
    Json::StreamWriterBuilder writer;
    return Json::writeString(writer, json);
}

Json::Value Deserialize(const std::string& str) {
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

} // namespace astate
