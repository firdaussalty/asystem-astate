#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/option.h"
#include "core/atensor.h"
#include "core/shardedkey.h"
#include "transfer/types.h"

namespace astate {

enum class TransferEngineBackendType : uint8_t { PULL = 1, PUSH = 2, CACHE = 3 };

const uint8_t NO_OP = 0x0;
const uint8_t READ = 0x1;
const uint8_t WRITE = 0x2;

static bool IsRead(std::atomic<uint8_t>& type) {
    return (type.load() & READ) != NO_OP;
}
static bool IsWrite(std::atomic<uint8_t>& type) {
    return (type.load() & WRITE) != NO_OP;
}

static void SetRead(std::atomic<uint8_t>& type) {
    if (!IsRead(type)) {
        type.store(type.load() | READ);
    }
}

static void SetWrite(std::atomic<uint8_t>& type) {
    if (!IsWrite(type)) {
        type.store(type.load() | WRITE);
    }
}

static void Clear(std::atomic<uint8_t>& type) {
    type.store(NO_OP);
}

static std::string GetDataOperationString(std::atomic<uint8_t>& type) {
    if (IsRead(type)) {
        return "READ";
    }
    if (IsWrite(type)) {
        return "WRITE";
    }
    return "NO_OP";
}

#define BYTES_TO_KB(bytes) (bytes / 1024.0)
#define BYTES_TO_MB(bytes) (bytes / 1024.0 / 1024.0)
#define BYTES_TO_GB(bytes) (bytes / 1024.0 / 1024.0 / 1024.0)

/*
 * TensorTransferService is a base class for all tensor transfer services.
 * It provides the interface for all tensor transfer services.
 */
class TensorTransferService {
 public:
    virtual ~TensorTransferService() = default;
    virtual bool Start(const Options& options, const AParallelConfig& parallel_config) = 0;
    virtual void Stop() = 0;
    [[nodiscard]] virtual bool IsRunning() const = 0;

    virtual bool Put(int64_t seq_id, const ShardedKey& tensor_key, const ATensor& atensor) = 0;
    virtual bool MultiPut(int64_t seq_id, const std::vector<std::pair<ShardedKey, ATensor>>& atensors) = 0;

    virtual bool Get(int64_t seq_id, const ShardedKey& tensor_key, ATensor& atensor) = 0;
    virtual bool MultiGet(int64_t seq_id, std::vector<std::pair<ShardedKey, ATensor>>& atensors) = 0;

    virtual bool
    RawGet(int64_t seq_id, const ATStorage& astorage, const NodeInfo& node_info, const void* remote_addr, size_t len)
        = 0;

    virtual bool PreRegisterMemory(ATStorage& atensor_storage) = 0;
    virtual std::vector<CompactTensorInfo>
    GetCompactTensorInfos(int64_t seq_id, std::unordered_map<ShardedKey, ATensor, ShardedKeyHash> atensors) = 0;

    virtual void Complete() = 0;

    [[nodiscard]] virtual std::vector<std::pair<ShardedKey, ATensor>>
    GetAllTensorShards(int64_t seq_id, std::function<bool(const ShardedKey&)> filter) = 0;
};

using TensorTransferDistribution
    = std::unordered_map<NodeInfo, std::unordered_set<ShardedKey, ShardedKeyHash>, NodeInfoHash>;

/**
 * @brief Distribute the tensor transfer tasks to the nodes
 * @param tensor_transfer_meta The tensor transfer meta
 * @param main_replica The replica number needed for each tensor shard
 * @return TensorTransferDistribution The tensor transfer distribution
 */
TensorTransferDistribution
DistributeTensorTransferMeta(const TransferTensorMeta& tensor_transfer_meta, int main_replica);

} // namespace astate
