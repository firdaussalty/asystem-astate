#include <spdlog/spdlog.h>

#include "core/utils.h"
#include "transfer/tensor_transfer_service.h"
#include "transfer/types.h"

namespace astate {

/**
 * @brief Validate the tensor transfer meta
 * @param tensor_transfer_meta The tensor transfer meta to be validated
 */
void ValidateTensorTransferMeta(const TransferTensorMeta& tensor_transfer_meta) {
    for (const auto& [tensor_key, transfer_meta_list] : tensor_transfer_meta) {
        if (transfer_meta_list.empty()) {
            SPDLOG_ERROR("transfer_meta_list is empty, tensor_key: {}", tensor_key.key);
            throw std::runtime_error("transfer_meta_list is empty, tensor_key: " + tensor_key.key);
        }

        const auto& sample_meta = transfer_meta_list[0];
        for (const auto& meta : transfer_meta_list) {
            if (meta.addr == nullptr || meta.size == 0 || meta.rkey.empty() || meta.node_info.hostname_or_ip.empty()
                || meta.node_info.rdma_port == 0 || meta.node_info.ctrl_flow_port == 0 || meta.atensor == nullptr) {
                SPDLOG_ERROR(
                    "meta is invalid, tensor_key: {}, addr: {}, size: {}, "
                    "rkey: {}, node_info: {}, rdma_port: {}, "
                    "ctrl_flow_port: {}",
                    tensor_key.key,
                    meta.addr,
                    meta.size,
                    meta.rkey,
                    meta.node_info.hostname_or_ip,
                    meta.node_info.rdma_port,
                    meta.node_info.ctrl_flow_port);
                throw std::runtime_error("meta is invalid, tensor_key: " + tensor_key.key);
            }

            if (meta.atensor->dtype != sample_meta.atensor->dtype) {
                SPDLOG_ERROR(
                    "mismatch dtype, tensor_key: {}, meta.atensor_->dtype: {}, "
                    "sample_meta.atensor_->dtype: {}",
                    tensor_key.key,
                    static_cast<int>(meta.atensor->dtype),
                    static_cast<int>(sample_meta.atensor->dtype));
                throw std::runtime_error(
                    "mismatch dtype, tensor_key: " + tensor_key.key
                    + ", meta.atensor_->dtype: " + std::to_string(static_cast<int>(meta.atensor->dtype))
                    + ", sample_meta.atensor_->dtype: " + std::to_string(static_cast<int>(sample_meta.atensor->dtype)));
            }

            if (meta.atensor->dim_num != sample_meta.atensor->dim_num) {
                SPDLOG_ERROR(
                    "mismatch dim_num, tensor_key: {}, meta.atensor_->dim_num: "
                    "{}, sample_meta.atensor_->dim_num: {}",
                    tensor_key.key,
                    meta.atensor->dim_num,
                    sample_meta.atensor->dim_num);
                throw std::runtime_error(
                    "mismatch dim_num, tensor_key: " + tensor_key.key
                    + ", meta.atensor_->dim_num: " + std::to_string(meta.atensor->dim_num)
                    + ", sample_meta.atensor_->dim_num: " + std::to_string(sample_meta.atensor->dim_num));
            }
            if (memcmp(meta.atensor->size, sample_meta.atensor->size, sizeof(int64_t) * meta.atensor->dim_num) != 0) {
                SPDLOG_ERROR("mismatch size, tensor_key: {}", tensor_key.key);
                throw std::runtime_error("mismatch size, tensor_key: " + tensor_key.key);
            }
            if (memcmp(meta.atensor->stride, sample_meta.atensor->stride, sizeof(int64_t) * meta.atensor->dim_num)
                != 0) {
                SPDLOG_ERROR("mismatch stride, tensor_key: {}", tensor_key.key);
                throw std::runtime_error("mismatch stride, tensor_key: " + tensor_key.key);
            }
        }
    }
}

using TensorShardPair = std::pair<ShardedKey, std::vector<TensorRDMAInfo>>;

/**
 * @brief Determine the most suitable nodes to transfer a tensor shard
 * @param shard_pair The RDMA info list of a tensor shard
 * @param main_replica The replica number needed for each tensor shard
 * @param node_size_map The node size map
 * @return std::vector<NodeInfo> The most suitable nodes
 */
std::vector<NodeInfo> DetermineTopNodeForShard(
    const TensorShardPair& shard_pair,
    int main_replica,
    std::unordered_map<NodeInfo, int64_t, NodeInfoHash>& node_size_map) {
    std::vector<NodeInfo> ret;
    ret.reserve(main_replica);

    auto node_infocompare = [](const std::pair<int64_t, NodeInfo>& a, const std::pair<int64_t, NodeInfo>& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        if (a.second.hostname_or_ip != b.second.hostname_or_ip) {
            return a.second.hostname_or_ip < b.second.hostname_or_ip;
        }
        if (a.second.rdma_port != b.second.rdma_port) {
            return a.second.rdma_port < b.second.rdma_port;
        }
        return a.second.ctrl_flow_port < b.second.ctrl_flow_port;
    };

    std::priority_queue<
        std::pair<int64_t, NodeInfo>,
        std::vector<std::pair<int64_t, NodeInfo>>,
        decltype(node_infocompare)>
        node_pq(node_infocompare);

    for (const auto& transfer_meta : shard_pair.second) {
        NodeInfo node_info = transfer_meta.node_info;

        auto it = node_size_map.find(node_info);
        int64_t node_size = 0;
        if (it != node_size_map.end()) {
            node_size = it->second;
        }

        node_pq.emplace(node_size, node_info);
    }

    auto total_size = static_cast<int64_t>(GetTensorTotalByteSize(*shard_pair.second[0].atensor));
    // determine the top main_replica nodes & update node_size_map
    for (int i = 0; i < main_replica && !node_pq.empty(); i++) {
        auto [node_size, node_info] = node_pq.top();
        node_pq.pop();
        ret.emplace_back(node_info);
        // update node_size_map
        node_size_map[node_info] = node_size + total_size;
    }
    return ret;
}

TensorTransferDistribution
DistributeTensorTransferMeta(const TransferTensorMeta& tensor_transfer_meta, int main_replica) {
    ValidateTensorTransferMeta(tensor_transfer_meta);

    auto tensor_shard_pair_compare = [](const TensorShardPair& a, const TensorShardPair& b) {
        if (a.second.size() != b.second.size()) {
            return a.second.size() < b.second.size();
        }

        auto a_size = GetTensorTotalByteSize(*a.second[0].atensor);
        auto b_size = GetTensorTotalByteSize(*b.second[0].atensor);
        if (a_size != b_size) {
            return a_size < b_size;
        }

        if (a.first.key != b.first.key) {
            return a.first.key < b.first.key;
        }

        if (a.first.global_offset.size() == b.first.global_offset.size()) {
            for (int i = 0; i < a.first.global_offset.size(); i++) {
                if (a.first.global_offset[i] != b.first.global_offset[i]) {
                    return a.first.global_offset[i] < b.first.global_offset[i];
                }
            }
        }
        // for most cases, if the global offset is the same, then the tensor shard pairs are the same
        return a.first.global_offset.size() < b.first.global_offset.size();
    };

    std::priority_queue<TensorShardPair, std::vector<TensorShardPair>, decltype(tensor_shard_pair_compare)>
        shard_pair_pq(tensor_shard_pair_compare);
    for (const auto& [tensor_key, transfer_meta_list] : tensor_transfer_meta) {
        // to guarantee that tensor meta info is the same on each node
        // we sort the transfer meta list strictly by hostname, rdma_port and ctrl_flow_port
        std::sort(
            transfer_meta_list.begin(), transfer_meta_list.end(), [](const TensorRDMAInfo& a, const TensorRDMAInfo& b) {
                if (a.node_info.hostname_or_ip != b.node_info.hostname_or_ip) {
                    return a.node_info.hostname_or_ip < b.node_info.hostname_or_ip;
                }
                if (a.node_info.rdma_port != b.node_info.rdma_port) {
                    return a.node_info.rdma_port < b.node_info.rdma_port;
                }
                return a.node_info.ctrl_flow_port < b.node_info.ctrl_flow_port;
            });
        shard_pair_pq.emplace(tensor_key, transfer_meta_list);
    }

    TensorTransferDistribution ret;
    std::unordered_map<NodeInfo, int64_t, NodeInfoHash> node_size_map;
    while (!shard_pair_pq.empty()) {
        auto shard_pair = shard_pair_pq.top();
        shard_pair_pq.pop();

        std::vector<NodeInfo> top_nodes = DetermineTopNodeForShard(shard_pair, main_replica, node_size_map);
        for (const auto& node : top_nodes) {
            auto it = ret.find(node);
            if (it == ret.end()) {
                ret[node] = std::unordered_set<ShardedKey, ShardedKeyHash>();
            }
            ret[node].emplace(shard_pair.first);
        }
    }
    return ret;
}

} // namespace astate
