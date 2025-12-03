#pragma once

#include <string>
#include <utility>

#include "astate/sharded_key.h"
#include "astate/types.h"

namespace pybind11 {
class object;
}

namespace astate {

class TensorTable {
 public:
    explicit TensorTable(std::string name)
        : name_(std::move(name)) {}
    virtual ~TensorTable() = default;

    TensorTable(const TensorTable&) = delete;
    TensorTable& operator=(const TensorTable&) = delete;

    /**
     * Store a single tensor to the table
     * @param seq_id Sequence ID used to identify data batch
     * @param tensor_key Sharded key that uniquely identifies tensor
     * @param tensor Tensor object to store (Python object)
     * @return true if successful, false if failed
     */
    virtual bool Put(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& tensor) = 0;

    /**
      * Retrieve a single tensor from the table (in-place update)
      * @param seq_id Sequence ID used to identify data batch
      * @param tensor_key Sharded key of the tensor to retrieve
      * @param py_tensor Output parameter, tensor will be updated to this object if found
      * @return true if found and successfully updated, false if not found
      */
    virtual bool Get(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& py_tensor) = 0;

    /**
      * Get tensor object based on metadata
      * @param seq_id Sequence ID used to identify data batch
      * @param tensor_key Sharded key of the tensor to retrieve
      * @param tensor_meta Tensor metadata information including dtype, shape, device, etc.
      * @return Python tensor object if found, None if not found
      */
    virtual pybind11::object GetTensor(int64_t seq_id, const ShardedKey& tensor_key, const TorchTensorMeta& tensor_meta)
        = 0;

    /**
      * Store multiple tensors in batch
      * @param seq_id Sequence ID used to identify data batch
      * @param tensor_list Tensor list containing key-value pairs <sharded_key, tensor_object>
      * @return true if all tensors are successfully stored, false if any fails
      */
    virtual bool MultiPut(int64_t seq_id, const std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_list) = 0;

    /**
      * Retrieve multiple tensors in batch (in-place update)
      * @param seq_id Sequence ID used to identify data batch
      * @param tensor_list Input/output parameter containing keys to retrieve, found tensors will be updated to
      * corresponding objects
      * @return true if at least one tensor is found, false if none found
      */
    virtual bool MultiGet(int64_t seq_id, std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_list) = 0;

    /**
      * Get multiple tensor objects based on metadata list
      * @param seq_id Sequence ID used to identify data batch
      * @param tensor_meta_list List of tensor metadata containing tensor keys and corresponding metadata
      * @return List of found tensors, missing tensors will not be included in the result
      */
    virtual std::vector<std::pair<ShardedKey, pybind11::object>>
    MultiGetTensor(int64_t seq_id, const std::vector<std::pair<ShardedKey, TorchTensorMeta>>& tensor_meta_list) = 0;

    /**
      * Complete all operations for the specified sequence ID
      *
      * This interface "may" be a blocking interface, depending on the underlying implementation mode:
      *      - In some modes, this interface needs to wait until both read and write groups complete
      *      - In some modes, this interface needs to wait until data in memory is persisted to storage
      *      - The actual behavior depends on the table type and configuration
      * @param seq_id Sequence ID to complete
      */
    virtual void Complete(int64_t seq_id) = 0;

    /**
      * Scan all tensor metadata for the specified sequence ID
      * @param seq_id Sequence ID to scan
      * @return List of all tensor metadata in this sequence,
      *         containing tensor names and metadata information
      */
    virtual std::vector<std::pair<std::string, TorchTensorMeta>> ScanTensorMeta(int64_t seq_id) = 0;

    virtual void PrefetchCachedTensors(int64_t seq_id) = 0;

    [[nodiscard]] const std::string& Name() const { return name_; }

 protected:
    std::string name_;
};

} // namespace astate
