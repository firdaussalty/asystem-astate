#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include "astate/tensor_storage.h"
#include "astate/types.h"
#include "common/log_utils.h"
#include "common/option.h"
#include "core/atensor.h"
#include "core/shardedkey.h"
#include "core/tensor_table.h"
#include "transfer/tensor_transfer_service.h"

namespace astate {

// Context for ATensorStorage
struct ATensorStorageCtx {
    Options options;
    AParallelConfig parallel_config{};
    TensorTable* tensor_table = nullptr;
    TensorTransferService* transfer_service = nullptr;

    ATensorStorageCtx() = default;

    ~ATensorStorageCtx() { Cleanup(); }

    bool Init(const Options& options, const AParallelConfig& parallel_config);

    void Cleanup() const {
        if (transfer_service != nullptr) {
            transfer_service->Stop();
        }
    }
};

// Class of ATensorStorage to provide a unified interface for tensor storage
class ATensorStorageImpl : public ATensorStorage {
 public:
    ATensorStorageImpl() = default;

    ~ATensorStorageImpl() override {
        for (auto& table : registered_tables_) {
            std::shared_ptr<TensorTable> table_ptr = std::static_pointer_cast<TensorTable>(table.second.lock());
            // table_ptr->close();
        }
        registered_tables_.clear();
    }

    bool Init(const AParallelConfig& parallel_config) override;

    /**
     * Create or get a tensor table instance
     * @param table_type The type of table implementation
     * @param table_name The name of the table
     * @param parallel_config The parallel configuration for the table
     * @return A shared pointer to the TensorTable instance
     */
    std::shared_ptr<TensorTable> RegisterTable(TensorTableType table_type, const std::string& table_name) override;

 private:
    std::shared_ptr<ATensorStorageCtx> ctx_ = std::make_shared<ATensorStorageCtx>();

    std::unordered_map<std::string, std::weak_ptr<TensorTable>> registered_tables_;

    // Mutex for thread-safe access
    std::mutex register_mutex_;

    std::shared_ptr<TensorTable> CreateTable(TensorTableType table_type, const std::string& table_name);
};

} // namespace astate
