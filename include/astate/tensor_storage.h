#pragma once

#include <memory>
#include <string>

#include "astate/config.h"
#include "astate/tensor_table.h"
#include "astate/types.h"

namespace astate {

class ATensorStorage {
 public:
    ATensorStorage() = default;
    virtual ~ATensorStorage() = default;

    ATensorStorage(const ATensorStorage&) = delete;
    ATensorStorage& operator=(const ATensorStorage&) = delete;

    virtual bool Init(const AParallelConfig& parallel_config) = 0;

    /**
     * Create or get a tensor table instance
     * @param table_type The type of table implementation
     * @param table_name The name of the table
     * @param parallel_config The parallel configuration for the table
     * @return A shared pointer to the TensorTable instance
     */
    virtual std::shared_ptr<TensorTable> RegisterTable(TensorTableType table_type, const std::string& table_name) = 0;
};

std::shared_ptr<ATensorStorage> CreateATensorStorage(const AParallelConfig& parallel_config);

} // namespace astate
