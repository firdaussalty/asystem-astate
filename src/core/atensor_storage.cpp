#include "core/atensor_storage.h"

#include "core/remote_tensor_table.h"
#include "transfer/tensor_transfer_pull.h"

namespace astate {

bool ATensorStorageCtx::Init(const Options& options, const AParallelConfig& parallel_config) {
    this->options = options;
    this->parallel_config = parallel_config;
    // TransferServiceBuilder::Build(TransferEngineBackendType, options_);
    if (GetOptionValue<std::string>(options, TENSOR_TRANSFER_SERVICE_TYPE) == "PULL") {
        transfer_service = new TensorTransferPull(this);
        return transfer_service->Start(options, parallel_config);
    }
    SPDLOG_INFO("Tensor transfer service type is not specified, skip init "
                "transfer service");
    return true;
}

bool ATensorStorageImpl::Init(const AParallelConfig& parallel_config) {
    // Load configs from ENV
    Options options;
    LoadOptions(options);

    // Initialize tensor storage context
    if (!ctx_->Init(options, parallel_config)) {
        SPDLOG_ERROR("ATensorStorageCtx init failed");
        return false;
    }
    return true;
}

std::shared_ptr<TensorTable>
ATensorStorageImpl::CreateTable(TensorTableType table_type, const std::string& table_name) {
    std::lock_guard<std::mutex> lock(register_mutex_);

    // Check if table already exists and is still valid
    auto it = registered_tables_.find(table_name);
    if (it != registered_tables_.end()) {
        if (auto existing = it->second.lock()) {
            // Table exists and is still valid
            return existing;
        }
        // Table exists but has been destroyed, remove it
        registered_tables_.erase(it);
    }

    // Create new table based on type
    std::shared_ptr<TensorTable> table;
    switch (table_type) {
        case TensorTableType::IN_MEMORY:
            table = std::make_shared<InMemoryTensorTable>(table_name);
            break;
        case TensorTableType::REMOTE:
            table = std::make_shared<RemoteTensorTable>(table_name, ctx_);
            if (ctx_->tensor_table == nullptr) {
                ctx_->tensor_table = table.get();
            } else {
                throw std::runtime_error("multi remote tensor tables are NOT supported yet!");
            }
            break;
        default:
            throw std::invalid_argument("Unsupported TensorTable type");
    }

    registered_tables_[table_name] = table;
    return table;
}

std::shared_ptr<TensorTable>
ATensorStorageImpl::RegisterTable(TensorTableType table_type, const std::string& table_name) {
    pybind11::gil_scoped_release release;
    // Backward compatibility: default to IN_MEMORY implementation
    return CreateTable(table_type, table_name);
}

std::shared_ptr<ATensorStorage> CreateATensorStorage(const AParallelConfig& parallel_config) {
    static std::mutex ats_mutex;
    static std::shared_ptr<ATensorStorage> ats_instance;

    const std::string log_name = "astate_storage";
    char* argv[] = {const_cast<char*>(log_name.c_str())};
    astate::InitAstateLog(1, argv);

    pybind11::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(ats_mutex);
    if (ats_instance == nullptr) {
        ats_instance = std::make_shared<ATensorStorageImpl>();
        bool init = ats_instance->Init(parallel_config);
        if (!init) {
            SPDLOG_ERROR("Failed to initialize ATensorStorage");
            ats_instance.reset();
            return nullptr;
        }
    }
    return ats_instance;
}

} // namespace astate
