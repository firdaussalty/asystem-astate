#include "core/remote_tensor_table.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/Context.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "common/lock_utils.h"
#include "common/numa_aware_allocator.h"
#include "common/option.h"
#include "core/atensor.h"
#include "core/atensor_storage.h"
#include "core/shardedkey.h"
#include "core/tensor_sharded_ops.h"
#include "core/tensor_table.h"
#include "core/utils.h"

namespace astate {

// Remote implementation of TensorTable
RemoteTensorTable::RemoteTensorTable(const std::string& name, std::shared_ptr<ATensorStorageCtx> ctx)
    : TensorTable(name),
      ctx_(std::move(ctx)),
      enable_write_gpu_async_copy_(GetOptionValue<bool>(ctx_->options, TRANSFER_ENGINE_ENABLE_WRITE_GPU_ASYNC_COPY)),
      enable_read_gpu_async_copy_(GetOptionValue<bool>(ctx_->options, TRANSFER_ENGINE_ENABLE_READ_GPU_ASYNC_COPY)),
      small_tensor_compact_cache_size_(
          GetOptionValue<long>(ctx_->options, TRANSFER_ENGINE_SMALL_TENSOR_COMPACT_CACHE_SIZE)),
      small_tensor_size_(GetOptionValue<long>(ctx_->options, TRANSFER_ENGINE_SMALL_TENSOR_SIZE)),
      enable_local_cache_prefetch_(GetOptionValue<bool>(ctx_->options, TRANSFER_ENGINE_ENABLE_LOCAL_CACHE_PREFETCH)),
      local_cached_tensors_size_limit_(GetOptionValue<long>(ctx_->options, TRANSFER_ENGINE_LOCAL_CACHE_TENSORS_SIZE)),
      pinned_memory_enabled_(torch::cuda::is_available()),
      enable_numa_allocation_(GetOptionValue<bool>(ctx_->options, TRANSFER_ENGINE_ENABLE_NUMA_ALLOCATION)) {
    is_debug_mode_ = GetOptionValue<bool>(ctx_->options, ASTATE_DEBUG_MODE);
    numa_allocator_.Initialize();

    UpdateGlobalParallelConfig(ctx_->options);

    // Check if current env is CUDA or CPU only, If CUDA found, pinned memory is available.

    SPDLOG_INFO("Pinned memory in current env: {}", pinned_memory_enabled_);

    // Initialize copy thread pool


    SPDLOG_INFO("Enable GPU async copy: write {}, read {}", enable_write_gpu_async_copy_, enable_read_gpu_async_copy_);
    int copy_thread_num = GetOptionValue<int>(ctx_->options, TRANSFER_ENGINE_COPY_THREAD_NUM);
    long copy_bucket_mem_size = GetOptionValue<long>(ctx_->options, TRANSFER_ENGINE_COPY_BUCKET_MEM_SIZE);
    long copy_large_bucket_mem_size = GetOptionValue<long>(ctx_->options, TRANSFER_ENGINE_COPY_LARGE_BUCKET_MEM_SIZE);
    int copy_large_thread_num = GetOptionValue<int>(ctx_->options, TRANSFER_ENGINE_COPY_LARGE_THREAD_NUM);
    int copy_small_thread_num = GetOptionValue<int>(ctx_->options, TRANSFER_ENGINE_COPY_SMALL_THREAD_NUM);


    if (small_tensor_compact_cache_size_ < small_tensor_size_) {
        SPDLOG_ERROR(
            "small_tensor_compact_cache_size_ < small_tensor_size_, "
            "small_tensor_compact_cache_size_: {}, "
            "small_tensor_size_: {}",
            small_tensor_compact_cache_size_,
            small_tensor_size_);
        throw std::invalid_argument("illegal state: small_tensor_compact_cache_size_ < "
                                    "small_tensor_size_");
    }
    if (small_tensor_compact_cache_size_ > copy_bucket_mem_size) {
        SPDLOG_ERROR(
            "small_tensor_compact_cache_size_ > copy_bucket_mem_size, "
            "small_tensor_compact_cache_size_: {}, "
            "copy_bucket_mem_size: {}",
            small_tensor_compact_cache_size_,
            copy_bucket_mem_size);
        throw std::invalid_argument("illegal state: small_tensor_compact_cache_size_ > "
                                    "copy_bucket_mem_size");
    }

    perf_metrics_controller_ = std::make_shared<PerfMetricsController>("remote_tensor_table", ctx_->options);
    enable_log_tensor_meta_ = GetOptionValue<bool>(ctx_->options, TRANSFER_ENGINE_LOG_TENSOR_META);
    SPDLOG_INFO(
        "Enbale log tensor meta: {}, Enable perf metrics: {}",
        enable_log_tensor_meta_,
        perf_metrics_controller_->IsPerfMetricsEnabled());

    if (ctx_->parallel_config.IsInference()) {
        copy_thread_pool_ = std::make_unique<astate::CUDAStreamResourceThreadPool<torch::Tensor>>(
            [&, copy_bucket_mem_size]() {
                torch::Tensor tensor = CreateZeroTensor(
                    {copy_bucket_mem_size},
                    torch::ScalarType::Byte,
                    torch::DeviceType::CPU,
                    false,
                    pinned_memory_enabled_);
                ATStorage atensor_storage = TensorStorageToATStorage(tensor);
                ctx_->transfer_service->PreRegisterMemory(atensor_storage);
                return tensor;
            },
            [](torch::Tensor& tensor) { tensor.reset(); },
            copy_thread_num);
        copy_bucket_mem_size_ = copy_bucket_mem_size;

        copy_large_thread_pool_ = std::make_unique<astate::CUDAStreamResourceThreadPool<torch::Tensor>>(
            [&, copy_large_bucket_mem_size]() {
                torch::Tensor tensor = CreateZeroTensor(
                    {copy_large_bucket_mem_size},
                    torch::ScalarType::Byte,
                    torch::DeviceType::CPU,
                    false,
                    pinned_memory_enabled_);
                ATStorage atensor_storage = TensorStorageToATStorage(tensor);
                ctx_->transfer_service->PreRegisterMemory(atensor_storage);
                return tensor;
            },
            [](torch::Tensor& tensor) { tensor.reset(); },
            copy_large_thread_num);
        copy_large_bucket_mem_size_ = copy_large_bucket_mem_size;
    } else {
        thread_pool_ = std::make_unique<astate::CUDAStreamThreadPool>(copy_thread_num);
        small_tensor_compact_cache_offset_ = 0;
        small_tensor_compact_cache_ = CreateZeroTensor(
            {small_tensor_compact_cache_size_},
            torch::ScalarType::Byte,
            torch::DeviceType::CPU,
            false,
            pinned_memory_enabled_);
    }
}

bool RemoteTensorTable::Put(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& py_tensor) {
    pybind11::gil_scoped_release release;

    auto start_time = std::chrono::high_resolution_clock::now();
    try {
        // Convert py_tensor to torch::Tensor
        const torch::Tensor& source_tensor = PyObjectToTensor(py_tensor);

        auto local_copy = GetOrCreateLocalTensorCopy(tensor_key, source_tensor);

        if ((source_tensor.is_cuda() || local_copy->is_cuda()) && enable_write_gpu_async_copy_) {
            c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
            c10::cuda::CUDAStreamGuard guard(stream);
            local_copy->copy_(source_tensor, true);
            stream.synchronize();
        } else {
            local_copy->copy_(source_tensor);
        }
        std::shared_ptr<ATensor> atensor = TensorToATensor(*local_copy);
        bool result = ctx_->transfer_service->Put(seq_id, tensor_key, *atensor);

        auto end_time = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO(
            "put::total cost {} us",
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        return result;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error in put with seq_id {} and tensor_key {}: {}", seq_id, tensor_key.key, e.what());
        throw e;
    }
}

bool RemoteTensorTable::Get(int64_t seq_id, const ShardedKey& tensor_key, pybind11::object& py_tensor) {
    pybind11::gil_scoped_release release;
    try {
        // Convert py_tensor to torch::Tensor
        const torch::Tensor& target_tensor = PyObjectToTensor(py_tensor);
        std::shared_ptr<ATensor> target_atensor;
        if (target_tensor.requires_grad()) {
            target_atensor = TensorToATensor(target_tensor.detach());
        } else {
            target_atensor = TensorToATensor(target_tensor);
        }

        std::future<void> copy_future = SubmitTransferTask(seq_id, tensor_key, target_tensor);
        copy_future.get();
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error in get with seq_id {} and tensor_key {}: {}", seq_id, tensor_key.key, e.what());
        throw e;
    }
}

bool RemoteTensorTable::MultiPut(
    int64_t seq_id, const std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_list) {
    pybind11::gil_scoped_release release;
    if (is_debug_mode_) {
        SPDLOG_INFO("RemoteTensorTable::multi_put for seq_id {} with {} tensors", seq_id, tensor_list.size());
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Convert py_tensor to ATensor using local tensor copies
        std::vector<std::pair<ShardedKey, ATensor>> atensor_list;
        bool need_sync = false;
        std::vector<std::future<std::vector<ShardedATensor>>> copy_futures;
        copy_futures.reserve(tensor_list.size());
        for (const auto& pair : tensor_list) {
            copy_futures.push_back(thread_pool_->Submit([this, seq_id, &pair](
                                                            const std::shared_ptr<c10::cuda::CUDAStream>& stream) {
                auto start_time = std::chrono::high_resolution_clock::now();
                const torch::Tensor& source_tensor = PyObjectToTensor(pair.second);
                if (stream != nullptr && source_tensor.device().index() != stream->device_index()) {
                    SPDLOG_ERROR(
                        "multi_put: source_tensor device index {} does not "
                        "match stream device index {}",
                        source_tensor.device().index(),
                        stream->device_index());
                }

                std::vector<ShardedATensor> local_sharded_tensors;
                auto reshard_tensors = GetOrCreateLocalReshardTensors(pair.first, source_tensor);
                for (const auto& reshard_tensor : reshard_tensors) {
                    auto sharded_key = reshard_tensor.first;
                    auto reshard_source_tensor = reshard_tensor.second;
                    auto local_copy = GetOrCreateLocalTensorCopy(sharded_key, reshard_source_tensor);

                    auto start_copy_time = std::chrono::high_resolution_clock::now();
                    if ((reshard_source_tensor.is_cuda() || local_copy->is_cuda()) && stream != nullptr
                        && enable_write_gpu_async_copy_) {
                        c10::cuda::CUDAStreamGuard guard(*stream);
                        local_copy->copy_(reshard_source_tensor, true);
                        stream->synchronize();
                    } else {
                        local_copy->copy_(reshard_source_tensor);
                    }

                    auto end_time = std::chrono::high_resolution_clock::now();
                    if (perf_metrics_controller_->ShouldLogPerfMetric(seq_id)) {
                        SPDLOG_INFO(
                            "multi_put::tensor_key: {}, total cost {} us, "
                            "prepare cost {} us, copy cost {} us",
                            pair.first.key,
                            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(),
                            std::chrono::duration_cast<std::chrono::microseconds>(start_copy_time - start_time).count(),
                            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_copy_time).count());
                    }

                    local_sharded_tensors.emplace_back(sharded_key, TensorToATensor(*local_copy));
                }

                // return std::make_pair(pair.first, TensorToATensor(*local_copy));
                return local_sharded_tensors;
            }));
        }

        for (auto& future : copy_futures) {
            // auto pair = future.get();
            // atensor_list.emplace_back(pair.first, *pair.second);
            auto sharded_tensors = future.get();
            for (const auto& it : sharded_tensors) {
                atensor_list.emplace_back(it.first, *it.second);
            }
        }

        // cudaDeviceSynchronize();

        bool ret = ctx_->transfer_service->MultiPut(seq_id, atensor_list);

        auto end_time = std::chrono::high_resolution_clock::now();
        if (is_debug_mode_) {
            SPDLOG_INFO(
                "multi_put::total cost {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
        return ret;
    } catch (const std::exception& e) {
        auto exception_end_time = std::chrono::high_resolution_clock::now();
        SPDLOG_ERROR("Error in multi_put with seq_id {}: {}", seq_id, e.what());
        SPDLOG_ERROR(
            "multi_put::total cost {} us",
            std::chrono::duration_cast<std::chrono::microseconds>(exception_end_time - start_time).count());
        throw e;
    }
}

bool RemoteTensorTable::MultiGet(int64_t seq_id, std::vector<std::pair<ShardedKey, pybind11::object>>& tensor_dict) {
    pybind11::gil_scoped_release release;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    try {
        // Step 1: Initialize data structures
        auto step1_start = std::chrono::high_resolution_clock::now();
        std::unordered_map<ShardedKey, std::vector<std::pair<ShardedKey, torch::Tensor>>&, ShardedKeyHash>
            candidates_map;
        std::vector<std::pair<ShardedKey, ATensor>> query_list;
        auto step1_end = std::chrono::high_resolution_clock::now();
        auto step1_duration = std::chrono::duration_cast<std::chrono::microseconds>(step1_end - step1_start);
        SPDLOG_INFO("Step 1 - Initialize data structures: {} us", step1_duration.count());

        // Step 2: Convert py_objects to tensors and get candidates
        auto step2_start = std::chrono::high_resolution_clock::now();
        size_t total_tensor_size = 0;
        size_t total_small_tensor_size = 0;
        std::vector<std::future<void>> copy_futures;
        std::unordered_map<ShardedKey, const torch::Tensor&, ShardedKeyHash> small_tensors;

        auto rng = std::default_random_engine{};
        std::shuffle(tensor_dict.begin(), tensor_dict.end(), rng);
        for (auto& pair : tensor_dict) {
            const torch::Tensor& target_tensor = PyObjectToTensor(pair.second);
            SPDLOG_INFO("[REMOTETensorTable] multi get device index: {}", target_tensor.get_device());
            UpdateReadingTensorsMeta(seq_id, pair.first, target_tensor);

            size_t tensor_size = GetTensorTotalByteSize(target_tensor);
            total_tensor_size += tensor_size;
            if (tensor_size <= small_tensor_size_) {
                small_tensors.emplace(pair.first, target_tensor);
                total_small_tensor_size += tensor_size;
                continue;
            }

            copy_futures.push_back(SubmitTransferTask(seq_id, pair.first, target_tensor));
        }
        auto step2_end = std::chrono::high_resolution_clock::now();
        auto step2_duration = std::chrono::duration_cast<std::chrono::microseconds>(step2_end - step2_start);
        SPDLOG_INFO(
            "Step 2 - Convert py_objects and Submit copy tasks: {} us, "
            "copy_futures size: {}",
            step2_duration.count(),
            copy_futures.size());

        auto step3_start = std::chrono::high_resolution_clock::now();
        MultiGetCompactTensors(seq_id, small_tensors);
        auto step3_end = std::chrono::high_resolution_clock::now();
        auto step3_duration = std::chrono::duration_cast<std::chrono::microseconds>(step3_end - step3_start);
        SPDLOG_INFO("Step 3 - multi_get_compact_tensors: {} us", step3_duration.count());

        // Step 3: Wait for all copy operations to complete
        auto step4_start = std::chrono::high_resolution_clock::now();
        for (auto& future : copy_futures) {
            future.get();
        }
        auto step4_end = std::chrono::high_resolution_clock::now();
        auto step4_duration = std::chrono::duration_cast<std::chrono::microseconds>(step4_end - step4_start);
        SPDLOG_INFO("Step 4 - Wait for copy operations completion: {} us", step4_duration.count());

        // Total time calculation
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);
        SPDLOG_INFO(
            "RemoteTensorTable::multi_get completed for seq_id {} - Total "
            "tensor size: {} MB - Total small tensor "
            "size: {} MB - Total time: {} us ({:.2f} ms)",
            seq_id,
            total_tensor_size / 1024 / 1024,
            total_small_tensor_size / 1024 / 1024,
            total_duration.count(),
            total_duration.count() / 1000.0);

        return true;
    } catch (const std::exception& e) {
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);
        SPDLOG_ERROR("Error in multi_get for seq_id {} after {} us: {}", seq_id, total_duration.count(), e.what());
        throw e;
    }
}

bool RemoteTensorTable::MultiGetCompactTensors(
    int64_t seq_id, const std::unordered_map<ShardedKey, const torch::Tensor&, ShardedKeyHash>& small_tensors) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::unordered_map<ShardedKey, ATensor, ShardedKeyHash> tensor_shards;
    std::unordered_map<ShardedKey, std::vector<std::pair<ShardedKey, const torch::Tensor&>>, ShardedKeyHash>
        target_tensor_map;
    for (const auto& pair : small_tensors) {
        ShardedKey sharded_key = pair.first;
        const torch::Tensor& target_tensor = pair.second;

        const std::vector<std::tuple<ShardedKey, ShardedKey, ATensor>>& candidates
            = GetRemoteTensorShards(sharded_key, seq_id, *TensorToATensor(target_tensor), false);
        for (const std::tuple<ShardedKey, ShardedKey, ATensor>& candidate : candidates) {
            const ShardedKey& raw_sharded_key = std::get<0>(candidate);
            const ATensor& atensor = std::get<2>(candidate);
            tensor_shards.emplace(raw_sharded_key, atensor);
            target_tensor_map[raw_sharded_key].emplace_back(sharded_key, target_tensor);
        }
    }

    std::vector<CompactTensorInfo> compact_tensor_infos;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        compact_tensor_infos = compact_tensor_infos_;
        if (compact_tensor_infos.size() <= 0) {
            compact_tensor_infos = ctx_->transfer_service->GetCompactTensorInfos(seq_id, tensor_shards);
            compact_tensor_infos_ = compact_tensor_infos;
            cached_small_tensor_shards_ = tensor_shards;
        }
    }
    auto compact_tensor_infos_end = std::chrono::high_resolution_clock::now();
    auto compact_tensor_infos_duration
        = std::chrono::duration_cast<std::chrono::microseconds>(compact_tensor_infos_end - start_time);
    SPDLOG_INFO(
        "multi_get_compact_tensors::compact_tensor_infos cost {} us for {} "
        "compact tensor infos from "
        "small_tensors: {}",
        compact_tensor_infos_duration.count(),
        compact_tensor_infos.size(),
        small_tensors.size());

    auto submit_start = std::chrono::high_resolution_clock::now();
    std::vector<std::future<bool>> read_futures{};
    read_futures.reserve(compact_tensor_infos.size());
    for (const auto& compact_tensor_info : compact_tensor_infos) {
        read_futures.push_back(copy_thread_pool_->Submit(
            [&](torch::Tensor& local_cache, const std::shared_ptr<c10::cuda::CUDAStream>& stream) {
                ATStorage astorage = TensorStorageToATStorage(local_cache);
                if (!ctx_->transfer_service->RawGet(
                        seq_id,
                        astorage,
                        compact_tensor_info.node_info,
                        compact_tensor_info.addr,
                        compact_tensor_info.size)) {
                    SPDLOG_ERROR(
                        "Failed to read tensor from remote for seq_id {} and "
                        "compact_tensor_info.node_info: {}:{} and "
                        "compact_tensor_info.addr: {} and "
                        "compact_tensor_info.size: {} local_addr:{} "
                        "local_size:{}",
                        seq_id,
                        compact_tensor_info.node_info.hostname_or_ip,
                        compact_tensor_info.node_info.rdma_port,
                        compact_tensor_info.addr,
                        compact_tensor_info.size,
                        astorage.data,
                        astorage.storage_size);
                    throw std::runtime_error("Failed to read tensor from remote for seq_id " + std::to_string(seq_id));
                }

                for (const auto& pair : compact_tensor_info.atensors) {
                    const ShardedKey& shard_key = pair.first;
                    const ATensor& atensor = pair.second;

                    char* local_cache_ptr = static_cast<char*>(local_cache.data_ptr())
                        + GetStorageByteOffset(pair.second.dtype, atensor.storage_offset);

                    std::vector<int64_t> sizes(pair.second.size, pair.second.size + pair.second.dim_num);
                    std::vector<int64_t> strides(pair.second.stride, pair.second.stride + pair.second.dim_num);
                    torch::Tensor local_tensor = torch::from_blob(
                        local_cache_ptr,
                        sizes,
                        strides,
                        torch::TensorOptions()
                            .dtype(ATDtypeToTorchDtype(pair.second.dtype))
                            .device(torch::Device(torch::DeviceType::CPU))
                            .layout(torch::Layout::Strided)
                            .memory_format(torch::MemoryFormat::Contiguous)
                            .pinned_memory(pinned_memory_enabled_)
                            .requires_grad(false));

                    const auto& target_pair_list = target_tensor_map.at(shard_key);
                    for (const auto& target_pair : target_pair_list) {
                        const torch::Tensor& target_tensor = target_pair.second;
                        if (stream != nullptr && target_tensor.device().is_cuda()
                            && target_tensor.device().index() != stream->device_index()) {
                            SPDLOG_ERROR(
                                "multi_get_compact_tensors: target_tensor "
                                "device index {} not match thread pool device "
                                "index {}",
                                target_tensor.device().index(),
                                stream->device_index());
                        }

                        if (target_tensor.requires_grad()) {
                            CopyTensorWithShardedKeysUnsafe(
                                shard_key,
                                local_tensor,
                                target_pair.first,
                                target_tensor.detach(),
                                stream.get(),
                                enable_read_gpu_async_copy_);
                        } else {
                            CopyTensorWithShardedKeysUnsafe(
                                shard_key,
                                local_tensor,
                                target_pair.first,
                                target_tensor,
                                stream.get(),
                                enable_read_gpu_async_copy_);
                        }
                    }
                }
                return true;
            }));
    }
    auto submit_end = std::chrono::high_resolution_clock::now();
    auto submit_duration = std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start);
    SPDLOG_INFO(
        "multi_get_compact_tensors::Submit cost {} us for {} tasks", submit_duration.count(), read_futures.size());

    auto read_start = std::chrono::high_resolution_clock::now();
    for (auto& future : read_futures) {
        future.get();
    }
    auto read_end = std::chrono::high_resolution_clock::now();
    auto read_duration = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start);
    SPDLOG_INFO(
        "multi_get_compact_tensors::wait for read&copy completion cost {} us "
        "for {} tasks and {} small tensors",
        read_duration.count(),
        read_futures.size(),
        small_tensors.size());
    return true;
}

// New interface methods - currently not implemented for RemoteTensorTable
pybind11::object
RemoteTensorTable::GetTensor(int64_t seq_id, const ShardedKey& tensor_key, const TorchTensorMeta& tensor_meta) {
    pybind11::gil_scoped_release release;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    try {
        // Create target tensor
        torch::Tensor ret = CreateZeroTensor(tensor_meta.size, tensor_meta.dtype, tensor_meta.device.type());

        // Get remote tensor shards
        auto target_tensor = TensorToATensor(ret);

        // Submit copy task
        std::future<void> copy_future = SubmitTransferTask(seq_id, tensor_key, ret);

        // Copy data to target tensor
        copy_future.get();

        // Convert to pybind11 object
        auto result = TensorToPyObject(ret);

        // Total time calculation - 只在seq_id变化时打印
        // if (print_cuda) {
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);
        SPDLOG_INFO("get_tensor::total cost {} us", total_duration.count());
        // astate::::print_cuda_memory_summary("RemoteTensorTable::get_tensor_after_convert");
        // }

        return result;
    } catch (const std::exception& e) {
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);
        SPDLOG_ERROR(
            "Error in get_tensor for seq_id {}, tensor_key: {} after {} us: {}",
            seq_id,
            tensor_key.key,
            total_duration.count(),
            e.what());
        throw e;
    }
}

std::vector<std::pair<ShardedKey, pybind11::object>> RemoteTensorTable::MultiGetTensor(
    int64_t seq_id, const std::vector<std::pair<ShardedKey, TorchTensorMeta>>& tensor_meta_list) {
    pybind11::gil_scoped_release release;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    SPDLOG_INFO(
        "RemoteTensorTable::MultiGetTensor for seq_id {} with {} tensor "
        "metas",
        seq_id,
        tensor_meta_list.size());

    try {
        // Initialize data structures
        std::unordered_map<ShardedKey, torch::Tensor, ShardedKeyHash> tensor_map;

        // Create tensors and get candidates
        std::vector<std::future<void>> copy_futures;
        for (const auto& pair : tensor_meta_list) {
            tensor_map[pair.first] = CreateZeroTensor(pair.second.size, pair.second.dtype, pair.second.device.type());

            copy_futures.push_back(SubmitTransferTask(seq_id, pair.first, tensor_map[pair.first]));
        }

        // Wait for all copy operations to complete
        for (auto& future : copy_futures) {
            future.get();
        }

        // Convert results to pybind11 objects
        std::vector<std::pair<ShardedKey, pybind11::object>> result;
        result.reserve(tensor_map.size());
        for (auto& pair : tensor_map) {
            result.emplace_back(pair.first, TensorToPyObject(pair.second));
        }

        // Total time calculation
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);
        SPDLOG_INFO("MultiGetTensor::total cost {} us", total_duration.count());

        return result;
    } catch (const std::exception& e) {
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);
        SPDLOG_ERROR("Error in MultiGetTensor for seq_id {} after {} us: {}", seq_id, total_duration.count(), e.what());
        throw e;
    }
}

void RemoteTensorTable::Complete(int64_t seq_id) {
    pybind11::gil_scoped_release release;

    // cudaDeviceSynchronize();

    ctx_->transfer_service->Complete();
    LogReadingTensorsMeta(seq_id);

    // 重置last_logged_seq_id，为下一个seq_id做准备
    last_logged_seq_id_.store(-1);

    SPDLOG_INFO(
        "Seq {} completed. copy_task_counter_ {} copy_large_task_counter_ {}",
        seq_id,
        copy_task_counter_,
        copy_large_task_counter_);
    copy_task_counter_ = 0;
    copy_large_task_counter_ = 0;
    // SPDLOG_INFO("Seq {} completed.", seq_id);

    // For remote table, this could potentially be implemented to:
    // - Wait for all RDMA operations to complete
    // - Synchronize with remote nodes
    // - Flush any pending operations

    // clear the ready flags in local_cached_tensors_
    if (enable_local_cache_prefetch_) {
        {
            std::lock_guard<std::mutex> lock(local_cached_tensors_mutex_);
            for (auto& cached_tensor : local_cached_tensors_) {
                cached_tensor.second.second = false;
            }
        }
    }
}

std::vector<std::pair<std::string, TorchTensorMeta>> RemoteTensorTable::ScanTensorMeta(int64_t seq_id) {
    pybind11::gil_scoped_release release;
    std::vector<std::pair<std::string, TorchTensorMeta>> result;
    std::vector<std::pair<ShardedKey, ATensor>> atensor_list = ctx_->transfer_service->GetAllTensorShards(
        seq_id, [](const ShardedKey& /*sharded_key*/) -> bool { return true; });
    for (const auto& pair : atensor_list) {
        const ShardedKey& sharded_key = pair.first;
        const ATensor& atensor = pair.second;
        result.emplace_back(sharded_key.key, GetTorchTensorMeta(atensor));
    }
    return result;
}

std::shared_ptr<torch::Tensor>
RemoteTensorTable::GetLocalPrefetchCachedTensor(const ShardedKey& sharded_key, const torch::Tensor& target_tensor) {
    if (!enable_local_cache_prefetch_) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(local_cached_tensors_mutex_);

    auto cache_it = local_cached_tensors_.find(sharded_key);
    if (cache_it != local_cached_tensors_.end()) {
        // If the local cached tensor is found and ready, return it.
        if (cache_it->second.second) {
            return cache_it->second.first;
        }
    } else if (local_cached_tensors_size_ < local_cached_tensors_size_limit_) {
        // If the local cached tensor is not found and cache size is less than limit, create a new one.
        torch::Tensor local_cached_tensor = torch::empty(
            target_tensor.sizes(),
            torch::TensorOptions()
                .dtype(target_tensor.dtype())
                .device(torch::Device(torch::DeviceType::CPU))
                .layout(torch::Layout::Strided)
                .memory_format(torch::MemoryFormat::Contiguous)
                .pinned_memory(pinned_memory_enabled_)
                .requires_grad(false));
        auto atensor_storage = TensorStorageToATStorage(local_cached_tensor);
        ctx_->transfer_service->PreRegisterMemory(atensor_storage);

        std::pair<std::shared_ptr<torch::Tensor>, bool> local_tensor
            = std::make_pair(std::make_shared<torch::Tensor>(std::move(local_cached_tensor)), false);
        local_cached_tensors_.emplace(sharded_key, local_tensor);
        local_cached_tensors_size_ += static_cast<long>(GetTensorTotalByteSize(*local_tensor.first));

        SPDLOG_INFO(
            "Created new local cached tensor for shard key {}: dtype={}, "
            "sizes={}",
            sharded_key.key,
            local_tensor.first->dtype().name(),
            local_tensor.first->sizes().size());
    }

    return nullptr;
}

std::shared_ptr<TensorDict> RemoteTensorTable::ReadTensors(
    int64_t seq_id, const ShardedKey& sharded_key, const torch::Tensor& target_tensor, torch::Tensor& local_cache) {
    std::shared_ptr<TensorDict> tensors = std::make_shared<TensorDict>();

    // Part of the tensors are cached locally. For the updating of these tensors, we can directly use the
    // cached tensor for further updating, while the left tensors would be read from remote instances.
    auto local_cached_tensor = GetLocalPrefetchCachedTensor(sharded_key, target_tensor);
    if (local_cached_tensor != nullptr) {
        tensors->emplace(sharded_key, local_cached_tensor);
    } else {
        // Get remote tensor shards that need to be fetched
        const auto& remote_shards = GetRemoteTensorShards(sharded_key, seq_id, *TensorToATensor(target_tensor));

        // Step 1: Initialize data structures
        std::vector<std::pair<ShardedKey, ATensor>> remote_query_list;
        remote_query_list.reserve(remote_shards.size());

        char* local_cache_ptr = static_cast<char*>(local_cache.data_ptr());

        // Step 2: Create tensors and get candidates
        int64_t offset = 0;
        auto total_size = GetTensorTotalByteSize(local_cache);
        for (const auto& tuple : remote_shards) {
            ShardedKey raw_sharded_key = std::get<0>(tuple);
            ShardedKey adjusted_sharded_key = std::get<1>(tuple);
            const auto& atensor = std::get<2>(tuple);

            auto size = static_cast<int64_t>(GetTensorTotalByteSize(atensor));
            // Check if the local cache is large enough to hold the remote shard
            if (offset + size > total_size) {
                SPDLOG_ERROR(
                    "Copy bucket memory size is too small for tensor {} with "
                    "total size {} and current size {}",
                    sharded_key.key,
                    total_size,
                    (offset + size));
                throw std::runtime_error(
                    "Copy bucket memory size is too small for tensor " + sharded_key.key + " with total size "
                    + std::to_string(total_size) + " and current size " + std::to_string(offset + size));
            }
            std::vector<int64_t> sizes(atensor.size, atensor.size + atensor.dim_num);
            std::vector<int64_t> strides(atensor.stride, atensor.stride + atensor.dim_num);
            torch::Tensor tensor = torch::from_blob(
                local_cache_ptr + offset,
                sizes,
                strides,
                torch::TensorOptions()
                    .dtype(ATDtypeToTorchDtype(atensor.dtype))
                    .device(torch::Device(torch::DeviceType::CPU))
                    .layout(torch::Layout::Strided)
                    .memory_format(torch::MemoryFormat::Contiguous)
                    .pinned_memory(pinned_memory_enabled_)
                    .requires_grad(false));
            // TODO(root): this is a temporary solution to fix the issue that the remote tensor is not
            // aligned with the local tensor.
            //       we should fix this issue in the future.
            std::shared_ptr<ATensor> atensor_ptr = TensorToATensor(tensor);
            atensor_ptr->storage_offset = atensor.storage_offset;
            remote_query_list.emplace_back(raw_sharded_key, *atensor_ptr);
            tensors->emplace(adjusted_sharded_key, std::make_shared<torch::Tensor>(std::move(tensor)));
            offset += size;
        }

        // Step 3: Fetch all data from remote in one batch
        if (!remote_query_list.empty()) {
            // if (!ctx_->transfer_service->MultiGet(seq_id, remote_query_list)) {
            //     SPDLOG_ERROR("Failed to read tensor from remote for seq_id {}", seq_id);
            //     throw std::runtime_error("Failed to read tensor from remote for seq_id " + std::to_string(seq_id));
            // }
            for (auto query : remote_query_list) {
                if (!ctx_->transfer_service->Get(seq_id, query.first, query.second)) {
                    SPDLOG_ERROR("Failed to read tensor from remote for seq_id {}: {}", seq_id, query.first.ToString());
                    throw std::runtime_error("Failed to read tensor from remote for seq_id " + std::to_string(seq_id));
                }
            }
        }
    }

    return tensors;
}

std::future<void> RemoteTensorTable::SubmitTransferTask(
    int64_t seq_id, const ShardedKey& sharded_key, const torch::Tensor& target_tensor) {
    auto copy_task = [seq_id, sharded_key, &target_tensor, this](
                         torch::Tensor& local_cache, const std::shared_ptr<c10::cuda::CUDAStream>& stream) mutable {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Read tensors from local cache or remote instances
        std::shared_ptr<TensorDict> tensors = ReadTensors(seq_id, sharded_key, target_tensor, local_cache);

        auto copy_time = std::chrono::high_resolution_clock::now();
        if (stream != nullptr && target_tensor.device().is_cuda()
            && target_tensor.device().index() != stream->device_index()) {
            SPDLOG_ERROR(
                "target_tensor device index {} not match thread pool "
                "device index {}",
                target_tensor.device().index(),
                stream->device_index());
        }

        for (const auto& pair : *tensors) {
            CopyTensorWithShardedKeysUnsafe(
                pair.first, *pair.second, sharded_key, target_tensor, stream.get(), enable_read_gpu_async_copy_);
        }
        // cudaDeviceSynchronize();

        // Reset the tensors
        // for (auto& pair : *tensors) {
        //    pair.second->reset();
        //}

        auto end_time = std::chrono::high_resolution_clock::now();
        if (perf_metrics_controller_->ShouldLogPerfMetric(seq_id)) {
            SPDLOG_INFO(
                "SubmitTransferTask::tensor_key: {}, total cost {} us, "
                "read cost {} us, copy cost {} us",
                sharded_key.key,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(),
                std::chrono::duration_cast<std::chrono::microseconds>(copy_time - start_time).count(),
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - copy_time).count());
        }
    };

    size_t cache_size_needed = GetTensorTotalByteSize(target_tensor) * ctx_->parallel_config.tp_size;
    if (cache_size_needed <= this->copy_bucket_mem_size_) {
        copy_task_counter_++;
        return copy_thread_pool_->Submit(copy_task);
    }
    if (cache_size_needed <= this->copy_large_bucket_mem_size_) {
        copy_large_task_counter_++;
        return copy_large_thread_pool_->Submit(copy_task);
    }
    SPDLOG_WARN(
        "{} with size might be needed {} and copy large bucket mem size {}",
        "Copy bucket memory size might be too small for tensor " + sharded_key.key,
        cache_size_needed,
        this->copy_large_bucket_mem_size_);
    copy_large_task_counter_++;
    return copy_large_thread_pool_->Submit(copy_task);
}

std::vector<ReshardingInfo>
RemoteTensorTable::ReshardTensor(const ShardedKey& tensor_key, const torch::Tensor& source_tensor) const {
    std::vector<ReshardingInfo> ret;

    // TODO(root): Following is the temp solution to reshard the tensor in column dimension by configuration.
    // It is supposed to find the the resharding strategy according to the training and inference tensor layouts
    // automatically in the furture.
    int infer_tp_size = inference_parallel_config_.tp_size;
    if (infer_tp_size <= 0) {
        return ret; // Train tp size or infer tp size not configured.
    }

    int dim_index = 1; // only consider the resharding in column dimension currently
    size_t origin_size = source_tensor.size(dim_index);
    size_t split_size = tensor_key.global_shape[dim_index] / infer_tp_size;
    auto split_num = origin_size / split_size;
    SPDLOG_INFO(
        "Start to reshard tensor {}: dim_index={}, origin_size={}, "
        "split_num={}, split_size={}",
        tensor_key.ToString(),
        dim_index,
        origin_size,
        split_num,
        split_size);

    for (auto i = 0; i < split_num; i++) {
        ReshardingInfo reshard_info;
        ShardedKey reshard_key = tensor_key;
        reshard_key.global_offset[dim_index] += static_cast<int64_t>(i * split_size);
        SetShardedKey(reshard_info, reshard_key);
        SetDimIndex(reshard_info, dim_index);
        SetStart(reshard_info, i * split_size);
        SetOffset(reshard_info, split_size);
        SPDLOG_INFO(
            "Reshard info: key={}, dim_index={}, start={}, offset={}",
            GetShardedKey(reshard_info).ToString(),
            GetDimIndex(reshard_info),
            GetStart(reshard_info),
            GetOffset(reshard_info));

        ret.emplace_back(reshard_info);
    }

    return ret;
}

std::unordered_map<ShardedKey, torch::Tensor, ShardedKeyHash>
RemoteTensorTable::GetOrCreateLocalReshardTensors(const ShardedKey& tensor_key, const torch::Tensor& source_tensor) {
    std::unordered_map<ShardedKey, torch::Tensor, ShardedKeyHash> ret;

    if (!IsTensorReshardingKey(tensor_key) || GetTensorTotalByteSize(source_tensor) <= small_tensor_size_) {
        ret.emplace(tensor_key, source_tensor);
    } else {
        std::vector<ReshardingInfo> reshard_infos;

        {
            RWSpinGuard lock(rw_spin_lock_, true);
            auto it = tensor_resharding_map_.find(tensor_key);
            if (it != tensor_resharding_map_.end()) {
                reshard_infos = it->second;
            } else {
                reshard_infos = ReshardTensor(tensor_key, source_tensor);
                tensor_resharding_map_[tensor_key] = reshard_infos;
            }
        }

        if (reshard_infos.size() > 0) {
            for (const auto& reshard : reshard_infos) {
                auto reshard_tensor = source_tensor.narrow(GetDimIndex(reshard), GetStart(reshard), GetOffset(reshard));
                ret.emplace(GetShardedKey(reshard), std::move(reshard_tensor));
            }
        } else {
            ret.emplace(tensor_key, source_tensor);
        }
    }

    return ret;
}

std::shared_ptr<torch::Tensor>
RemoteTensorTable::GetOrCreateLocalTensorCopy(const ShardedKey& tensor_key, const torch::Tensor& source_tensor) {
    // std::lock_guard<std::mutex> lock(mutex_);
    {
        RWSpinGuard lock(rw_spin_lock_, false);
        auto it = local_tensor_mapping_.find(tensor_key);
        if (it != local_tensor_mapping_.end()) {
            return it->second.second;
        }
    }

    {
        RWSpinGuard lock(rw_spin_lock_, true);

        // Check again in case another thread has created the tensor copy
        auto it = local_tensor_mapping_.find(tensor_key);
        if (it != local_tensor_mapping_.end()) {
            return it->second.second;
        }

        auto src_sizes = source_tensor.sizes();
        size_t dim_num = src_sizes.size();
        std::vector<int64_t> size_array(src_sizes.begin(), src_sizes.end());
        torch::IntArrayRef sizes{size_array.data(), dim_num};

        std::shared_ptr<torch::Tensor> tensor_copy;
        auto tensor_storage_size = GetTensorTotalByteSize(source_tensor);
        if (tensor_storage_size > small_tensor_size_) {
            tensor_copy = std::make_shared<torch::Tensor>(CreateZeroTensor(
                sizes, source_tensor.dtype().toScalarType(), torch::DeviceType::CPU, false, pinned_memory_enabled_));
        } else {
            small_tensor_compact_cache_offset_ = static_cast<long>(
                AlignStorageOffset(small_tensor_compact_cache_offset_, source_tensor.dtype().toScalarType()));
            if (small_tensor_compact_cache_offset_ + tensor_storage_size > small_tensor_compact_cache_size_) {
                small_tensor_compact_cache_list_.push_back(small_tensor_compact_cache_);
                small_tensor_compact_cache_offset_ = 0;
                small_tensor_compact_cache_ = CreateZeroTensor(
                    {small_tensor_compact_cache_size_},
                    torch::ScalarType::Byte,
                    torch::DeviceType::CPU,
                    false,
                    pinned_memory_enabled_);
            }

            int64_t numel = GetTensorTotalSize(source_tensor);
            int64_t element_size = static_cast<int64_t>(GetItemSizeFromDtype(source_tensor.dtype().toScalarType()));
            int64_t element_offset = small_tensor_compact_cache_offset_ / element_size;
            tensor_copy
                = std::make_shared<torch::Tensor>(torch::from_blob(
                                                      static_cast<char*>(small_tensor_compact_cache_.data_ptr()),
                                                      {small_tensor_compact_cache_size_ / element_size},
                                                      torch::TensorOptions()
                                                          .dtype(source_tensor.dtype().toScalarType())
                                                          .device(torch::DeviceType::CPU)
                                                          .requires_grad(false)
                                                          .pinned_memory(pinned_memory_enabled_))
                                                      .slice(0, element_offset, element_offset + numel)
                                                      .view(sizes));
            small_tensor_compact_cache_offset_ += static_cast<long>(tensor_storage_size);
        }

        ShardedKey key_copy = ShardedKey{tensor_key};
        local_tensor_mapping_[key_copy] = std::make_pair(key_copy, tensor_copy);
        return tensor_copy;
    }
}

const std::vector<ShardedATensorTuple>& RemoteTensorTable::GetRemoteTensorShards(
    const ShardedKey& sharded_key, int64_t seq_id, const ATensor& target_tensor, bool try_prune_redundant_shard) {
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = shard_mapping_.find(sharded_key);
        if (it != shard_mapping_.end()) {
            return it->second;
        }
    }

    std::shared_ptr<std::vector<std::pair<ShardedKey, ATensor>>> tensor_meta_list
        = std::atomic_load(&tensor_meta_list_);
    if (tensor_meta_list == nullptr) {
        std::lock_guard<std::mutex> lock(tensor_meta_mutex_);
        if (tensor_meta_list_ == nullptr) {
            std::atomic_store(
                &tensor_meta_list_,
                std::make_shared<std::vector<std::pair<ShardedKey, ATensor>>>(
                    ctx_->transfer_service->GetAllTensorShards(
                        seq_id, [](const ShardedKey& /*candidate*/) -> bool { return true; })));
        }
        tensor_meta_list = tensor_meta_list_;
    }

    std::vector<std::pair<ShardedKey, ATensor>> atensor_list{};
    for (const auto& pair : *tensor_meta_list) {
        if (pair.first.key == sharded_key.key) {
            atensor_list.emplace_back(pair);
        }
    }
    auto target_candidates = astate::FindCoveringCandidates(atensor_list, sharded_key, target_tensor);
    std::vector<ShardedATensorTuple> ret;
    for (const auto& op : target_candidates) {
        auto& atensor = atensor_list[op.candidate_index].second;
        ShardedKey adjusted_sharded_key{atensor_list[op.candidate_index].first};
        if (try_prune_redundant_shard) {
            TryAdjustGlobalOffset(atensor, op.copy_shape, op.src_offset, adjusted_sharded_key);
        }
        ret.emplace_back(atensor_list[op.candidate_index].first, adjusted_sharded_key, atensor);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);

        shard_mapping_.emplace(sharded_key, ret);
        auto it = shard_mapping_.find(sharded_key);
        if (it != shard_mapping_.end()) {
            return it->second;
        }
        throw std::runtime_error("Failed to get remote tensor shards for seq_id " + std::to_string(seq_id));
    }
}

const std::vector<ShardedATensorTuple>&
RemoteTensorTable::GetRemoteTensorShards(const ShardedKey& sharded_key, int64_t seq_id, const ATensor& target_tensor) {
    return GetRemoteTensorShards(sharded_key, seq_id, target_tensor, true);
}

void RemoteTensorTable::PrefetchCachedTensors(int64_t seq_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    try {
        TensorDict prefetch_tensors;
        {
            std::lock_guard<std::mutex> lock(local_cached_tensors_mutex_);
            for (auto& cached_tensor_iter : local_cached_tensors_) {
                if (!cached_tensor_iter.second.second) {
                    prefetch_tensors.emplace(cached_tensor_iter.first, cached_tensor_iter.second.first);
                }
            }
        }

        // update the local cached tensors
        std::vector<std::future<void>> copy_futures;
        for (auto& tensor_iter : prefetch_tensors) {
            copy_futures.push_back(SubmitTransferTask(seq_id, tensor_iter.first, *tensor_iter.second));
        }

        for (auto& future : copy_futures) {
            future.get();
        }

        // set the ready flag of the local cached tensors to true
        {
            std::lock_guard<std::mutex> lock(local_cached_tensors_mutex_);
            for (auto& cached_tensor_iter : local_cached_tensors_) {
                cached_tensor_iter.second.second = true;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        SPDLOG_INFO(
            "Prefetched cached {} tensors for seq_id {} in {} ms",
            local_cached_tensors_.size(),
            seq_id,
            duration.count());
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to prefetch cached tensors for seq_id {}: {}", seq_id, e.what());
    }
}

torch::Tensor RemoteTensorTable::CreateZeroTensor(
    torch::IntArrayRef sizes,
    torch::ScalarType dtype,
    torch::DeviceType device_type,
    bool requires_grad,
    bool pinned_memory) {
    // Calculate the size of the tensor in bytes
    size_t element_size = GetItemSizeFromDtype(dtype);
    size_t total_elements = 1;
    for (int64_t size : sizes) {
        total_elements *= size;
    }
    size_t total_size = total_elements * element_size;
    // Try NUMA-aware allocation if enabled
    if (enable_numa_allocation_ && device_type == torch::DeviceType::CPU) {
        try {
            auto allocation_result = numa_allocator_.TryAllocateNearCurrentDevice(total_size);
            if (allocation_result.success) {
                // Successfully allocated with NUMA awareness
                // Create tensor from the allocated memory
                void* data_ptr = allocation_result.ptr;
                auto deleter = [this, allocation_result](void*) {
                    // Custom deleter to properly deallocate NUMA memory
                    numa_allocator_.Deallocate(allocation_result.ptr, allocation_result.size);
                };

                // Create tensor options
                auto options = torch::TensorOptions()
                                   .dtype(dtype)
                                   .device(device_type)
                                   .requires_grad(requires_grad)
                                   .pinned_memory(pinned_memory)
                                   .memory_format(torch::MemoryFormat::Contiguous);

                // Create tensor from blob with custom deleter
                return torch::from_blob(data_ptr, sizes, deleter, options);
            }
            // If allocation failed, fall back to default allocation
            SPDLOG_WARN("NUMA-aware allocation failed, falling back to default "
                        "allocation");
        } catch (const std::exception& e) {
            // If allocation throws an exception, fall back to default allocation
            SPDLOG_WARN(
                "NUMA-aware allocation threw exception: {}, falling back to "
                "default allocation",
                e.what());
        }
    }

    // Default allocation method
    return torch::zeros(
        sizes,
        torch::TensorOptions()
            .dtype(dtype)
            .device(device_type)
            .requires_grad(requires_grad)
            .pinned_memory(pinned_memory)
            .memory_format(torch::MemoryFormat::Contiguous));
}
} // namespace astate
