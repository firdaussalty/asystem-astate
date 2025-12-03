#include "transport/atensor_serializer.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory.h>

#include <spdlog/spdlog.h>
#include <torch/cuda.h>

#include "core/atensor.h"

namespace astate {

// Tensor meta is supposed to be stored in host(cpu) while data could be stored in device(gpu).
// So currently, we only need to consider the case that meta info is serialized/deserialized in memory.
size_t SerializeTensorMeta(const ATensor& atensor, void* buffer) {
    char* current = static_cast<char*>(buffer);
    SERIALIZE_FIELD_H2H(current, &(atensor.dim_num), sizeof(int32_t))
    SERIALIZE_FIELD_H2H(current, atensor.size, atensor.GetSizeSize())
    SERIALIZE_FIELD_H2H(current, atensor.stride, atensor.GetStrideSize())
    SERIALIZE_FIELD_H2H(current, &(atensor.dtype), sizeof(ATDtype))
    SERIALIZE_FIELD_H2H(current, &(atensor.conj), sizeof(bool))
    SERIALIZE_FIELD_H2H(current, &(atensor.neg), sizeof(bool))
    SERIALIZE_FIELD_H2H(current, &(atensor.requires_grad), sizeof(bool))
    SERIALIZE_FIELD_H2H(current, &(atensor.storage.storage_size), sizeof(int32_t))
    SERIALIZE_FIELD_H2H(current, &(atensor.storage.device.device_index), sizeof(ATDeviceIndex))
    SERIALIZE_FIELD_H2H(current, &(atensor.storage.device.device_type), sizeof(ATDeviceType))
    return atensor.GetTotalMetaSize();
}

std::shared_ptr<ATensor> DeserializeTensorMeta(void* buffer) {
    std::shared_ptr<ATensor> atensor_ptr = std::make_shared<ATensor>(ATensor{});
    char* current = static_cast<char*>(buffer);

    // Deserialize meta info
    // First deserialize dim_num to know the size of size and stride arrays
    DESERIALIZE_FIELD_H2H(&(atensor_ptr->dim_num), current, sizeof(int32_t))

    // Allocate memory for size and stride arrays
    atensor_ptr->size = new int64_t[atensor_ptr->dim_num];
    atensor_ptr->stride = new int64_t[atensor_ptr->dim_num];
    DESERIALIZE_FIELD_H2H(atensor_ptr->size, current, atensor_ptr->GetSizeSize())
    DESERIALIZE_FIELD_H2H(atensor_ptr->stride, current, atensor_ptr->GetStrideSize())

    // Now deserialize the rest of the meta info
    DESERIALIZE_FIELD_H2H(&(atensor_ptr->dtype), current, sizeof(ATDtype))
    DESERIALIZE_FIELD_H2H(&(atensor_ptr->conj), current, sizeof(bool))
    DESERIALIZE_FIELD_H2H(&(atensor_ptr->neg), current, sizeof(bool))
    DESERIALIZE_FIELD_H2H(&(atensor_ptr->requires_grad), current, sizeof(bool))
    DESERIALIZE_FIELD_H2H(&(atensor_ptr->storage.storage_size), current, sizeof(int32_t))
    DESERIALIZE_FIELD_H2H(&(atensor_ptr->storage.device.device_index), current, sizeof(ATDeviceIndex))
    DESERIALIZE_FIELD_H2H(&(atensor_ptr->storage.device.device_type), current, sizeof(ATDeviceType))
    return atensor_ptr;
}

std::string SerializeTensorMeta(const ATensor& atensor) {
    size_t meta_size = atensor.GetTotalMetaSize();
    std::string result;
    result.resize(meta_size);

    SerializeTensorMeta(atensor, result.data());
    return result;
}

std::shared_ptr<ATensor> DeserializeTensorMeta(const std::string& data) {
    if (data.empty()) {
        SPDLOG_ERROR("DeserializeTensorMeta failed: input data is empty");
        return nullptr;
    }

    return DeserializeTensorMeta(data);
}

cudaMemcpyKind GetCudaMemcpyKind(bool from_device, bool to_device) {
    cudaMemcpyKind kind{};
    if (from_device) {
        kind = to_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    } else {
        kind = to_device ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    }
    return kind;
}

// Tensor data is supposed to be stored in host or device.
size_t SerializeTensorData(const ATensor& atensor, void* buffer, bool to_device) {
    size_t serialize_size = 0;

    // Serilize the tensor data into buffer
    ATDeviceType device = atensor.GetDeviceType();
    if (device == ATDeviceType::CPU) {
        cudaMemcpyKind kind = GetCudaMemcpyKind(device == ATDeviceType::CUDA, to_device);
        if (kind == cudaMemcpyHostToHost) {
            memcpy(buffer, atensor.storage.data, atensor.storage.GetStorageDataSize());
        } else if (kind == cudaMemcpyHostToDevice) {
            cudaMemcpy(buffer, atensor.storage.data, atensor.storage.GetStorageDataSize(), kind);
        }
        serialize_size += atensor.storage.GetStorageDataSize();
    } else if (device == ATDeviceType::CUDA) {
        cudaMemcpyKind kind = GetCudaMemcpyKind(device == ATDeviceType::CUDA, to_device);
        cudaMemcpy(buffer, atensor.storage.data, atensor.storage.GetStorageDataSize(), kind);
        serialize_size += atensor.storage.GetStorageDataSize();
    } else {
        SPDLOG_WARN("Unsupported device type for serialization.");
    }
    return serialize_size;
}

void DeserializeTensorData(ATensor& atensor, void* buffer, bool from_device) {
    ATDeviceType device = atensor.GetDeviceType();
    size_t data_size = atensor.storage.GetStorageDataSize();
    if (device == ATDeviceType::CPU) {
        cudaMemcpyKind kind = GetCudaMemcpyKind(from_device, device == ATDeviceType::CUDA);
        atensor.storage.MallocData(data_size);
        if (kind == cudaMemcpyHostToHost) {
            memcpy(atensor.storage.data, buffer, data_size);
        } else if (kind == cudaMemcpyDeviceToHost) {
            cudaMemcpy(atensor.storage.data, buffer, data_size, cudaMemcpyDeviceToHost);
        }
    } else if (device == ATDeviceType::CUDA) {
        cudaMemcpyKind kind = GetCudaMemcpyKind(from_device, device == ATDeviceType::CUDA);
        atensor.storage.MallocData(data_size);
        cudaMemcpy(atensor.storage.data, buffer, data_size, kind);
    } else {
        SPDLOG_WARN("Unsupported device type for deserialization.");
    }
}
} // namespace astate
