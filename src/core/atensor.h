#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <memory.h>

#include <torch/types.h>

#include "astate/config.h"

namespace astate {

enum class ATDeviceType : int8_t {
    UNKNOWN_DEV = -1,
    CPU = 0,
    CUDA = 1,
};

enum class ATDtype : int8_t {
    Undefined = -1,
    Byte = 0,
    Char = 1,
    Short = 2,
    Int = 3,
    Long = 4,
    Half = 5,
    Float = 6,
    Double = 7,
    ComplexHalf = 8,
    ComplexFloat = 9,
    ComplexDouble = 10,
    Bool = 11,
    QInt8 = 12,
    QUInt8 = 13,
    QInt32 = 14,
    BFloat16 = 15,
    QUInt4x2 = 16,
    QUInt2x4 = 17,
    Bits1x8 = 18,
    Bits2x4 = 19,
    Bits4x2 = 20,
    Bits8 = 21,
    Bits16 = 22,
    Float8_e5m2 = 23,
    Float8_e4m3fn = 24,
    Float8_e5m2fnuz = 25,
    Float8_e4m3fnuz = 26,
    UInt16 = 27,
    UInt32 = 28,
    UInt64 = 29,
    UInt1 = 30,
    UInt2 = 31,
    UInt3 = 32,
    UInt4 = 33,
    UInt5 = 34,
    UInt6 = 35,
    UInt7 = 36,
    Int1 = 37,
    Int2 = 38,
    Int3 = 39,
    Int4 = 40,
    Int5 = 41,
    Int6 = 42,
    Int7 = 43,
    Float8_e8m0fnu = 44,
};

using ATDeviceIndex = int8_t;

struct ATDevice {
    ATDeviceType device_type = ATDeviceType::UNKNOWN_DEV;
    ATDeviceIndex device_index = 0;

    ATDevice() = default;
    ATDevice(const ATDevice& other) = default;

    ATDevice(ATDeviceType device_type, ATDeviceIndex device_index)
        : device_type(device_type),
          device_index(device_index) {}

    ATDevice& operator=(const ATDevice& other) = default;

    static int GetATDeviceSize() { return sizeof(ATDeviceType) + sizeof(ATDeviceIndex); }

    [[nodiscard]] std::string GetDeviceInfo() const {
        return "ATDevice(device_type=" + std::to_string(static_cast<int>(device_type))
            + ", device_index=" + std::to_string(device_index) + ")";
    }
};

struct ATStorage {
    // Metadata for the tensor
    size_t storage_size{};
    ATDevice device = {ATDeviceType::UNKNOWN_DEV, 0};

    // Pointer to the tensor data
    // Actually, torch::Tensor.data_ptr() = storage_.mutable_data() + data_type_.itemsize() * storage_offset_
    void* data = nullptr;

    ATStorage() = default;
    ATStorage(const ATStorage& other) = default;

    ~ATStorage() = default;

    ATStorage(size_t size, void* data, ATDevice device)
        : storage_size(size),
          device(device),
          data(data) {}

    ATStorage& operator=(const ATStorage& other) = default;

    static size_t GetStorageMetaSize() { return sizeof(size_t) + astate::ATDevice::GetATDeviceSize(); }

    [[nodiscard]] size_t GetStorageDataSize() const { return storage_size; }

    void MallocData(size_t data_size) {
        if (device.device_type == ATDeviceType::CPU) {
            if (data != nullptr) {
                free(data);
                data = nullptr;
            }
            data = malloc(data_size);
        } else if (device.device_type == ATDeviceType::CUDA) {
            if (data != nullptr) {
                cudaFree(data);
                data = nullptr;
            }
            cudaMalloc(&(data), data_size);
        } else {
            // SPDLOG_WARN("Unsupported device type for storage data malloc.");
        }
    }

    [[nodiscard]] std::string GetStorageInfo() const {
        return "ATStorage(storage_size=" + std::to_string(storage_size) + ", device=" + device.GetDeviceInfo() + ")";
    }

    [[nodiscard]] bool IsValid() const {
        return storage_size > 0 && device.device_type != ATDeviceType::UNKNOWN_DEV
            && (device.device_type == ATDeviceType::CPU || (device.device_index >= 0 && device.device_index < 10))
            && data != nullptr;
    }
};

struct ATensor {
    int64_t* size = nullptr;
    int64_t* stride = nullptr;
    int64_t storage_offset = 0;
    int32_t dim_num{};
    ATDtype dtype = ATDtype::Undefined;

    bool conj{};
    bool neg{};
    bool requires_grad{}; // optional

    // Tensor storage
    ATStorage storage;

    ATensor() = default;

    // Copy constructor - deep copy
    ATensor(const ATensor& other)
        : storage_offset(other.storage_offset),
          dim_num(other.dim_num),
          dtype(other.dtype),
          conj(other.conj),
          neg(other.neg),
          requires_grad(other.requires_grad),
          storage(other.storage) {
        if (other.size != nullptr && dim_num > 0) {
            size = new int64_t[dim_num];
            memcpy(size, other.size, sizeof(int64_t) * dim_num);
        } else {
            size = nullptr;
        }

        if (other.stride != nullptr && dim_num > 0) {
            stride = new int64_t[dim_num];
            memcpy(stride, other.stride, sizeof(int64_t) * dim_num);
        } else {
            stride = nullptr;
        }
    }

    // Move constructor
    ATensor(ATensor&& other) noexcept
        : size(other.size),
          stride(other.stride),
          storage_offset(other.storage_offset),
          dim_num(other.dim_num),
          dtype(other.dtype),
          conj(other.conj),
          neg(other.neg),
          requires_grad(other.requires_grad),
          storage(other.storage) {
        // Take ownership of pointers
        other.size = nullptr;
        other.stride = nullptr;
        other.dim_num = 0;
    }

    // Move assignment operator
    ATensor& operator=(ATensor&& other) noexcept {
        if (this != &other) {
            // Clean up existing memory

            delete[] size;


            delete[] stride;


            // Take ownership
            size = other.size;
            stride = other.stride;
            storage_offset = other.storage_offset;
            dim_num = other.dim_num;
            dtype = other.dtype;
            conj = other.conj;
            neg = other.neg;
            requires_grad = other.requires_grad;
            storage = other.storage;

            // Reset other object
            other.size = nullptr;
            other.stride = nullptr;
            other.dim_num = 0;
        }
        return *this;
    }

    ~ATensor() {
        // Free size and stride arrays
        if (size != nullptr) {
            delete[] size;
            size = nullptr;
        }
        if (stride != nullptr) {
            delete[] stride;
            stride = nullptr;
        }
    }

    ATensor(
        int64_t* size,
        int64_t* stride,
        int64_t storage_offset,
        int32_t dim_num,
        ATDtype dtype,
        bool conj,
        bool neg,
        ATStorage storage,
        bool req_grad = false)
        : size(new int64_t[dim_num]),
          stride(new int64_t[dim_num]),
          storage_offset(storage_offset),
          dim_num(dim_num),
          dtype(dtype),
          conj(conj),
          neg(neg),
          requires_grad(req_grad),
          storage(storage) {
        memcpy(this->size, size, GetSizeSize());
        memcpy(this->stride, stride, GetStrideSize());
    }

    // Assignment operator - deep copy
    ATensor& operator=(const ATensor& other) {
        if (this == &other) {
            return *this; // Self-assignment check
        }

        // Clean up existing memory
        if (size != nullptr) {
            delete[] size;
            size = nullptr;
        }
        if (stride != nullptr) {
            delete[] stride;
            stride = nullptr;
        }

        // Copy primitive fields
        storage_offset = other.storage_offset;
        dim_num = other.dim_num;
        dtype = other.dtype;
        conj = other.conj;
        neg = other.neg;
        requires_grad = other.requires_grad;
        storage = other.storage;

        // Deep copy size and stride arrays
        if (other.size != nullptr && dim_num > 0) {
            size = new int64_t[dim_num];
            memcpy(size, other.size, sizeof(int64_t) * dim_num);
        } else {
            size = nullptr;
        }

        if (other.stride != nullptr && dim_num > 0) {
            stride = new int64_t[dim_num];
            memcpy(stride, other.stride, sizeof(int64_t) * dim_num);
        } else {
            stride = nullptr;
        }

        return *this;
    }

    [[nodiscard]] size_t GetSizeSize() const { return sizeof(int64_t) * dim_num; }

    [[nodiscard]] size_t GetStrideSize() const { return sizeof(int64_t) * dim_num; }

    [[nodiscard]] size_t GetTotalMetaSize() const {
        return (sizeof(int64_t) * dim_num) // size
            + (sizeof(int64_t) * dim_num) // stride
            + sizeof(int64_t) // storage_offset
            + sizeof(int32_t) // dim_num
            + sizeof(int32_t) // dtype
            + (sizeof(bool) * 3) // conj + neg + requires_grad
            + astate::ATStorage::GetStorageMetaSize(); // storage meta part
    }

    [[nodiscard]] size_t GetTotalSize() const { return GetTotalMetaSize() + storage.GetStorageDataSize(); }

    [[nodiscard]] ATDeviceType GetDeviceType() const { return storage.device.device_type; }

    [[nodiscard]] std::string GetSizesInfo() const {
        std::ostringstream oss;
        oss << "[";
        if (dim_num > 0) {
            oss << size[0];
            for (int32_t i = 1; i < dim_num; ++i) {
                oss << ", " << size[i];
            }
        }
        oss << "]";
        return oss.str();
    }

    [[nodiscard]] std::string GetStridesInfo() const {
        std::ostringstream oss;
        oss << "[";
        if (dim_num > 0) {
            oss << stride[0];
            for (int32_t i = 1; i < dim_num; ++i) {
                oss << ", " << stride[i];
            }
        }
        oss << "]";
        return oss.str();
    }

    [[nodiscard]] std::string GetTensorInfo() const {
        return GetTensorMetaInfo() + ", storage=" + storage.GetStorageInfo();
    }

    [[nodiscard]] std::string GetTensorMetaInfo() const {
        return "ATensor(sizes=" + GetSizesInfo() + ", strides=" + GetStridesInfo() + ", storage_offset="
            + std::to_string(storage_offset) + ", dtype=" + std::to_string(static_cast<int>(dtype))
            + ", conj=" + std::to_string(conj) + ", neg=" + std::to_string(static_cast<int>(neg))
            + ", requires_grad=" + std::to_string(static_cast<int>(requires_grad)) + ")";
    }

    [[nodiscard]] bool IsShapeEqual(const ATensor& other) const {
        return std::equal(size, size + dim_num, other.size) && std::equal(stride, stride + dim_num, other.stride);
    }

    [[nodiscard]] bool IsValid() const {
        return size != nullptr && stride != nullptr && dim_num > 0 && dtype != ATDtype::Undefined && storage_offset >= 0
            && storage.IsValid();
    }

    [[nodiscard]] ATensor Clone(bool only_meta = false) const {
        ATensor ret = ATensor(*this);
        if (!only_meta) {
            ret.storage.data = nullptr;
            ret.storage.storage_size = 0;
            ret.storage.device = {ATDeviceType::UNKNOWN_DEV, 0};
        }
        return ret;
    }
};

// Add operator<< for ATensor
inline std::ostream& operator<<(std::ostream& os, const ATensor& tensor) {
    os << tensor.GetTensorInfo();
    return os;
}

// ARole的辅助函数
inline std::string RoleToString(ARole role) {
    return role == ARole::TRAINING ? "TRAINING" : "INFERENCE";
}

inline ARole FromString(const std::string& str) {
    return str == "TRAINING" ? ARole::TRAINING : ARole::INFERENCE;
}

// 为ARole添加流输出操作符重载
inline std::ostream& operator<<(std::ostream& os, ARole role) {
    os << RoleToString(role);
    return os;
}

struct GlobalParallelConfig {
    int32_t dp_size{0};
    int32_t tp_size{0};
    int32_t pp_size{0};
    int32_t cp_size{0};
    int32_t ep_size{0};
    int32_t etp_size{0};

    [[nodiscard]] std::string ToString() const {
        return "GlobalParallelConfig(dp=" + std::to_string(dp_size) + ", tp=" + std::to_string(tp_size)
            + ", pp=" + std::to_string(pp_size) + ", cp=" + std::to_string(cp_size) + ", ep=" + std::to_string(ep_size)
            + ", etp=" + std::to_string(etp_size) + ")";
    }
};

} // namespace astate
