#pragma once

#include <cstdint>
#include <exception>
#include <stdexcept>

#include <c10/util/typeid.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "core/atensor.h"
#include "core/shardedkey.h"

namespace astate {

/**
 * @brief Convert py::object to torch::Tensor
 * @param py_obj Python object that should be a PyTorch tensor
 * @return torch::Tensor Converted PyTorch C++ Tensor
 * @throws std::runtime_error if conversion fails
 */
inline const torch::Tensor& PyObjectToTensor(const pybind11::object& py_obj) {
    try {
        // Check if object is None
        if (py_obj.is_none()) {
            throw std::runtime_error("Cannot convert None to torch::Tensor");
        }

        // Use PyTorch's THPVariable_Unpack function for conversion
        // This is the standard method PyTorch uses internally to convert from Python tensor to C++ tensor
        return THPVariable_Unpack(reinterpret_cast<THPVariable*>(py_obj.ptr()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert py::object to torch::Tensor: " + std::string(e.what()));
    }
}

/**
 * @brief Convert torch::Tensor to py::object
 * @param tensor PyTorch C++ Tensor
 * @return pybind11::object Converted Python object
 * @throws std::runtime_error if conversion fails
 */
inline pybind11::object TensorToPyObject(const torch::Tensor& tensor) {
    try {
        // Use PyTorch's THPVariable_Wrap function for conversion
        // This is the standard method PyTorch uses internally to convert from C++ tensor to Python tensor
        PyObject* py_tensor = THPVariable_Wrap(tensor);

        if (py_tensor == nullptr) {
            throw std::runtime_error("THPVariable_Wrap returned null pointer");
        }

        // Wrap PyObject* into pybind11::object
        // steal parameter indicates we take ownership of the object
        return pybind11::reinterpret_steal<pybind11::object>(py_tensor);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert torch::Tensor to py::object: " + std::string(e.what()));
    }
}

/**
 * @brief Copy storage data from source torch::Tensor to destination py::object tensor
 * @param source_tensor Source PyTorch C++ Tensor to copy data from
 * @param dest_py_obj Destination Python object that can be converted to torch::Tensor
 * @throws std::runtime_error if copy operation fails
 */
inline void CopyTensorStorage(const torch::Tensor& source_tensor, pybind11::object& dest_py_obj) {
    try {
        // Convert destination py::object to torch::Tensor
        const torch::Tensor& dest_tensor = PyObjectToTensor(dest_py_obj);

        // Validate tensors compatibility
        if (source_tensor.numel() != dest_tensor.numel()) {
            throw std::runtime_error(
                "Source and destination tensors must have the same number of "
                "elements. "
                "Source: "
                + std::to_string(source_tensor.numel()) + ", Destination: " + std::to_string(dest_tensor.numel()));
        }

        if (source_tensor.dtype() != dest_tensor.dtype()) {
            throw std::runtime_error(
                "Source and destination tensors must have the same data type. "
                "Source: "
                + std::to_string(static_cast<int>(source_tensor.scalar_type()))
                + ", Destination: " + std::to_string(static_cast<int>(dest_tensor.scalar_type())));
        }

        // Check if both tensors are on the same device
        if (source_tensor.device() != dest_tensor.device()) {
            // If devices are different, copy with device transfer
            dest_tensor.copy_(source_tensor, /*non_blocking=*/false);
        } else {
            // If devices are the same, perform direct memory copy
            dest_tensor.copy_(source_tensor);
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to copy tensor storage: " + std::string(e.what()));
    }
}

/**
 * @brief Get the itemsize of a tensor
 * @param tensor PyTorch C++ Tensor
 * @return int64_t Itemsize of the tensor
 */
inline size_t GetTensorItemSize(const torch::Tensor& tensor) {
    return tensor.dtype().itemsize();
}

/**
 * @brief Get the itemsize of a dtype
 * @param dtype PyTorch dtype
 * @return int64_t Itemsize of the dtype
 */
inline size_t GetItemSizeFromDtype(torch::ScalarType dtype) {
    return caffe2::TypeMeta::fromScalarType(dtype).itemsize();
}

/**
 * @brief Convert torch::ScalarType to ATDtype
 *        Reference: pytorch/torch/csrc/api/include/torch/types.h
 * @param scalar_type PyTorch scalar type
 * @return astate::ATDtype Corresponding ATensor data type
 * @throws std::runtime_error if conversion fails
 */
inline astate::ATDtype TorchDtypeToATDtype(torch::ScalarType scalar_type) {
    switch (scalar_type) {
        case torch::ScalarType::Byte:
            return astate::ATDtype::Byte;
        case torch::ScalarType::Char:
            return astate::ATDtype::Char;
        case torch::ScalarType::Short:
            return astate::ATDtype::Short;
        case torch::ScalarType::Int:
            return astate::ATDtype::Int;
        case torch::ScalarType::Long:
            return astate::ATDtype::Long;
        case torch::ScalarType::Half:
            return astate::ATDtype::Half;
        case torch::ScalarType::Float:
            return astate::ATDtype::Float;
        case torch::ScalarType::Double:
            return astate::ATDtype::Double;
        case torch::ScalarType::ComplexHalf:
            return astate::ATDtype::ComplexHalf;
        case torch::ScalarType::ComplexFloat:
            return astate::ATDtype::ComplexFloat;
        case torch::ScalarType::ComplexDouble:
            return astate::ATDtype::ComplexDouble;
        case torch::ScalarType::Bool:
            return astate::ATDtype::Bool;
        case torch::ScalarType::QInt8:
            return astate::ATDtype::QInt8;
        case torch::ScalarType::QUInt8:
            return astate::ATDtype::QUInt8;
        case torch::ScalarType::QInt32:
            return astate::ATDtype::QInt32;
        case torch::ScalarType::BFloat16:
            return astate::ATDtype::BFloat16;
        case torch::ScalarType::QUInt4x2:
            return astate::ATDtype::QUInt4x2;
        case torch::ScalarType::QUInt2x4:
            return astate::ATDtype::QUInt2x4;
        case torch::ScalarType::Bits1x8:
            return astate::ATDtype::Bits1x8;
        case torch::ScalarType::Bits2x4:
            return astate::ATDtype::Bits2x4;
        case torch::ScalarType::Bits4x2:
            return astate::ATDtype::Bits4x2;
        case torch::ScalarType::Bits8:
            return astate::ATDtype::Bits8;
        case torch::ScalarType::Bits16:
            return astate::ATDtype::Bits16;
        case torch::ScalarType::Float8_e5m2:
            return astate::ATDtype::Float8_e5m2;
        case torch::ScalarType::Float8_e4m3fn:
            return astate::ATDtype::Float8_e4m3fn;
        case torch::ScalarType::Float8_e5m2fnuz:
            return astate::ATDtype::Float8_e5m2fnuz;
        case torch::ScalarType::Float8_e4m3fnuz:
            return astate::ATDtype::Float8_e4m3fnuz;
        case torch::ScalarType::UInt16:
            return astate::ATDtype::UInt16;
        case torch::ScalarType::UInt32:
            return astate::ATDtype::UInt32;
        case torch::ScalarType::UInt64:
            return astate::ATDtype::UInt64;
        case torch::ScalarType::UInt1:
            return astate::ATDtype::UInt1;
        case torch::ScalarType::UInt2:
            return astate::ATDtype::UInt2;
        case torch::ScalarType::UInt3:
            return astate::ATDtype::UInt3;
        case torch::ScalarType::UInt4:
            return astate::ATDtype::UInt4;
        case torch::ScalarType::UInt5:
            return astate::ATDtype::UInt5;
        case torch::ScalarType::UInt6:
            return astate::ATDtype::UInt6;
        case torch::ScalarType::UInt7:
            return astate::ATDtype::UInt7;
        case torch::ScalarType::Int1:
            return astate::ATDtype::Int1;
        case torch::ScalarType::Int2:
            return astate::ATDtype::Int2;
        case torch::ScalarType::Int3:
            return astate::ATDtype::Int3;
        case torch::ScalarType::Int4:
            return astate::ATDtype::Int4;
        case torch::ScalarType::Int5:
            return astate::ATDtype::Int5;
        case torch::ScalarType::Int6:
            return astate::ATDtype::Int6;
        case torch::ScalarType::Int7:
            return astate::ATDtype::Int7;
        default:
            throw std::runtime_error("Unsupported torch::ScalarType: " + std::to_string(static_cast<int>(scalar_type)));
    }
}

/**
 * @brief Convert ATDtype to torch::ScalarType
 * @param atdtype ATensor data type
 * @return torch::ScalarType Corresponding PyTorch scalar type
 * @throws std::runtime_error if conversion fails
 */
inline torch::ScalarType ATDtypeToTorchDtype(astate::ATDtype atdtype) {
    switch (atdtype) {
        case astate::ATDtype::Byte:
            return torch::ScalarType::Byte;
        case astate::ATDtype::Char:
            return torch::ScalarType::Char;
        case astate::ATDtype::Short:
            return torch::ScalarType::Short;
        case astate::ATDtype::Int:
            return torch::ScalarType::Int;
        case astate::ATDtype::Long:
            return torch::ScalarType::Long;
        case astate::ATDtype::Half:
            return torch::ScalarType::Half;
        case astate::ATDtype::Float:
            return torch::ScalarType::Float;
        case astate::ATDtype::Double:
            return torch::ScalarType::Double;
        case astate::ATDtype::ComplexHalf:
            return torch::ScalarType::ComplexHalf;
        case astate::ATDtype::ComplexFloat:
            return torch::ScalarType::ComplexFloat;
        case astate::ATDtype::ComplexDouble:
            return torch::ScalarType::ComplexDouble;
        case astate::ATDtype::Bool:
            return torch::ScalarType::Bool;
        case astate::ATDtype::QInt8:
            return torch::ScalarType::QInt8;
        case astate::ATDtype::QUInt8:
            return torch::ScalarType::QUInt8;
        case astate::ATDtype::QInt32:
            return torch::ScalarType::QInt32;
        case astate::ATDtype::BFloat16:
            return torch::ScalarType::BFloat16;
        case astate::ATDtype::QUInt4x2:
            return torch::ScalarType::QUInt4x2;
        case astate::ATDtype::QUInt2x4:
            return torch::ScalarType::QUInt2x4;
        case astate::ATDtype::Bits1x8:
            return torch::ScalarType::Bits1x8;
        case astate::ATDtype::Bits2x4:
            return torch::ScalarType::Bits2x4;
        case astate::ATDtype::Bits4x2:
            return torch::ScalarType::Bits4x2;
        case astate::ATDtype::Bits8:
            return torch::ScalarType::Bits8;
        case astate::ATDtype::Bits16:
            return torch::ScalarType::Bits16;
        case astate::ATDtype::Float8_e5m2:
            return torch::ScalarType::Float8_e5m2;
        case astate::ATDtype::Float8_e4m3fn:
            return torch::ScalarType::Float8_e4m3fn;
        case astate::ATDtype::Float8_e5m2fnuz:
            return torch::ScalarType::Float8_e5m2fnuz;
        case astate::ATDtype::Float8_e4m3fnuz:
            return torch::ScalarType::Float8_e4m3fnuz;
        case astate::ATDtype::UInt16:
            return torch::ScalarType::UInt16;
        case astate::ATDtype::UInt32:
            return torch::ScalarType::UInt32;
        case astate::ATDtype::UInt64:
            return torch::ScalarType::UInt64;
        case astate::ATDtype::UInt1:
            return torch::ScalarType::UInt1;
        case astate::ATDtype::UInt2:
            return torch::ScalarType::UInt2;
        case astate::ATDtype::UInt3:
            return torch::ScalarType::UInt3;
        case astate::ATDtype::UInt4:
            return torch::ScalarType::UInt4;
        case astate::ATDtype::UInt5:
            return torch::ScalarType::UInt5;
        case astate::ATDtype::UInt6:
            return torch::ScalarType::UInt6;
        case astate::ATDtype::UInt7:
            return torch::ScalarType::UInt7;
        case astate::ATDtype::Int1:
            return torch::ScalarType::Int1;
        case astate::ATDtype::Int2:
            return torch::ScalarType::Int2;
        case astate::ATDtype::Int3:
            return torch::ScalarType::Int3;
        case astate::ATDtype::Int4:
            return torch::ScalarType::Int4;
        case astate::ATDtype::Int5:
            return torch::ScalarType::Int5;
        case astate::ATDtype::Int6:
            return torch::ScalarType::Int6;
        case astate::ATDtype::Int7:
            return torch::ScalarType::Int7;
        default:
            throw std::runtime_error("Unsupported ATDtype: " + std::to_string(static_cast<int>(atdtype)));
    }
}

/**
 * @brief Convert torch::Device to ATDevice
 * @param device PyTorch device
 * @return astate::ATDevice Corresponding ATensor device
 * @throws std::runtime_error if conversion fails
 */
inline astate::ATDevice TorchDeviceToATDevice(const torch::Device& device) {
    astate::ATDeviceType device_type{};
    switch (device.type()) {
        case torch::DeviceType::CPU:
            device_type = astate::ATDeviceType::CPU;
            break;
        case torch::DeviceType::CUDA:
            device_type = astate::ATDeviceType::CUDA;
            break;
        default:
            throw std::runtime_error(
                "Unsupported torch::DeviceType: " + std::to_string(static_cast<int>(device.type())));
    }

    return {device_type, static_cast<astate::ATDeviceIndex>(device.index())};
}

/**
 * @brief Convert ATDevice to torch::Device
 * @param atdevice ATensor device
 * @return torch::Device Corresponding PyTorch device
 * @throws std::runtime_error if conversion fails
 */
inline torch::Device ATDeviceToTorchDevice(const astate::ATDevice& atdevice) {
    torch::DeviceType device_type{};
    switch (atdevice.device_type) {
        case astate::ATDeviceType::CPU:
            device_type = torch::DeviceType::CPU;
            break;
        case astate::ATDeviceType::CUDA:
            device_type = torch::DeviceType::CUDA;
            break;
        default:
            throw std::runtime_error(
                "Unsupported ATDeviceType: " + std::to_string(static_cast<int>(atdevice.device_type)));
    }

    return {device_type, static_cast<torch::DeviceIndex>(atdevice.device_index)};
}

/**
 * @brief Get the storage offset in bytes
 * @param dtype PyTorch scalar type
 * @param storage_offset Storage offset
 * @return int64_t Storage offset in bytes
 */
inline size_t GetStorageByteOffset(const torch::ScalarType dtype, const int64_t storage_offset) {
    return storage_offset * GetItemSizeFromDtype(dtype);
}

/**
 * @brief Get the storage offset in bytes
 * @param dtype ATensor data type
 * @param storage_offset Storage offset
 * @return int64_t Storage offset in bytes
 */
inline size_t GetStorageByteOffset(const astate::ATDtype dtype, const int64_t storage_offset) {
    return storage_offset * GetItemSizeFromDtype(ATDtypeToTorchDtype(dtype));
}

/**
 * @brief Get the total element size of a tensor
 * @param atensor ATensor
 * @return int64_t Total element size of the tensor
 */
inline int64_t GetTensorTotalSize(const astate::ATensor& atensor) {
    int64_t total_size = 1;
    for (int32_t i = 0; i < atensor.dim_num; ++i) {
        total_size *= atensor.size[i];
    }
    return total_size;
}

/**
 * @brief Get the total element size of a tensor
 * @param tensor PyTorch C++ Tensor
 * @return int64_t Total element size of the tensor
 */
inline int64_t GetTensorTotalSize(const torch::Tensor& tensor) {
    return tensor.numel();
}

/**
 * @brief Get the total byte size of a tensor
 * @param atensor ATensor
 * @return int64_t Total byte size of the tensor
 */
inline size_t GetTensorTotalByteSize(const astate::ATensor& atensor) {
    return GetTensorTotalSize(atensor) * GetItemSizeFromDtype(ATDtypeToTorchDtype(atensor.dtype));
}

/**
 * @brief Get the total size of a tensor in bytes
 * @param tensor PyTorch C++ Tensor
 * @return int64_t Total size of the tensor in bytes
 */
inline size_t GetTensorTotalByteSize(const torch::Tensor& tensor) {
    return GetTensorTotalSize(tensor) * GetItemSizeFromDtype(tensor.scalar_type());
}

/**
 * @brief Convert torch::Tensor storage to ATStorage
 * @param tensor PyTorch C++ Tensor
 * @return astate::ATStorage Converted ATensor storage
 * @throws std::runtime_error if conversion fails
 */
inline astate::ATStorage TensorStorageToATStorage(const torch::Tensor& tensor) {
    try {
        // Convert device
        astate::ATDevice atdevice = TorchDeviceToATDevice(tensor.device());

        // Get storage information from tensor
        const auto& untyped_storage = tensor.storage();

        // Create ATStorage with proper offset and size
        astate::ATStorage atstorage(untyped_storage.nbytes(), untyped_storage.mutable_data(), atdevice);

        return atstorage;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert tensor storage to ATStorage: " + std::string(e.what()));
    }
}

inline bool IsContiguousShard(const astate::ATensor& atensor, const std::vector<int64_t>& copy_shape) {
    if (atensor.dim_num == copy_shape.size()) {
        int64_t stride = 1;
        for (int32_t i = atensor.dim_num - 1; i > 0; --i) {
            if (atensor.size[i] != copy_shape[i] || atensor.stride[i] != stride) {
                return false;
            }
            stride *= atensor.size[i];
        }
        return true;
    }
    return false;
}

inline bool TryAdjustGlobalOffset(
    astate::ATensor& atensor,
    const std::vector<int64_t>& copy_shape,
    const std::vector<int64_t>& copy_offset,
    astate::ShardedKey& sharded_key) {
    if (IsContiguousShard(atensor, copy_shape)) {
        for (int32_t i = 0; i < atensor.dim_num; ++i) {
            atensor.size[i] = copy_shape[i];
        }
        atensor.storage_offset += atensor.stride[0] * copy_offset[0];
        sharded_key.global_offset[0] += copy_offset[0];
        return true;
    }
    return false;
}

/**
 * @brief Convert ATStorage to torch::Tensor compatible storage information
 * @param atstorage ATensor storage
 * @param sizes Tensor sizes for creating the tensor
 * @param strides Tensor strides for creating the tensor
 * @param dtype Tensor data type
 * @param device Tensor device
 * @return torch::Tensor Tensor created from ATStorage
 * @throws std::runtime_error if conversion fails
 *
 * @note (TODO: echo.zxj) Currently, the torch::Tensor storage release mechanism
 * is not resolved. To avoid double free issues, this method allocates a separate
 * storage space and copies data from ATStorage to the newly allocated space.
 * This will be optimized to zero-copy approach in the future.
 */
inline torch::Tensor ATStorageToTensor(
    const astate::ATStorage& atstorage,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    const int64_t storage_offset,
    torch::ScalarType dtype,
    const torch::Device& device) {
    try {
        // Create tensor options
        torch::TensorOptions options
            = torch::TensorOptions().dtype(dtype).device(device).layout(torch::Layout::Strided);

        torch::Tensor tensor = torch::zeros(sizes, options);

        // Create tensor from adjusted data pointer
        torch::Tensor tmp_tensor = torch::from_blob(atstorage.data, sizes, strides, options);
        tmp_tensor.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
        tensor.copy_(tmp_tensor);

        // Free the temporary tensor
        tmp_tensor.reset();

        return tensor;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert ATStorage to tensor: " + std::string(e.what()));
    }
}

/**
 * @brief Convert torch::Tensor to ATensor
 * @param tensor PyTorch C++ Tensor
 * @return std::shared_ptr<astate::ATensor> Converted ATensor
 * @throws std::runtime_error if conversion fails
 */
inline std::shared_ptr<astate::ATensor> TensorToATensor(const torch::Tensor& tensor) {
    try {
        // Get tensor properties
        auto sizes = tensor.sizes();
        auto strides = tensor.strides();
        int64_t storage_offset = tensor.storage_offset();
        auto dim_num = static_cast<int32_t>(sizes.size());

        // Convert data type
        astate::ATDtype atdtype = TorchDtypeToATDtype(tensor.scalar_type());

        // Create size and stride arrays
        std::vector<int64_t> size_array(sizes.begin(), sizes.end());
        std::vector<int64_t> stride_array(strides.begin(), strides.end());

        for (int32_t i = 0; i < dim_num; ++i) {
            size_array[i] = sizes[i];
            stride_array[i] = strides[i];
        }

        // Convert storage using the dedicated method
        astate::ATStorage storage = TensorStorageToATStorage(tensor);

        // Create ATensor
        auto atensor = std::make_shared<astate::ATensor>(
            size_array.data(),
            stride_array.data(),
            storage_offset,
            dim_num,
            atdtype,
            tensor.is_conj(), // conj
            tensor.is_neg(), // neg
            storage,
            tensor.requires_grad());

        return atensor;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert torch::Tensor to ATensor: " + std::string(e.what()));
    }
}

/**
 * @brief Convert ATensor to torch::Tensor
 * @param atensor ATensor to convert
 * @return torch::Tensor Converted PyTorch C++ Tensor
 * @throws std::runtime_error if conversion fails
 */
inline torch::Tensor ATensorToTensor(const astate::ATensor& atensor) {
    try {
        // Convert data type
        torch::ScalarType scalar_type = ATDtypeToTorchDtype(atensor.dtype);

        // Convert device
        torch::Device device = ATDeviceToTorchDevice(atensor.storage.device);

        // Create size and stride vectors
        std::vector<int64_t> sizes(atensor.size, atensor.size + atensor.dim_num);
        std::vector<int64_t> strides(atensor.stride, atensor.stride + atensor.dim_num);

        // Use the dedicated storage conversion method
        torch::Tensor tensor
            = ATStorageToTensor(atensor.storage, sizes, strides, atensor.storage_offset, scalar_type, device);

        // Set requires_grad if needed
        if (atensor.requires_grad) {
            tensor.set_requires_grad(true);
        }

        // Apply conjugate and negation if needed
        if (atensor.conj) {
            tensor = tensor.conj();
        }
        if (atensor.neg) {
            tensor = tensor.neg();
        }

        return tensor;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to convert ATensor to torch::Tensor: " + std::string(e.what()));
    }
}

/**
 * @brief Check if py::object is a valid tensor object
 * @param py_obj Python object
 * @return bool Returns true if it's a valid tensor object, false otherwise
 */
inline bool IsTensorObject(const pybind11::object& py_obj) {
    try {
        if (py_obj.is_none()) {
            return false;
        }

        // Try to convert, if successful it means it's a valid tensor object
        PyObjectToTensor(py_obj);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

/**
 * @brief Batch convert torch::Tensor array to ATensor array
 * @param tensors PyTorch C++ Tensor array
 * @return std::vector<std::shared_ptr<astate::ATensor>> Converted ATensor array
 */
inline std::vector<std::shared_ptr<astate::ATensor>> TensorsToATensors(const std::vector<torch::Tensor>& tensors) {
    std::vector<std::shared_ptr<astate::ATensor>> atensors;
    atensors.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        atensors.push_back(TensorToATensor(tensor));
    }

    return atensors;
}

/**
 * @brief Batch convert ATensor array to torch::Tensor array
 * @param atensors ATensor array
 * @return std::vector<torch::Tensor> Converted PyTorch C++ Tensor array
 */
inline std::vector<torch::Tensor> ATensorsToTensors(const std::vector<std::shared_ptr<astate::ATensor>>& atensors) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(atensors.size());

    for (const auto& atensor : atensors) {
        tensors.push_back(ATensorToTensor(*atensor));
    }

    return tensors;
}

inline astate::TorchTensorMeta GetTorchTensorMeta(const astate::ATensor& atensor) {
    return {
        ATDtypeToTorchDtype(atensor.dtype),
        std::vector<int64_t>(atensor.size, atensor.size + atensor.dim_num),
        ATDeviceToTorchDevice(atensor.storage.device)};
}

} // namespace astate
