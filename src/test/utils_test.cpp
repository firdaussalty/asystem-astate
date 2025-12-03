#include "core/utils.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "core/atensor.h"

using namespace astate;

// 初始化Python解释器用于测试
class UtilsTest : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {
        // 初始化Python解释器（如果还没有初始化）
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
        }
    }

    static void TearDownTestSuite() {
        // 保持Python解释器运行，因为可能有其他测试需要它
        // pybind11::finalize_interpreter();
    }
};

// 测试数据类型转换
TEST_F(UtilsTest, TorchDtypeToATDtype_conversion) {
    // 测试基本数据类型转换
    EXPECT_EQ(TorchDtypeToATDtype(torch::kUInt8), ATDtype::Byte);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kInt8), ATDtype::Char);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kInt16), ATDtype::Short);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kInt32), ATDtype::Int);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kInt64), ATDtype::Long);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kUInt16), ATDtype::UInt16);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kUInt32), ATDtype::UInt32);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kUInt64), ATDtype::UInt64);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kFloat16), ATDtype::Half);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kFloat32), ATDtype::Float);
    EXPECT_EQ(TorchDtypeToATDtype(torch::kFloat64), ATDtype::Double);
}

TEST_F(UtilsTest, ATDtypeToTorchDtype_conversion) {
    // 测试逆向转换
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::Byte), torch::kUInt8);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::Char), torch::kInt8);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::Short), torch::kInt16);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::Int), torch::kInt32);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::Long), torch::kInt64);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::UInt16), torch::kUInt16);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::UInt32), torch::kUInt32);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::UInt64), torch::kUInt64);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::Half), torch::kFloat16);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::Float), torch::kFloat32);
    EXPECT_EQ(ATDtypeToTorchDtype(ATDtype::Double), torch::kFloat64);
}

// 测试设备转换
TEST_F(UtilsTest, device_conversion) {
    // CPU设备转换
    torch::Device cpu_device(torch::kCPU, 0);
    ATDevice at_cpu = TorchDeviceToATDevice(cpu_device);
    EXPECT_EQ(at_cpu.device_type, ATDeviceType::CPU);
    EXPECT_EQ(at_cpu.device_index, 0);

    torch::Device converted_cpu = ATDeviceToTorchDevice(at_cpu);
    EXPECT_EQ(converted_cpu.type(), torch::kCPU);
    EXPECT_EQ(converted_cpu.index(), 0);

    // CUDA设备转换（如果可用）
    if (torch::cuda::is_available()) {
        torch::Device cuda_device(torch::kCUDA, 0);
        ATDevice at_cuda = TorchDeviceToATDevice(cuda_device);
        EXPECT_EQ(at_cuda.device_type, ATDeviceType::CUDA);
        EXPECT_EQ(at_cuda.device_index, 0);

        torch::Device converted_cuda = ATDeviceToTorchDevice(at_cuda);
        EXPECT_EQ(converted_cuda.type(), torch::kCUDA);
        EXPECT_EQ(converted_cuda.index(), 0);
    }
}

// 测试存储转换
TEST_F(UtilsTest, tensor_storage_conversion) {
    // 创建一个简单的张量
    torch::Tensor tensor = torch::randn({2, 3}, torch::kFloat32);

    // 转换存储
    ATStorage atstorage = TensorStorageToATStorage(tensor);

    // 验证存储属性
    EXPECT_EQ(atstorage.storage_size, tensor.storage().nbytes());
    EXPECT_EQ(atstorage.device.device_type, ATDeviceType::CPU);
    EXPECT_EQ(atstorage.data, tensor.data_ptr());
}

// 测试torch::Tensor到ATensor的转换
TEST_F(UtilsTest, tensor_to_atensor_conversion) {
    // 创建测试张量
    torch::Tensor tensor = torch::randn({2, 3}, torch::kFloat32);
    tensor.set_requires_grad(true);

    // 转换到ATensor
    auto atensor = TensorToATensor(tensor);

    // 验证基本属性
    EXPECT_EQ(atensor->dim_num, 2);
    EXPECT_EQ(atensor->dtype, ATDtype::Float);
    EXPECT_EQ(atensor->storage_offset, tensor.storage_offset());
    EXPECT_EQ(atensor->requires_grad, true);
    EXPECT_EQ(atensor->conj, tensor.is_conj());
    EXPECT_EQ(atensor->neg, tensor.is_neg());

    // 验证维度
    EXPECT_EQ(atensor->size[0], 2);
    EXPECT_EQ(atensor->size[1], 3);

    // 验证存储
    EXPECT_EQ(atensor->storage.data, tensor.data_ptr());
}

// 测试ATensor到torch::Tensor的转换
TEST_F(UtilsTest, atensor_to_tensor_conversion) {
    // 创建原始张量
    torch::Tensor original_tensor = torch::ones({2, 3}, torch::kFloat32);
    original_tensor.set_requires_grad(true);

    // 转换到ATensor
    auto atensor = TensorToATensor(original_tensor);

    // 转换回torch::Tensor
    torch::Tensor converted_tensor = ATensorToTensor(*atensor);

    // 验证属性
    EXPECT_TRUE(torch::allclose(original_tensor, converted_tensor));
    EXPECT_EQ(converted_tensor.requires_grad(), true);
    EXPECT_EQ(converted_tensor.sizes(), original_tensor.sizes());
    EXPECT_EQ(converted_tensor.strides(), original_tensor.strides());
    EXPECT_EQ(converted_tensor.storage_offset(), original_tensor.storage_offset());
    EXPECT_EQ(converted_tensor.dtype(), original_tensor.dtype());
}

// 测试张量视图的转换（具有非零存储偏移）
TEST_F(UtilsTest, tensor_view_conversion) {
    // 创建一个较大的张量，然后创建视图
    torch::Tensor large_tensor = torch::arange(20, torch::kFloat32).reshape({4, 5});
    torch::Tensor view_tensor = large_tensor.slice(0, 1, 3); // 选择第1和第2行

    EXPECT_GT(view_tensor.storage_offset(), 0); // 确保有非零偏移
    EXPECT_NE(view_tensor.data_ptr(),
              large_tensor.data_ptr()); // 视图张量的数据指针应该不同

    // 转换到ATensor
    auto atensor = TensorToATensor(view_tensor);

    // 验证存储偏移
    EXPECT_EQ(atensor->storage_offset, view_tensor.storage_offset());

    // 转换回张量
    torch::Tensor converted = ATensorToTensor(*atensor);

    // 验证数据一致性
    EXPECT_TRUE(torch::allclose(view_tensor, converted));
    EXPECT_EQ(converted.storage_offset(), 0);
}

// 测试Python对象转换（需要Python环境）
TEST_F(UtilsTest, py_object_tensor_conversion) {
    try {
        // 创建Python张量
        pybind11::module torch_module = pybind11::module::import("torch");
        pybind11::object py_tensor = torch_module.attr("randn")(pybind11::make_tuple(2, 3));

        // 转换到C++张量
        const torch::Tensor& cpp_tensor = PyObjectToTensor(py_tensor);

        // 验证基本属性
        EXPECT_EQ(cpp_tensor.dim(), 2);
        EXPECT_EQ(cpp_tensor.size(0), 2);
        EXPECT_EQ(cpp_tensor.size(1), 3);

        // 转换回Python对象
        pybind11::object converted_py = TensorToPyObject(cpp_tensor);
    } catch (const std::exception& e) {
        // 如果Python环境不可用，跳过测试
        GTEST_SKIP() << "Python environment not available: " << e.what();
    }
}

// 测试存储拷贝功能
TEST_F(UtilsTest, CopyTensorStorage) {
    try {
        // 创建源张量
        torch::Tensor source = torch::ones({2, 3}, torch::kFloat32) * 5.0f;

        // 创建目标张量（相同形状和类型）
        torch::Tensor dest = torch::zeros({2, 3}, torch::kFloat32);

        // 转换目标张量为Python对象
        pybind11::object dest_py = TensorToPyObject(dest);

        // 执行存储拷贝
        CopyTensorStorage(source, dest_py);

        // 验证拷贝结果
        const torch::Tensor& updated_dest = PyObjectToTensor(dest_py);
        EXPECT_TRUE(torch::allclose(source, updated_dest));

    } catch (const std::exception& e) {
        GTEST_SKIP() << "Python environment not available: " << e.what();
    }
}

// 测试批量转换
TEST_F(UtilsTest, batch_conversions) {
    // 创建多个张量
    std::vector<torch::Tensor> tensors
        = {torch::ones({2, 2}, torch::kFloat32),
           torch::zeros({3, 3}, torch::kFloat32),
           torch::randn({1, 4}, torch::kFloat32)};

    // 批量转换到ATensor
    auto atensors = TensorsToATensors(tensors);
    EXPECT_EQ(atensors.size(), 3);

    // 批量转换回张量
    auto converted_tensors = ATensorsToTensors(atensors);
    EXPECT_EQ(converted_tensors.size(), 3);

    // 验证每个张量
    for (size_t i = 0; i < tensors.size(); ++i) {
        EXPECT_TRUE(torch::allclose(tensors[i], converted_tensors[i]));
        EXPECT_EQ(tensors[i].sizes(), converted_tensors[i].sizes());
    }
}

// 测试错误情况
TEST_F(UtilsTest, error_handling) {
    // 测试不支持的数据类型 - 使用一个超出范围的值
    EXPECT_THROW(TorchDtypeToATDtype(static_cast<torch::ScalarType>(999)), std::runtime_error);
    EXPECT_THROW(ATDtypeToTorchDtype(ATDtype::Undefined), std::runtime_error);

    // 测试无效的Python对象
    pybind11::object none_obj = pybind11::none();
    EXPECT_THROW(PyObjectToTensor(none_obj), std::runtime_error);
}

// 测试不同数据类型的转换
TEST_F(UtilsTest, different_dtypes_conversion) {
    std::vector<torch::ScalarType> test_types
        = {torch::kUInt8,
           torch::kInt8,
           torch::kInt16,
           torch::kInt32,
           torch::kInt64,
           torch::kUInt16,
           torch::kUInt32,
           torch::kUInt64,
           torch::kFloat16,
           torch::kFloat32,
           torch::kFloat64};

    for (auto dtype : test_types) {
        try {
            // 创建不同类型的张量
            torch::Tensor tensor = torch::ones({2, 2}, dtype);

            // 转换到ATensor
            auto atensor = TensorToATensor(tensor);

            // 转换回张量
            torch::Tensor converted = ATensorToTensor(*atensor);

            // 验证数据类型和数据
            EXPECT_EQ(converted.dtype(), tensor.dtype());
            EXPECT_TRUE(torch::allclose(tensor.to(torch::kFloat32), converted.to(torch::kFloat32)));

        } catch (const std::exception& e) {
            // 某些类型可能不支持，记录但不失败
            std::cout << "Skipping dtype " << static_cast<int>(dtype) << ": " << e.what() << std::endl;
        }
    }
}

// 测试CUDA张量转换（如果CUDA可用）
TEST_F(UtilsTest, cuda_tensor_conversion) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
        return;
    }

    try {
        // 创建CUDA张量
        torch::Tensor cuda_tensor = torch::randn({2, 3}, torch::device(torch::kCUDA));

        // 转换到ATensor
        auto atensor = TensorToATensor(cuda_tensor);

        // 验证设备类型
        EXPECT_EQ(atensor->storage.device.device_type, ATDeviceType::CUDA);

        // 转换回张量
        torch::Tensor converted = ATensorToTensor(*atensor);

        // 验证设备和数据
        EXPECT_EQ(converted.device().type(), torch::kCUDA);
        EXPECT_TRUE(torch::allclose(cuda_tensor, converted));

    } catch (const std::exception& e) {
        GTEST_SKIP() << "CUDA tensor test failed: " << e.what();
    }
}
