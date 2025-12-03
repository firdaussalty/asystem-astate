#include "transport/atensor_serializer.h"

#include <gtest/gtest.h>

#include "core/atensor.h"

using namespace astate;

TEST(atensor_seriliazer_test, serialize_deserialize_meta) {
    // 1. 构造一个 ATensor 对象
    int64_t size[] = {2, 3};
    int64_t stride[] = {3, 1};
    int32_t dim_num = 2;
    ATDtype dtype = ATDtype::Float;
    bool conj = false;
    bool neg = false;
    bool requires_grad = true;
    ATStorage storage(24, nullptr, ATDevice(ATDeviceType::CPU, 0)); // 假设 24 字节数据

    ATensor atensor(size, stride, 0, dim_num, dtype, conj, neg, storage, requires_grad);

    // 2. 分配 buffer 用于序列化
    std::vector<char> buffer(atensor.GetTotalMetaSize());

    // 3. 序列化
    int serializeSize = SerializeTensorMeta(atensor, buffer.data());
    EXPECT_EQ(serializeSize, atensor.GetTotalMetaSize());

    // 4. 反序列化
    auto deserialized = DeserializeTensorMeta(buffer.data());

    // 5. 验证反序列化后的 meta 信息是否一致
    EXPECT_EQ(deserialized->dim_num, atensor.dim_num);
    EXPECT_EQ(deserialized->dtype, atensor.dtype);
    EXPECT_EQ(deserialized->storage_offset, atensor.storage_offset);
    EXPECT_EQ(deserialized->conj, atensor.conj);
    EXPECT_EQ(deserialized->neg, atensor.neg);
    EXPECT_EQ(deserialized->requires_grad, atensor.requires_grad);

    // 验证 size 和 stride
    for (int i = 0; i < dim_num; ++i) {
        EXPECT_EQ(deserialized->size[i], atensor.size[i]);
        EXPECT_EQ(deserialized->stride[i], atensor.stride[i]);
    }

    // 验证 storage meta
    EXPECT_EQ(deserialized->storage.storage_size, atensor.storage.storage_size);
    EXPECT_EQ(deserialized->storage.device.device_type, atensor.storage.device.device_type);
    EXPECT_EQ(deserialized->storage.device.device_index, atensor.storage.device.device_index);
}

TEST(atensor_seriliazer_test, serialize_deserialize_data_cpu) {
    // 1. 构造一个 CPU ATensor 对象
    int64_t size[] = {2, 3};
    int64_t stride[] = {3, 1};
    int32_t dim_num = 2;
    ATDtype dtype = ATDtype::Float;
    bool conj = false;
    bool neg = false;
    bool requires_grad = true;

    // 分配并初始化数据
    float* data = new float[6]; // 2x3 的 float 数组
    for (int i = 0; i < 6; i++) {
        data[i] = static_cast<float>(i);
    }
    ATStorage storage(6 * sizeof(float), data, ATDevice(ATDeviceType::CPU, 0));

    ATensor atensor(size, stride, 0, dim_num, dtype, conj, neg, storage, requires_grad);

    // 2. 分配 buffer 用于序列化
    std::vector<char> buffer(atensor.storage.GetStorageDataSize());

    // 3. 序列化
    int serializeSize = SerializeTensorData(atensor, buffer.data());
    EXPECT_EQ(serializeSize, atensor.storage.GetStorageDataSize());

    // 4. 反序列化
    ATensor deserialized(
        size,
        stride,
        0,
        dim_num,
        dtype,
        conj,
        neg,
        ATStorage(6 * sizeof(float), nullptr, ATDevice(ATDeviceType::CPU, 0)),
        requires_grad);
    DeserializeTensorData(deserialized, buffer.data());

    // 5. 验证反序列化后的数据是否一致
    float* original_data = static_cast<float*>(atensor.storage.data);
    float* deserialized_data = static_cast<float*>(deserialized.storage.data);
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(deserialized_data[i], original_data[i]);
    }

    // 清理内存
    delete[] data;
}

TEST(atensor_seriliazer_test, serialize_deserialize_data_cuda) {
    GTEST_SKIP() << "Skip CUDA test case in CPU environment";

    // 1. 构造一个 CUDA ATensor 对象
    int64_t size[] = {2, 3};
    int64_t stride[] = {3, 1};
    int32_t dim_num = 2;
    ATDtype dtype = ATDtype::Float;
    bool conj = false;
    bool neg = false;
    bool requires_grad = true;

    // 分配并初始化数据
    float* host_data = new float[6]; // 2x3 的 float 数组
    for (int i = 0; i < 6; i++) {
        host_data[i] = static_cast<float>(i);
    }

    // 分配 GPU 内存并复制数据
    float* device_data;
    cudaMalloc(&device_data, 6 * sizeof(float));
    cudaMemcpy(device_data, host_data, 6 * sizeof(float), cudaMemcpyHostToDevice);

    ATStorage storage(6 * sizeof(float), device_data, ATDevice(ATDeviceType::CUDA, 0));
    ATensor atensor(size, stride, 0, dim_num, dtype, conj, neg, storage, requires_grad);

    // 2. 分配 buffer 用于序列化
    std::vector<char> buffer(atensor.storage.GetStorageDataSize());

    // 3. 序列化 - 从设备到主机
    int serializeSize = SerializeTensorData(atensor, buffer.data(), false);
    EXPECT_EQ(serializeSize, atensor.storage.GetStorageDataSize());

    // 4. 反序列化 - 从主机到设备
    ATensor deserialized(
        size,
        stride,
        0,
        dim_num,
        dtype,
        conj,
        neg,
        ATStorage(6 * sizeof(float), nullptr, ATDevice(ATDeviceType::CUDA, 0)),
        requires_grad);
    DeserializeTensorData(deserialized, buffer.data(), false);

    // 5. 验证反序列化后的数据是否一致
    float* deserialized_host_data = new float[6];
    cudaMemcpy(deserialized_host_data, deserialized.storage.data, 6 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(deserialized_host_data[i], host_data[i]);
    }

    // 清理内存
    delete[] host_data;
    delete[] deserialized_host_data;
    cudaFree(device_data);
    cudaFree(deserialized.storage.data);
}

TEST(atensor_seriliazer_test, serialize_deserialize_data_invalid_device) {
    // 1. 构造一个无效设备类型的 ATensor 对象
    int64_t size[] = {2, 3};
    int64_t stride[] = {3, 1};
    int32_t dim_num = 2;
    ATDtype dtype = ATDtype::Float;
    bool conj = false;
    bool neg = false;
    bool requires_grad = true;

    // 分配并初始化数据
    float* data = new float[6];
    for (int i = 0; i < 6; i++) {
        data[i] = static_cast<float>(i);
    }
    ATStorage storage(6 * sizeof(float), data, ATDevice(ATDeviceType::UNKNOWN_DEV, 0));

    ATensor atensor(size, stride, 0, dim_num, dtype, conj, neg, storage, requires_grad);

    // 2. 分配 buffer 用于序列化
    std::vector<char> buffer(atensor.storage.GetStorageDataSize());

    // 3. 序列化 - 应该返回 0 因为设备类型不支持
    int serializeSize = SerializeTensorData(atensor, buffer.data());
    EXPECT_EQ(serializeSize, 0);

    // 4. 反序列化 - 应该不会修改数据
    ATensor deserialized(
        size,
        stride,
        0,
        dim_num,
        dtype,
        conj,
        neg,
        ATStorage(6 * sizeof(float), nullptr, ATDevice(ATDeviceType::UNKNOWN_DEV, 0)),
        requires_grad);
    DeserializeTensorData(deserialized, buffer.data());
    EXPECT_EQ(deserialized.storage.data, nullptr);

    // 清理内存
    delete[] data;
}
