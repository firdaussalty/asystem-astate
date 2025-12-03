#pragma once

#include <cstdint>
#include <vector>

#include <torch/types.h>

namespace astate {

enum class ARole : uint8_t { TRAINING = 0, INFERENCE = 1 };

enum class TensorTableType : uint8_t {
    IN_MEMORY,
    REMOTE,
};

struct TorchTensorMeta {
    torch::Dtype dtype;
    std::vector<int64_t> size;
    torch::Device device;

    TorchTensorMeta()
        : dtype(torch::Dtype::Float),
          device(torch::Device(torch::DeviceType::CPU)) {}

    TorchTensorMeta(torch::Dtype d, std::vector<int64_t> s, torch::Device dev)
        : dtype(d),
          size(std::move(s)),
          device(dev) {}
};

} // namespace astate
