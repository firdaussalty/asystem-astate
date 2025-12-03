#pragma once

#include <cstdint>

namespace astate {

enum class TransportType : uint8_t { HTTP = 1, RDMA = 2, UNKNOWN = 255 };

} // namespace astate
