#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace astate {

struct ShardedKey {
    std::string key;
    std::vector<int64_t> global_shape;
    std::vector<int64_t> global_offset;

    [[nodiscard]] std::string ToString() const;
};

inline bool operator==(const ShardedKey& lhs, const ShardedKey& rhs) {
    return lhs.key == rhs.key && lhs.global_shape == rhs.global_shape && lhs.global_offset == rhs.global_offset;
}

} // namespace astate
