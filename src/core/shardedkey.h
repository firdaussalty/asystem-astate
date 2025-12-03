#pragma once

#include <utility>

#include "astate/sharded_key.h"

namespace astate {

// Define hash function object in the same namespace
struct ShardedKeyHash {
    size_t operator()(const ShardedKey& key) const {
        size_t h1 = std::hash<std::string>{}(key.key);

        // Calculate hash for vector
        size_t h2 = 0;
        for (const auto& x : key.global_shape) {
            h2 = (h2 * 31) + std::hash<int64_t>{}(x);
        }

        size_t h3 = 0;
        for (auto& x : key.global_offset) {
            h3 = (h3 * 31) + std::hash<int64_t>{}(x);
        }

        // Combine hash values
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

} // namespace astate
