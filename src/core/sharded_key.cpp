#include "astate/sharded_key.h"

#include <sstream>
#include <string>

namespace astate {

std::string ShardedKey::ToString() const {
    std::ostringstream oss;
    oss << "key=" << key << ", globalShape=[";
    if (global_shape.size() > 0) {
        oss << global_shape[0];
        for (size_t i = 1; i < global_shape.size(); ++i) {
            oss << ", " << global_shape[i];
        }
    }
    oss << "], globalOffset=[";
    if (global_offset.size() > 0) {
        oss << global_offset[0];
        for (size_t i = 1; i < global_offset.size(); ++i) {
            oss << ", " << global_offset[i];
        }
    }
    oss << "]";
    return oss.str();
}

} // namespace astate
