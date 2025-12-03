#include "core/atensor.h"

#include "astate/config.h"

namespace astate {

std::string AParallelConfig::ToString() const {
    std::string result = "AParallelConfig(role=";
    result += RoleToString(role);
    result += ", role_size=" + std::to_string(role_size);
    result += ", role_rank=" + std::to_string(role_rank);
    result += ", dp=" + std::to_string(dp_rank) + "/" + std::to_string(dp_size);
    result += ", tp=" + std::to_string(tp_rank) + "/" + std::to_string(tp_size);
    result += ", pp=" + std::to_string(pp_rank) + "/" + std::to_string(pp_size);
    result += ", cp=" + std::to_string(cp_rank) + "/" + std::to_string(cp_size);
    result += ", ep=" + std::to_string(ep_rank) + "/" + std::to_string(ep_size);
    result += ", etp=" + std::to_string(etp_rank) + "/" + std::to_string(etp_size) + ")";
    return result;
}

} // namespace astate
