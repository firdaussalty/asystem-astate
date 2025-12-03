#pragma once

#include <string>

#include "astate/types.h"

namespace astate {

struct AParallelConfig {
    ARole role;
    int32_t role_size;
    int32_t role_rank;
    int32_t dp_size;
    int32_t dp_rank;
    int32_t tp_size;
    int32_t tp_rank;
    int32_t pp_size;
    int32_t pp_rank;
    int32_t cp_size;
    int32_t cp_rank;
    int32_t ep_size;
    int32_t ep_rank;
    int32_t etp_size;
    int32_t etp_rank;

    AParallelConfig() = default;

    AParallelConfig(
        ARole role,
        int32_t role_size,
        int32_t role_rank,
        int32_t dp_size = 1,
        int32_t dp_rank = 0,
        int32_t tp_size = 1,
        int32_t tp_rank = 0,
        int32_t pp_size = 1,
        int32_t pp_rank = 0,
        int32_t cp_size = 1,
        int32_t cp_rank = 0,
        int32_t ep_size = 1,
        int32_t ep_rank = 0,
        int32_t etp_size = 1,
        int32_t etp_rank = 0)
        : role(role),
          role_size(role_size),
          role_rank(role_rank),
          dp_size(dp_size),
          dp_rank(dp_rank),
          tp_size(tp_size),
          tp_rank(tp_rank),
          pp_size(pp_size),
          pp_rank(pp_rank),
          cp_size(cp_size),
          cp_rank(cp_rank),
          ep_size(ep_size),
          ep_rank(ep_rank),
          etp_size(etp_size),
          etp_rank(etp_rank) {}

    [[nodiscard]] bool IsTraining() const { return role == ARole::TRAINING; }

    [[nodiscard]] bool IsInference() const { return role == ARole::INFERENCE; }

    [[nodiscard]] std::string ToString() const;
};

} // namespace astate
