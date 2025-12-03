#pragma once

#include <chrono>

#include "common/retry/retry_policy.h"

namespace astate {

class TimeBoundedRetry : public RetryPolicy {
 public:
    explicit TimeBoundedRetry(std::chrono::milliseconds max_duration);

    [[nodiscard]] int GetAttemptCount() const override;

    bool Attempt() override;

    virtual std::chrono::milliseconds ComputeNextWaitTime() = 0;

 private:
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;
    int attempt_count_;
};

} // namespace astate
