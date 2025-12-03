#pragma once

#include <chrono>

#include "common/retry/retry_policy.h"


namespace astate {

class TimeoutRetry : public RetryPolicy {
 public:
    TimeoutRetry(int64_t retry_timeout_ms, int sleep_ms);

    [[nodiscard]] int GetAttemptCount() const override;

    bool Attempt() override;

 private:
    int64_t retry_timeout_ms_;
    int64_t sleep_ms_;
    std::chrono::steady_clock::time_point start_time_;
    int attempt_count_;
    bool first_attempt_;
};

} // namespace astate
