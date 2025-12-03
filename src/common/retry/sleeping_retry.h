#pragma once

#include <chrono>

#include "common/retry/retry_policy.h"

namespace astate {

class SleepingRetry : public RetryPolicy {
 public:
    explicit SleepingRetry(int max_retries);

    [[nodiscard]] int GetAttemptCount() const override;
    bool Attempt() override;

    virtual std::chrono::milliseconds GetSleepTime() = 0;

 private:
    int max_retries_;
    int attempt_count_;
};

} // namespace astate
