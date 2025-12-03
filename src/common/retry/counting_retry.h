#pragma once

#include <chrono>
#include <thread>

#include "common/retry/retry_policy.h"

namespace astate {

/**
 * An option which allows retrying based on maximum count.
 */
class CountingRetry : public RetryPolicy {
 public:
    /**
     * Constructs a retry facility which allows max number of retries.
     *
     * @param max_retries max number of retries
     */
    explicit CountingRetry(int max_retries);

    /**
     * Reset the count of retries.
     */
    void Reset();

    // RetryPolicy interface
    [[nodiscard]] int GetAttemptCount() const override;
    bool Attempt() override;

 private:
    int max_retries_;
    int attempt_count_;
};

class CountingAndSleepRetryPolicy : public CountingRetry {
 public:
    CountingAndSleepRetryPolicy(int max_retries, int sleep_ms)
        : CountingRetry(max_retries),
          sleep_ms_(sleep_ms) {}

    bool Attempt() override {
        if (CountingRetry::Attempt()) {
            if (GetAttemptCount() > 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
            }
            return true;
        }
        return false;
    }

 private:
    int sleep_ms_;
};

} // namespace astate
