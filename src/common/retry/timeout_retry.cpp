#include "common/retry/timeout_retry.h"

#include <stdexcept>
#include <thread>


namespace astate {

TimeoutRetry::TimeoutRetry(int64_t retry_timeout_ms, int sleep_ms)
    : retry_timeout_ms_(retry_timeout_ms),
      sleep_ms_(sleep_ms),
      attempt_count_(0),
      first_attempt_(true) {
    if (retry_timeout_ms <= 0) {
        throw std::invalid_argument("Retry timeout must be a positive number");
    }
    if (sleep_ms < 0) {
        throw std::invalid_argument("sleep_ms cannot be negative");
    }
}

int TimeoutRetry::GetAttemptCount() const {
    return attempt_count_;
}

bool TimeoutRetry::Attempt() {
    if (first_attempt_) {
        // first attempt, set the start time
        start_time_ = std::chrono::steady_clock::now();
        attempt_count_++;
        first_attempt_ = false;
        return true;
    }

    if (sleep_ms_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
    }

    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_);

    if (elapsed.count() <= retry_timeout_ms_) {
        attempt_count_++;
        return true;
    }
    return false;
}

} // namespace astate
