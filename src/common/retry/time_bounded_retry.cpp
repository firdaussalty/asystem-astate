#include "common/retry/time_bounded_retry.h"

#include <thread>


namespace astate {

TimeBoundedRetry::TimeBoundedRetry(std::chrono::milliseconds max_duration)
    : start_time_(std::chrono::steady_clock::now()),
      end_time_(start_time_ + max_duration),
      attempt_count_(0) {
}

int TimeBoundedRetry::GetAttemptCount() const {
    return attempt_count_;
}

bool TimeBoundedRetry::Attempt() {
    if (attempt_count_ == 0) {
        attempt_count_++;
        return true;
    }

    auto now = std::chrono::steady_clock::now();
    if (now >= end_time_) {
        return false;
    }

    auto next_wait_time = ComputeNextWaitTime();
    if (now + next_wait_time > end_time_) {
        next_wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - now);
    }

    if (next_wait_time > std::chrono::milliseconds(0)) {
        try {
            std::this_thread::sleep_for(next_wait_time);
        } catch (...) {
            // Sleep interrupted
            return false;
        }
    }

    attempt_count_++;
    return true;
}

} // namespace astate
