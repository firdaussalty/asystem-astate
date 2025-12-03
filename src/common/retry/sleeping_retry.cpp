#include "common/retry/sleeping_retry.h"

#include <stdexcept>
#include <thread>


namespace astate {

SleepingRetry::SleepingRetry(int max_retries)
    : max_retries_(max_retries),
      attempt_count_(0) {
    if (max_retries <= 0) {
        throw std::invalid_argument("Max retries must be a positive number");
    }
}

int SleepingRetry::GetAttemptCount() const {
    return attempt_count_;
}

bool SleepingRetry::Attempt() {
    if (attempt_count_ <= max_retries_) {
        if (attempt_count_ == 0) {
            // first attempt, do not sleep
            attempt_count_++;
            return true;
        }

        try {
            auto sleep_time = GetSleepTime();
            std::this_thread::sleep_for(sleep_time);
            attempt_count_++;
            return true;
        } catch (...) {
            // Sleep interrupted or other error
            return false;
        }
    }
    return false;
}

} // namespace astate
