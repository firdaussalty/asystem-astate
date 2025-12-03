#include "common/retry/exponential_backoff_retry.h"

#include <algorithm>
#include <climits>
#include <random>
#include <stdexcept>


namespace astate {

ExponentialBackoffRetry::ExponentialBackoffRetry(int base_sleep_time_ms, int max_sleep_ms, int max_retries)
    : SleepingRetry(max_retries),
      base_sleep_time_ms_(base_sleep_time_ms),
      max_sleep_ms_(max_sleep_ms) {
    if (base_sleep_time_ms < 0) {
        throw std::invalid_argument("Base must be a positive number, or 0");
    }
    if (max_sleep_ms < 0) {
        throw std::invalid_argument("Max must be a positive number, or 0");
    }
}

std::chrono::milliseconds ExponentialBackoffRetry::GetSleepTime() {
    int count = GetAttemptCount();
    if (count >= 30) {
        // current logic overflows at 30, so set value to max
        return std::chrono::milliseconds(max_sleep_ms_);
    } // use randomness to avoid contention between many operations using the same
    // retry policy
    static thread_local std::random_device random_device;
    static thread_local std::mt19937 gen(random_device());

    int lower_bound = 1 << count;
    int upper_bound = 1 << (count + 1);
    std::uniform_int_distribution<int> dis(lower_bound, upper_bound - 1);

    int sleep_ms = base_sleep_time_ms_ * dis(gen);
    return std::chrono::milliseconds(std::min(SafeAbs(sleep_ms, max_sleep_ms_), max_sleep_ms_));
}

int ExponentialBackoffRetry::SafeAbs(int value, int default_value) {
    int result = std::abs(value);
    if (result == INT_MIN) {
        result = default_value;
    }
    return result;
}

} // namespace astate
