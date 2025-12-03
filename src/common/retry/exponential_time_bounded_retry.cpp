#include "common/retry/exponential_time_bounded_retry.h"

#include <chrono>
#include <cstdint>
#include <random>

namespace astate {

// ExponentialTimeBoundedRetry implementation
ExponentialTimeBoundedRetry::ExponentialTimeBoundedRetry(
    std::chrono::milliseconds max_duration,
    std::chrono::milliseconds initial_sleep,
    std::chrono::milliseconds max_sleep,
    bool skip_initial_sleep)
    : TimeBoundedRetry(max_duration),
      max_sleep_(max_sleep),
      next_sleep_(initial_sleep),
      skip_initial_sleep_(skip_initial_sleep),
      sleep_skipped_(false) {
}

ExponentialTimeBoundedRetry::Builder ExponentialTimeBoundedRetry::CreateBuilder() {
    return {};
}

std::chrono::milliseconds ExponentialTimeBoundedRetry::ComputeNextWaitTime() {
    if (skip_initial_sleep_ && !sleep_skipped_) {
        sleep_skipped_ = true;
        return std::chrono::milliseconds(0);
    }

    auto next = next_sleep_;

    // 计算下一次的睡眠时间（指数退避）
    next_sleep_ = std::chrono::milliseconds(next_sleep_.count() * 2);
    if (next_sleep_ > max_sleep_) {
        next_sleep_ = max_sleep_;
    }

    // 添加jitter
    static thread_local std::random_device random_device;
    static thread_local std::mt19937 gen(random_device());
    std::uniform_real_distribution<double> dis(0.0, 0.1); // 0-10%的抖动

    double jitter_ratio = dis(gen);
    int64_t jitter = static_cast<int64_t>(jitter_ratio) * next.count();

    return next + std::chrono::milliseconds(jitter);
}

// Builder 实现
ExponentialTimeBoundedRetry::Builder::Builder()
    : max_duration_(std::chrono::milliseconds(0)),
      initial_sleep_(std::chrono::milliseconds(0)),
      max_sleep_(std::chrono::milliseconds(0)),
      skip_initial_sleep_(false) {
}

ExponentialTimeBoundedRetry::Builder&
ExponentialTimeBoundedRetry::Builder::WithMaxDuration(std::chrono::milliseconds max_duration) {
    max_duration_ = max_duration;
    return *this;
}

ExponentialTimeBoundedRetry::Builder&
ExponentialTimeBoundedRetry::Builder::WithInitialSleep(std::chrono::milliseconds initial_sleep) {
    initial_sleep_ = initial_sleep;
    return *this;
}

ExponentialTimeBoundedRetry::Builder&
ExponentialTimeBoundedRetry::Builder::WithMaxSleep(std::chrono::milliseconds max_sleep) {
    max_sleep_ = max_sleep;
    return *this;
}

ExponentialTimeBoundedRetry::Builder& ExponentialTimeBoundedRetry::Builder::WithSkipInitialSleep() {
    skip_initial_sleep_ = true;
    return *this;
}

std::unique_ptr<ExponentialTimeBoundedRetry> ExponentialTimeBoundedRetry::Builder::Build() {
    return std::unique_ptr<ExponentialTimeBoundedRetry>(
        new ExponentialTimeBoundedRetry(max_duration_, initial_sleep_, max_sleep_, skip_initial_sleep_));
}

} // namespace astate
