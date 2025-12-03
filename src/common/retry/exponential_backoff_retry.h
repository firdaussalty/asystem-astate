#pragma once

#include "common/retry/sleeping_retry.h"


namespace astate {

class ExponentialBackoffRetry : public SleepingRetry {
 public:
    ExponentialBackoffRetry(int base_sleep_time_ms, int max_sleep_ms, int max_retries);

    std::chrono::milliseconds GetSleepTime() override;

 private:
    int base_sleep_time_ms_;
    int max_sleep_ms_;

    static int SafeAbs(int value, int default_value);
};

} // namespace astate
