#pragma once

#include <chrono>
#include <thread>

#include <spdlog/spdlog.h>

namespace astate {

#define US_TO_MS(us) (us / 1000.0)
#define MS_TO_US(ms) (ms * 1000.0)
#define US_TO_SEC(us) (us / 1000000.0)
#define SEC_TO_US(sec) (sec * 1000000.0)
#define MS_TO_SEC(ms) (ms / 1000.0)
#define SEC_TO_MS(sec) (sec * 1000.0)

constexpr int64_t ONE_MINUTE_MS = 60000;

inline bool
WaitCondition(const std::function<bool()>& cond, const std::string& cond_name, int64_t max_ms, int interval_ms = 1) {
    std::chrono::milliseconds wait_time_ms(0);
    while (true) {
        if (cond()) {
            return true;
        }

        if (wait_time_ms.count() > max_ms) {
            SPDLOG_ERROR("Wait for condition [{}] timeout, {}ms", cond_name, wait_time_ms.count());
            break;
        }

        if (wait_time_ms.count() > 0 && wait_time_ms.count() % ONE_MINUTE_MS == 0) {
            SPDLOG_INFO("Wait for condition [{}] {}ms", cond_name, wait_time_ms.count());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        wait_time_ms += std::chrono::milliseconds(interval_ms);
    }
    return false;
}

} // namespace astate
