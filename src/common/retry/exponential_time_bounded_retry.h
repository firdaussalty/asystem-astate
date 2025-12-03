#pragma once

#include <chrono>
#include <memory>

#include "common/retry/time_bounded_retry.h"

namespace astate {

/**
 * A retry policy which uses exponential backoff and a maximum duration time
 * bound.
 *
 * A final retry will be performed at the time bound before giving up.
 */
class ExponentialTimeBoundedRetry : public TimeBoundedRetry {
 public:
    /**
     * Builder for ExponentialTimeBoundedRetry
     */
    class Builder {
     public:
        Builder();

        /**
         * @param max_duration max total duration to retry for
         * @return the builder
         */
        Builder& WithMaxDuration(std::chrono::milliseconds max_duration);

        /**
         * @param initial_sleep initial sleep interval between retries
         * @return the builder
         */
        Builder& WithInitialSleep(std::chrono::milliseconds initial_sleep);

        /**
         * @param max_sleep maximum sleep interval between retries
         * @return the builder
         */
        Builder& WithMaxSleep(std::chrono::milliseconds max_sleep);

        /**
         * first sleep will be skipped.
         *
         * @return the builder
         */
        Builder& WithSkipInitialSleep();

        /**
         * @return the built retry mechanism
         */
        std::unique_ptr<ExponentialTimeBoundedRetry> Build();

     private:
        std::chrono::milliseconds max_duration_;
        std::chrono::milliseconds initial_sleep_;
        std::chrono::milliseconds max_sleep_;
        bool skip_initial_sleep_;
    };

    /**
     * @return a builder
     */
    static Builder CreateBuilder();

    std::chrono::milliseconds ComputeNextWaitTime() override;

 private:
    ExponentialTimeBoundedRetry(
        std::chrono::milliseconds max_duration,
        std::chrono::milliseconds initial_sleep,
        std::chrono::milliseconds max_sleep,
        bool skip_initial_sleep);

    std::chrono::milliseconds max_sleep_;
    std::chrono::milliseconds next_sleep_;
    bool skip_initial_sleep_;
    bool sleep_skipped_;

    friend class Builder;
};

} // namespace astate
