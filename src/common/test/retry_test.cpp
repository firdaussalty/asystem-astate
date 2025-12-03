#include <chrono>
#include <stdexcept>

#include <gtest/gtest.h>

#include "common/retry/counting_retry.h"
#include "common/retry/exponential_backoff_retry.h"
#include "common/retry/exponential_time_bounded_retry.h"
#include "common/retry/retry_utils.h"
#include "common/retry/timeout_retry.h"

namespace astate {
class TestException : public std::runtime_error {
 public:
    explicit TestException(const std::string& msg)
        : std::runtime_error(msg) {}
};

TEST(RetryTest, CountingRetry) {
    CountingRetry retry(3);
    int count = 0;

    EXPECT_NO_THROW({
        RetryUtils::Retry(
            "test action",
            [&count]() {
                count++;
                if (count < 3) {
                    throw TestException("test error");
                }
            },
            retry);
    });

    EXPECT_EQ(count, 3);
}

TEST(RetryTest, ExponentialBackoffRetry) {
    ExponentialBackoffRetry retry(50, 1000, 3);
    int count = 0;
    auto start = std::chrono::steady_clock::now();

    EXPECT_NO_THROW({
        RetryUtils::Retry(
            "test action",
            [&count]() {
                count++;
                if (count < 3) {
                    throw TestException("test error");
                }
            },
            retry);
    });

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_EQ(count, 3);

    EXPECT_GT(elapsed.count(), 0);
}

TEST(RetryTest, TimeoutRetry) {
    TimeoutRetry retry(1000, 100);
    int count = 0;
    auto start = std::chrono::steady_clock::now();

    EXPECT_THROW(
        {
            RetryUtils::Retry(
                "test action",
                [&count]() {
                    count++;
                    throw TestException("test error"); // Always fail
                },
                retry);
        },
        TestException);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_GE(elapsed.count(), 1000); // Should have waited at least timeout duration
    EXPECT_GT(count, 0); // Should have made at least one attempt
}

TEST(RetryTest, ExponentialTimeBoundedRetry) {
    auto retry = ExponentialTimeBoundedRetry::Builder()
                     .WithMaxDuration(std::chrono::milliseconds(1000))
                     .WithInitialSleep(std::chrono::milliseconds(50))
                     .WithMaxSleep(std::chrono::milliseconds(300))
                     .Build();

    int count = 0;
    auto start = std::chrono::steady_clock::now();

    EXPECT_THROW(
        {
            RetryUtils::Retry(
                "test action",
                [&count]() {
                    count++;
                    throw TestException("test error"); // Always fail
                },
                *retry);
        },
        TestException);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_GE(elapsed.count(), 900); // Should have waited at least timeout duration
    EXPECT_GT(count, 0); // Should have made at least one attempt
}

TEST(RetryTest, CallableRetry) {
    CountingRetry retry(3);
    int count = 0;

    int result = 0;
    EXPECT_NO_THROW({
        result = RetryUtils::Retry<int>(
            "test callable",
            [&count]() -> int {
                count++;
                if (count < 3) {
                    throw TestException("test error");
                }
                return 42;
            },
            retry);
    });

    EXPECT_EQ(result, 42);
    EXPECT_EQ(count, 3);
}

TEST(RetryTest, DiscoveryRetryIntegration) {
    ExponentialBackoffRetry retry(1000, 10000, 3);
    int attempt_count = 0;
    auto start = std::chrono::steady_clock::now();

    EXPECT_NO_THROW({
        RetryUtils::Retry(
            "simulated network operation",
            [&attempt_count]() {
                attempt_count++;
                if (attempt_count < 3) {
                    throw TestException("Simulated network failure");
                }
            },
            retry);
    });

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_EQ(attempt_count, 3);
    EXPECT_GT(elapsed.count(), 0); // Should have some delay
}

} // namespace astate
