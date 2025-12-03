#include <chrono>
#include <stdexcept>

#include <gtest/gtest.h>

#include "common/retry/counting_retry.h"
#include "common/retry/retry_utils.h"

class TestException : public std::runtime_error {
 public:
    explicit TestException(const std::string& msg)
        : std::runtime_error(msg) {}
};

class CountingAndSleepRetryPolicyTest : public ::testing::Test {
 protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CountingAndSleepRetryPolicyTest, Constructor) {
    EXPECT_NO_THROW({ astate::CountingAndSleepRetryPolicy policy(3, 100); });

    EXPECT_NO_THROW({ astate::CountingAndSleepRetryPolicy policy(0, 100); });

    EXPECT_NO_THROW({ astate::CountingAndSleepRetryPolicy policy(3, 0); });

    EXPECT_THROW({ astate::CountingAndSleepRetryPolicy policy(-1, 100); }, std::invalid_argument);

    EXPECT_NO_THROW({ astate::CountingAndSleepRetryPolicy policy(3, -100); });
}

TEST_F(CountingAndSleepRetryPolicyTest, AttemptCount) {
    astate::CountingAndSleepRetryPolicy policy(3, 50);

    EXPECT_EQ(policy.GetAttemptCount(), 0);

    EXPECT_TRUE(policy.Attempt());
    EXPECT_EQ(policy.GetAttemptCount(), 1);

    EXPECT_TRUE(policy.Attempt());
    EXPECT_EQ(policy.GetAttemptCount(), 2);

    EXPECT_TRUE(policy.Attempt());
    EXPECT_EQ(policy.GetAttemptCount(), 3);

    EXPECT_TRUE(policy.Attempt());
    EXPECT_EQ(policy.GetAttemptCount(), 4);

    EXPECT_FALSE(policy.Attempt());
    EXPECT_EQ(policy.GetAttemptCount(), 4);
}

TEST_F(CountingAndSleepRetryPolicyTest, SleepFunctionality) {
    const int sleep_ms = 100;
    astate::CountingAndSleepRetryPolicy policy(3, sleep_ms);

    auto start = std::chrono::steady_clock::now();

    EXPECT_TRUE(policy.Attempt());
    auto after_first = std::chrono::steady_clock::now();
    auto first_duration = std::chrono::duration_cast<std::chrono::milliseconds>(after_first - start);
    EXPECT_LT(first_duration.count(), 50);

    EXPECT_TRUE(policy.Attempt());
    auto after_second = std::chrono::steady_clock::now();
    auto second_duration = std::chrono::duration_cast<std::chrono::milliseconds>(after_second - after_first);
    EXPECT_GE(second_duration.count(), sleep_ms - 10);

    EXPECT_TRUE(policy.Attempt());
    auto after_third = std::chrono::steady_clock::now();
    auto third_duration = std::chrono::duration_cast<std::chrono::milliseconds>(after_third - after_second);
    EXPECT_GE(third_duration.count(), sleep_ms - 10);
}

TEST_F(CountingAndSleepRetryPolicyTest, ZeroSleepTime) {
    astate::CountingAndSleepRetryPolicy policy(3, 0);

    auto start = std::chrono::steady_clock::now();

    EXPECT_TRUE(policy.Attempt());

    EXPECT_TRUE(policy.Attempt());
    auto after_second = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after_second - start);
    EXPECT_LT(duration.count(), 50);
}

TEST_F(CountingAndSleepRetryPolicyTest, IntegrationWithRetryUtils) {
    astate::CountingAndSleepRetryPolicy policy(3, 50);
    int attempt_count = 0;

    auto start = std::chrono::steady_clock::now();

    EXPECT_NO_THROW({
        astate::RetryUtils::Retry(
            "test action",
            [&attempt_count]() {
                attempt_count++;
                if (attempt_count < 3) {
                    throw TestException("test error");
                }
            },
            policy);
    });

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_EQ(attempt_count, 3);
    EXPECT_GE(elapsed.count(), 100);
}

TEST_F(CountingAndSleepRetryPolicyTest, ReturnValueIntegration) {
    astate::CountingAndSleepRetryPolicy policy(3, 50);
    int attempt_count = 0;

    int result = 0;
    EXPECT_NO_THROW({
        result = astate::RetryUtils::Retry<int>(
            "test callable",
            [&attempt_count]() -> int {
                attempt_count++;
                if (attempt_count < 3) {
                    throw TestException("test error");
                }
                return 42;
            },
            policy);
    });

    EXPECT_EQ(result, 42);
    EXPECT_EQ(attempt_count, 3);
}

TEST_F(CountingAndSleepRetryPolicyTest, MaxRetriesLimit) {
    astate::CountingAndSleepRetryPolicy policy(2, 10);
    int attempt_count = 0;

    auto start = std::chrono::steady_clock::now();

    EXPECT_THROW(
        {
            astate::RetryUtils::Retry(
                "test action",
                [&attempt_count]() {
                    attempt_count++;
                    throw TestException("always fail");
                },
                policy);
        },
        TestException);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_EQ(attempt_count, 3);
    EXPECT_GE(elapsed.count(), 20);
}

TEST_F(CountingAndSleepRetryPolicyTest, NonRetryableException) {
    astate::CountingAndSleepRetryPolicy policy(5, 50);
    int attempt_count = 0;

    auto start = std::chrono::steady_clock::now();

    EXPECT_THROW(
        {
            astate::RetryUtils::Retry(
                "test action",
                [&attempt_count]() {
                    attempt_count++;
                    throw astate::NonRetryableException("non-retryable error");
                },
                policy);
        },
        astate::NonRetryableException);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_EQ(attempt_count, 1);
    EXPECT_LT(elapsed.count(), 50);
}

TEST_F(CountingAndSleepRetryPolicyTest, EdgeCases) {
    astate::CountingAndSleepRetryPolicy policy_zero(0, 100);
    int attempt_count = 0;

    EXPECT_THROW(
        {
            astate::RetryUtils::Retry(
                "test action",
                [&attempt_count]() {
                    attempt_count++;
                    throw TestException("test error");
                },
                policy_zero);
        },
        TestException);

    EXPECT_EQ(attempt_count, 1);

    astate::CountingAndSleepRetryPolicy policy_large_sleep(2, 1000);
    attempt_count = 0;

    auto start = std::chrono::steady_clock::now();

    EXPECT_THROW(
        {
            astate::RetryUtils::Retry(
                "test action",
                [&attempt_count]() {
                    attempt_count++;
                    throw TestException("test error");
                },
                policy_large_sleep);
        },
        TestException);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_EQ(attempt_count, 3);
    EXPECT_GE(elapsed.count(), 2000);
}

TEST_F(CountingAndSleepRetryPolicyTest, ResetFunctionality) {
    astate::CountingAndSleepRetryPolicy policy(3, 50);

    EXPECT_TRUE(policy.Attempt());
    EXPECT_TRUE(policy.Attempt());
    EXPECT_EQ(policy.GetAttemptCount(), 2);

    policy.Reset();
    EXPECT_EQ(policy.GetAttemptCount(), 0);

    EXPECT_TRUE(policy.Attempt());
    EXPECT_EQ(policy.GetAttemptCount(), 1);
}

TEST_F(CountingAndSleepRetryPolicyTest, PerformanceBenchmark) {
    const int sleep_ms = 50;
    const int max_retries = 3;
    astate::CountingAndSleepRetryPolicy policy(max_retries, sleep_ms);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i <= max_retries; ++i) {
        policy.Attempt();
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int expected_min_time = max_retries * sleep_ms;
    int expected_max_time = (max_retries * sleep_ms) + 100;

    EXPECT_GE(elapsed.count(), expected_min_time);
    EXPECT_LE(elapsed.count(), expected_max_time);
}
