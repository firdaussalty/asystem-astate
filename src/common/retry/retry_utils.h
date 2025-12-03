#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "common/retry/retry_policy.h"


namespace astate {

/**
 * Base exception class for non-retryable errors.
 * When this exception is thrown, retry operations should be abandoned
 * immediately.
 */
class NonRetryableException : public std::runtime_error {
 public:
    explicit NonRetryableException(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * Interface for functions which return nothing and may throw exceptions.
 */
using RunnableThrowsException = std::function<void()>;

/**
 * Interface for functions which return a value and may throw exceptions.
 */
template <typename T>
using CallableThrowsException = std::function<T()>;

/**
 * Utilities for performing retries.
 */
class RetryUtils {
 public:
    RetryUtils() = delete; // prevent instantiation
    /**
     * Retries the given method until it doesn't throw an exception or the retry
     * policy expires. If the retry policy expires, the last exception generated
     * will be rethrown.
     *
     * @param action a description of the action that fits the phrase "Failed to
     * ${action}"
     * @param f the function to retry
     * @param policy the retry policy to use
     */
    static void Retry(const std::string& action, const RunnableThrowsException& func, RetryPolicy& policy);

    /**
     * Retries the given callable until it doesn't throw an exception or the retry
     * policy expires. If the retry policy expires, the last exception generated
     * will be rethrown.
     *
     * @param action a description of the action that fits the phrase "Failed to
     * ${action}"
     * @param f the callable to retry
     * @param policy the retry policy to use
     * @return the result of the callable
     */
    template <typename T>
    static T Retry(const std::string& action, const CallableThrowsException<T>& func, RetryPolicy& policy);

    /**
     * @return the best effort policy with no retry
     */
    static std::unique_ptr<RetryPolicy> NoRetryPolicy();
};

template <typename T>
T RetryUtils::Retry(const std::string& /*action*/, const CallableThrowsException<T>& func, RetryPolicy& policy) {
    std::exception_ptr last_exception = nullptr;
    while (policy.Attempt()) {
        try {
            return func();
        } catch (const NonRetryableException& e) {
            throw;
        } catch (...) {
            last_exception = std::current_exception();
        }
    }
    if (last_exception) {
        std::rethrow_exception(last_exception);
    }
    throw std::runtime_error("Retry failed without exception");
}

} // namespace astate
