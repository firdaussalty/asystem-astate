#include "common/retry/retry_utils.h"

#include <stdexcept>

#include "common/retry/retry_policy.h"


namespace astate {

void RetryUtils::Retry(const std::string& /*action*/, const RunnableThrowsException& func, RetryPolicy& policy) {
    std::exception_ptr last_exception = nullptr;
    while (policy.Attempt()) {
        try {
            func();
            return;
        } catch (const NonRetryableException& e) {
            // Non-retryable exception - rethrow immediately without retry
            throw;
        } catch (...) {
            last_exception = std::current_exception();
            // Log warning could be added here if logging is available
        }
    }
    if (last_exception) {
        std::rethrow_exception(last_exception);
    }
    throw std::runtime_error("Retry failed without exception");
}

} // namespace astate
