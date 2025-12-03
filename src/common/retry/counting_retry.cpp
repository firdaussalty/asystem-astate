#include "common/retry/counting_retry.h"

#include <stdexcept>

namespace astate {

CountingRetry::CountingRetry(int max_retries)
    : max_retries_(max_retries),
      attempt_count_(0) {
    if (max_retries < 0) {
        throw std::invalid_argument("Max retries must be a non-negative number");
    }
}

void CountingRetry::Reset() {
    attempt_count_ = 0;
}

int CountingRetry::GetAttemptCount() const {
    return attempt_count_;
}

bool CountingRetry::Attempt() {
    if (attempt_count_ <= max_retries_) {
        attempt_count_++;
        return true;
    }
    return false;
}

} // namespace astate
