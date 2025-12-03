#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace astate {

// A simple thread-safe message queue.
template <typename T>
class MessageQueue {
 public:
    void Push(T value) {
        std::unique_lock<std::mutex> lock(mtx_);
        q_.push(std::move(value));
        cond_.notify_one();
    }

    // Blocking wait until there is a message
    T Pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cond_.wait(lock, [this] { return !q_.empty(); });
        T val = std::move(q_.front());
        q_.pop();
        return val;
    }

    // Try to pop a message (non-blocking); optional, for polling
    bool TryPop(T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (q_.empty()) {
            return false;
        }
        value = std::move(q_.front());
        q_.pop();
        return true;
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return q_.size();
    }

 private:
    std::queue<T> q_;
    mutable std::mutex mtx_;
    std::condition_variable cond_;
};

} // namespace astate