#pragma once

#include <atomic>
#include <thread>

namespace astate {

class SpinLock {
 public:
    void Lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    void Unlock() { flag_.clear(std::memory_order_release); }

 private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

class SpinGuard {
 public:
    explicit SpinGuard(SpinLock& lock)
        : lock_(lock) {
        lock_.Lock();
    }
    ~SpinGuard() { lock_.Unlock(); }

 private:
    SpinLock& lock_;
};

class RWSpinLock {
 public:
    void LockRead() {
        for (;;) {
            int expected = state_.load(std::memory_order_relaxed);
            if (expected >= 0) {
                if (state_.compare_exchange_weak(
                        expected, expected + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
                    return;
                }
            } else {
                std::this_thread::yield();
            }
        }
    }

    void UnlockRead() { state_.fetch_sub(1, std::memory_order_release); }

    void LockWrite() {
        for (;;) {
            int expected = 0;
            // If no one is reading or writing, set write lock (state=-1), otherwise wait
            if (state_.compare_exchange_weak(expected, -1, std::memory_order_acquire, std::memory_order_relaxed)) {
                return;
            }
            std::this_thread::yield();
        }
    }

    void UnlockWrite() { state_.store(0, std::memory_order_release); }

 private:
    // 0=free, >0=read_count, -1=write_lock
    std::atomic<int> state_{0};
};

class RWSpinGuard {
 public:
    RWSpinGuard(RWSpinLock& lock, bool is_write)
        : lock_(lock),
          is_write_(is_write) {
        if (is_write_) {
            lock_.LockWrite();
        } else {
            lock_.LockRead();
        }
    }

    ~RWSpinGuard() {
        if (is_write_) {
            lock_.UnlockWrite();
        } else {
            lock_.UnlockRead();
        }
    }

 private:
    RWSpinLock& lock_;
    bool is_write_;
};

} // namespace astate
