#include "common/thread_pool.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <ATen/ops/empty.h>
#include <gtest/gtest.h>

#include "common/cuda_utils.h"

namespace astate {
class ThreadPoolTest : public ::testing::Test {
 protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ThreadPoolTest, BasicTaskExecution) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    futures.reserve(10);
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.Submit([&counter] { ++counter; }));
    }
    for (auto& f : futures) {
        f.get();
    }
    EXPECT_EQ(counter, 10);
}

TEST_F(ThreadPoolTest, ReturnValue) {
    ThreadPool pool(2);
    auto fut = pool.Submit([](int a, int b) { return a + b; }, 3, 5);
    EXPECT_EQ(fut.get(), 8);
}

TEST_F(ThreadPoolTest, ExceptionHandling) {
    ThreadPool pool(2);
    auto fut = pool.Submit([] { throw std::runtime_error("error"); });
    EXPECT_THROW(fut.get(), std::runtime_error);
}

TEST_F(ThreadPoolTest, WaitForTasks) {
    ThreadPool pool(2);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    futures.reserve(5);
    for (int i = 0; i < 5; ++i) {
        futures.push_back(pool.Submit([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            ++counter;
        }));
    }
    pool.WaitForTasks();
    for (auto& f : futures) {
        f.get();
    }
    EXPECT_EQ(counter, 5);
}

TEST_F(ThreadPoolTest, SubmitAfterDestructionThrows) {
    std::unique_ptr<ThreadPool> pool = std::make_unique<ThreadPool>(2);
    pool->Submit([] {}).get();
    pool.reset();
    // No direct way to test Submit after destruction, but destructor should not deadlock
    SUCCEED();
}

class CUDAStreamThreadPoolTest : public ::testing::Test {
 protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CUDAStreamThreadPoolTest, BasicTaskExecution) {
    GTEST_SKIP();
    CUDAStreamThreadPool pool(4);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    futures.reserve(10);
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.Submit([&counter](const std::shared_ptr<c10::cuda::CUDAStream>& stream) {
            ++counter;
            if (HasNvGpu()) {
                EXPECT_EQ(stream, nullptr); // testing env is cpu only
            }
        }));
    }
    for (auto& f : futures) {
        f.get();
    }
    EXPECT_EQ(counter, 10);
}

TEST_F(CUDAStreamThreadPoolTest, ReturnValue) {
    GTEST_SKIP();
    CUDAStreamThreadPool pool(2);
    auto fut = pool.Submit(
        [](const std::shared_ptr<c10::cuda::CUDAStream>& stream, int a, int b) {
            return a + b;
            if (HasNvGpu()) {
                EXPECT_EQ(stream, nullptr); // testing env is cpu only
            }
        },
        3,
        5);
    EXPECT_EQ(fut.get(), 8);
}

TEST_F(CUDAStreamThreadPoolTest, ExceptionHandling) {
    GTEST_SKIP();
    CUDAStreamThreadPool pool(2);
    auto fut = pool.Submit(
        [](const std::shared_ptr<c10::cuda::CUDAStream>& /*stream*/) { throw std::runtime_error("error"); });
    EXPECT_THROW(fut.get(), std::runtime_error);
}

TEST_F(CUDAStreamThreadPoolTest, WaitForTasks) {
    GTEST_SKIP();
    CUDAStreamThreadPool pool(2);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    futures.reserve(5);
    for (int i = 0; i < 5; ++i) {
        futures.push_back(pool.Submit([&counter](const std::shared_ptr<c10::cuda::CUDAStream>& stream) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            ++counter;
            if (HasNvGpu()) {
                EXPECT_EQ(stream, nullptr); // testing env is cpu only
            }
        }));
    }
    pool.WaitForTasks();
    for (auto& f : futures) {
        f.get();
    }
    EXPECT_EQ(counter, 5);
}

TEST_F(CUDAStreamThreadPoolTest, SubmitAfterDestructionThrows) {
    GTEST_SKIP();
    std::unique_ptr<CUDAStreamThreadPool> pool = std::make_unique<CUDAStreamThreadPool>(2);
    pool->Submit([](const std::shared_ptr<c10::cuda::CUDAStream>& stream) {}).get();
    pool.reset();
    // No direct way to test Submit after destruction, but destructor should not deadlock
    SUCCEED();
}

class CUDAStreamResourceThreadPoolTest : public ::testing::Test {
 protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CUDAStreamResourceThreadPoolTest, BasicTaskExecution) {
    GTEST_SKIP();
    CUDAStreamResourceThreadPool<torch::Tensor> pool(
        [&]() {
            torch::Tensor tensor = torch::empty(
                {1024},
                torch::TensorOptions()
                    .dtype(torch::ScalarType::Byte)
                    .device(torch::DeviceType::CPU)
                    .requires_grad(false)
                    .pinned_memory(torch::cuda::is_available()));
            return tensor;
        },
        [](torch::Tensor& tensor) { tensor.reset(); },
        2);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    futures.reserve(10);
    for (int i = 0; i < 10; ++i) {
        futures.push_back(
            pool.Submit([&counter](torch::Tensor& /*local*/, const std::shared_ptr<c10::cuda::CUDAStream>& stream) {
                ++counter;
                if (HasNvGpu()) {
                    EXPECT_EQ(stream, nullptr); // testing env is cpu only
                }
            }));
    }
    for (auto& f : futures) {
        f.get();
    }
    EXPECT_EQ(counter, 10);
}

} // namespace astate
