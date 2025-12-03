#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include "common/cuda_utils.h"

namespace astate {

using Thread = std::thread;
using Mutex = std::mutex;
using ConditionVariable = std::condition_variable;
using Queue = std::queue<std::function<void()>>;

static std::mutex cuda_stream_mutex;
static int current_cuda_device = -1;
static std::shared_ptr<c10::cuda::CUDAStream> GetSafeCudaStream() {
    std::lock_guard<std::mutex> lock(cuda_stream_mutex);

    if (!HasNvGpu()) {
        SPDLOG_INFO("No NVIDIA GPU detected, cannot create CUDA stream and "
                    "return null.");
        return nullptr;
    }

    // Retrieve and cache the current CUDA device
    if (current_cuda_device == -1) {
        cudaGetDevice(&current_cuda_device);
        SPDLOG_INFO("Current CUDA device: {}", current_cuda_device);
    }

    // Set CUDA device context for this thread
    // AT_CUDA_CHECK(cudaSetDevice(current_cuda_device));

    // Create and return a new CUDA stream from the pool
    auto stream = c10::cuda::getStreamFromPool(false, static_cast<c10::DeviceIndex>(current_cuda_device));
    SPDLOG_INFO("Created CUDA stream: {}", stream.id());

    // Verify if the stream is valid
    if (stream.device_index() != current_cuda_device) {
        SPDLOG_ERROR(
            "Failed to get CUDA stream for device {}: got stream for device {}",
            current_cuda_device,
            stream.device_index());
        throw std::runtime_error(
            "Failed to get CUDA stream from pool for device " + std::to_string(current_cuda_device));
    }

    return std::make_shared<c10::cuda::CUDAStream>(stream);
}

class ThreadPool {
 public:
    explicit ThreadPool(size_t threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock, [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty()) {
                            return;
                        }
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F, class... Args>
    auto Submit(F&& func, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("submit on stopped thread_pool");
            }

            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return res;
    }

    void WaitForTasks() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (tasks_.empty() || stop_) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    size_t GetTaskCount() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }

 private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_{};
};

class CUDAStreamThreadPool {
 public:
    explicit CUDAStreamThreadPool(size_t threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < threads; ++i) {
            auto stream = GetSafeCudaStream();
            streams_.emplace_back(stream);
            workers_.emplace_back([this, i] {
                while (true) {
                    std::function<void(std::shared_ptr<c10::cuda::CUDAStream>)> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock, [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty()) {
                            return;
                        }
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task(streams_[i]);
                }
            });
        }
    }

    template <class F, class... Args>
    auto Submit(F&& func, Args&&... args)
        -> std::future<std::invoke_result_t<F, std::shared_ptr<c10::cuda::CUDAStream>, Args...>> {
        using return_type = std::invoke_result_t<F, std::shared_ptr<c10::cuda::CUDAStream>, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type(std::shared_ptr<c10::cuda::CUDAStream>)>>(
            std::bind(std::forward<F>(func), std::placeholders::_1, std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("submit on stopped thread_pool");
            }

            tasks_.emplace([task](std::shared_ptr<c10::cuda::CUDAStream> stream) { (*task)(stream); });
        }
        condition_.notify_one();
        return res;
    }

    void WaitForTasks() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (tasks_.empty() || stop_) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    ~CUDAStreamThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

 private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void(std::shared_ptr<c10::cuda::CUDAStream>)>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_{};

    std::vector<std::shared_ptr<c10::cuda::CUDAStream>> streams_;
};

template <typename ResourceType>
class ResourceThreadPool {
 public:
    explicit ResourceThreadPool(
        std::function<ResourceType()> resource_creator,
        std::function<void(ResourceType&)> resource_destructor,
        size_t threads = std::thread::hardware_concurrency())
        : resource_destructor_(resource_destructor) {
        for (size_t i = 0; i < threads; ++i) {
            auto resource = resource_creator();
            resources_.push_back(resource);
            workers_.emplace_back([&, resource]() mutable {
                while (true) {
                    std::function<void(ResourceType&)> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock, [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty()) {
                            return;
                        }
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task(resource);
                }
            });
        }
    }

    template <class F, class... Args>
    auto Submit(F&& func, Args&&... args) -> std::future<std::invoke_result_t<F, ResourceType&, Args...>> {
        using return_type = std::invoke_result_t<F, ResourceType&, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type(ResourceType&, Args...)>>(
            std::bind(std::forward<F>(func), std::placeholders::_1, std::forward<Args>(args)...));

        std::future<return_type> ret = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("submit on stopped thread_pool");
            }

            tasks_.emplace([task](ResourceType& resource) { (*task)(resource); });
        }
        condition_.notify_one();
        return ret;
    }

    void WaitForTasks() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this] { return tasks_.empty(); });
    }

    ~ResourceThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
        for (ResourceType& res : resources_) {
            resource_destructor(res);
        }
    }

 private:
    bool stop_{};
    std::vector<ResourceType> resources_;
    std::function<void(ResourceType&)> resource_destructor_;
    std::vector<std::thread> workers_;

    std::queue<std::function<void(ResourceType&)>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
};

template <typename ResourceType>
class CUDAStreamResourceThreadPool {
 public:
    explicit CUDAStreamResourceThreadPool(
        std::function<ResourceType()> resource_creator,
        std::function<void(ResourceType&)> resource_destructor,
        size_t threads = std::thread::hardware_concurrency())
        : resource_destructor_(resource_destructor) {
        for (size_t i = 0; i < threads; ++i) {
            auto resource = resource_creator();
            resources_.emplace_back(resource);
            auto stream = GetSafeCudaStream();
            streams_.emplace_back(stream);
            workers_.emplace_back([this, i]() mutable {
                while (true) {
                    std::function<void(ResourceType&, std::shared_ptr<c10::cuda::CUDAStream>)> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock, [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty()) {
                            return;
                        }
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task(resources_[i], streams_[i]);
                }
            });
        }
    }

    template <class F, class... Args>
    auto Submit(F&& func, Args&&... args)
        -> std::future<std::invoke_result_t<F, ResourceType&, std::shared_ptr<c10::cuda::CUDAStream>, Args...>> {
        using return_type = std::invoke_result_t<F, ResourceType&, std::shared_ptr<c10::cuda::CUDAStream>, Args...>;

        auto task = std::make_shared<
            std::packaged_task<return_type(ResourceType&, std::shared_ptr<c10::cuda::CUDAStream>, Args...)>>(std::bind(
            std::forward<F>(func), std::placeholders::_1, std::placeholders::_2, std::forward<Args>(args)...));

        std::future<return_type> ret = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("submit on stopped thread_pool");
            }

            tasks_.emplace([task](ResourceType& resource, std::shared_ptr<c10::cuda::CUDAStream> stream) {
                (*task)(resource, stream);
            });
        }
        condition_.notify_one();
        return ret;
    }

    void WaitForTasks() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this] { return tasks_.empty(); });
    }

    ~CUDAStreamResourceThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
        for (ResourceType& res : resources_) {
            resource_destructor_(res);
        }
    }

 private:
    bool stop_{};
    std::vector<ResourceType> resources_;
    std::function<void(ResourceType&)> resource_destructor_;
    std::vector<std::thread> workers_;

    std::queue<std::function<void(ResourceType&, std::shared_ptr<c10::cuda::CUDAStream>)>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;

    std::vector<std::shared_ptr<c10::cuda::CUDAStream>> streams_;
};

// class MutexWaitQueueThreadPool {
//    public:
//     explicit MutexWaitQueueThreadPool(size_t thread_num = std::thread::hardware_concurrency())
//         : thread_pool_size_(thread_num) {
//         for (size_t i = 0; i < thread_num; ++i) {
//             worker_mutexs_.emplace_back(std::make_shared<Mutex>());
//             worker_assign_task_queues_.emplace_back(std::make_shared<Queue>());
//             worker_conditions_.emplace_back(std::make_shared<ConditionVariable>());
//             worker_spin_locks_.emplace_back(std::make_shared<SpinLock>());
//         }

//         for (size_t i = 0; i < thread_num; ++i) {
//             workers_.emplace_back([this, i] {
//                 auto mutex = this->worker_mutexs_.at(i);
//                 auto task_queue = this->worker_assign_task_queues_.at(i);
//                 auto condition = this->worker_conditions_.at(i);
//                 auto spin_lock = this->worker_spin_locks_.at(i);

//                 while (true) {
//                     std::function<void()> task;
//                     {
//                         // std::unique_lock<Mutex> lock(*mutex);
//                         // condition_->wait(lock, [this, &task_queue] { return this->stop_ || !task_queue->empty(); });

//                         while (true) {
//                             if (this->stop_) {
//                                 return;
//                             }
//                             if (!task_queue->empty()) {
//                                 break;
//                             }
//                             // std::this_thread::yield();
//                             std::this_thread::sleep_for(std::chrono::microseconds(1));
//                         }

//                         {
//                             // std::unique_lock<Mutex> lock(*mutex);
//                             SpinGuard lock(*spin_lock);
//                             task = std::move(task_queue->front());
//                             task_queue->pop();
//                         }
//                     }
//                     task();
//                 }
//             });
//         }
//     }

//     template <class F, class... Args>
//     inline auto submit(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
//         using return_type = typename std::invoke_result<F, Args...>::type;

//         auto task = std::make_shared<std::packaged_task<return_type()>>(
//             std::bind(std::forward<F>(f), std::forward<Args>(args)...));
//         std::future<return_type> res = task->get_future();

//         int submit_index = current_submit_thread_index_++;
//         submit_index = submit_index % thread_pool_size_;
//         current_submit_thread_index_ = current_submit_thread_index_ % thread_pool_size_;

//         auto mutex = worker_mutexs_.at(submit_index);
//         auto assign_task_queue = worker_assign_task_queues_.at(submit_index);
//         auto condition_ = worker_conditions_.at(submit_index);
//         auto spin_lock = worker_spin_locks_.at(submit_index);
//         {
//             // std::unique_lock<Mutex> lock(*mutex);
//             SpinGuard lock(*spin_lock);
//             if (stop_) {
//                 throw std::runtime_error("submit on stopped thread_pool");
//             }
//             assign_task_queue->emplace([task]() { (*task)(); });
//         }
//         condition_->notify_one();
//         return res;
//     }

//     void WaitForTasks() {
//         for (size_t i = 0; i < thread_pool_size_; ++i) {
//             auto mutex = this->worker_mutexs_.at(i);
//             auto task_queue = this->worker_assign_task_queues_.at(i);
//             auto condition_ = this->worker_conditions_.at(i);

//             {
//                 // std::unique_lock<Mutex> lock(*mutex);
//                 // condition_->wait(lock, [&task_queue] { return task_queue->empty(); });
//                 while (true) {
//                     if (task_queue->empty() || stop_) {
//                         break;
//                     }
//                     std::this_thread::sleep_for(std::chrono::microseconds(100));
//                 }
//             }
//         }
//     }

//     ~MutexWaitQueueThreadPool() {
//         for (size_t i = 0; i < thread_pool_size_; ++i) {
//             auto mutex = this->worker_mutexs_.at(i);
//             std::unique_lock<Mutex> lock(*mutex);
//             stop_ = true;
//         }

//         for (size_t i = 0; i < thread_pool_size_; ++i) {
//             auto condition_ = this->worker_conditions_.at(i);
//             auto& worker = this->workers_.at(i);
//             condition_->notify_all();
//             worker.join();
//         }
//     }

//    private:
//     size_t thread_pool_size_;
//     volatile int current_submit_thread_index_{};
//     std::vector<Thread> workers_;
//     std::vector<std::shared_ptr<Mutex>> worker_mutexs_;
//     std::vector<std::shared_ptr<SpinLock>> worker_spin_locks_;
//     std::vector<std::shared_ptr<Queue>> worker_assign_task_queues_;
//     std::vector<std::shared_ptr<ConditionVariable>> worker_conditions_;

//     volatile bool stop_{};
// };

} // namespace astate
