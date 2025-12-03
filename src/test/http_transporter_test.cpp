#include "transport/http_transporter.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "common/network_utils.h"
#include "common/option.h"

namespace astate {
class HTTPTransporterTest : public ::testing::Test {
 public:
    std::string TEST_HOST = GetLocalHostnameOrIP();
    static constexpr int TEST_PORT = 8080;
    static constexpr int TEST_CLIENT_PORT = 8081;

    void SetUp() override {
        // 设置测试服务器配置
        server_options_[TRANSFER_ENGINE_SERVICE_ADDRESS] = std::string(TEST_HOST);
        server_options_[TRANSFER_ENGINE_SERVICE_PORT] = std::to_string(TEST_PORT);
        server_options_[TRANSFER_ENGINE_SERVICE_FIXED_PORT] = "true";
        server_options_[TRANSFER_ENGINE_READ_THREAD_NUM] = std::to_string(4);
        server_options_[TRANSFER_ENGINE_SERVICE_SKIP_DISCOVERY] = "true";
        server_options_[TRANSPORT_RECEIVE_RETRY_COUNT] = "1";
        server_options_[TRANSPORT_RECEIVE_RETRY_SLEEP_MS] = "1000";
        server_options_[TRANSPORT_SEND_RETRY_COUNT] = "1";
        server_options_[TRANSPORT_SEND_RETRY_SLEEP_MS] = "1000";
        SPDLOG_INFO("Server options: {}", ToString(server_options_));

        // 设置测试客户端配置
        client_options_[TRANSFER_ENGINE_SERVICE_ADDRESS] = std::string(TEST_HOST);
        client_options_[TRANSFER_ENGINE_SERVICE_PORT] = std::to_string(TEST_CLIENT_PORT);
        client_options_[TRANSFER_ENGINE_SERVICE_FIXED_PORT] = "true";
        client_options_[TRANSFER_ENGINE_READ_THREAD_NUM] = std::to_string(4);
        client_options_[TRANSFER_ENGINE_SERVICE_SKIP_DISCOVERY] = "true";
        client_options_[TRANSPORT_RECEIVE_RETRY_COUNT] = "1";
        client_options_[TRANSPORT_RECEIVE_RETRY_SLEEP_MS] = "1000";
        client_options_[TRANSPORT_SEND_RETRY_COUNT] = "1";
        client_options_[TRANSPORT_SEND_RETRY_SLEEP_MS] = "1000";
        SPDLOG_INFO("Client options: {}", ToString(client_options_));

        // 禁用动态绑定, 使用静态绑定
        server_transporter_ = std::make_unique<HTTPTransporter>();
        client_transporter_ = std::make_unique<HTTPTransporter>();
    }

    void TearDown() override {
        if (server_transporter_->IsRunning()) {
            server_transporter_->Stop();
        }
        if (client_transporter_->IsRunning()) {
            client_transporter_->Stop();
        }

        // 给服务器一些时间来完全停止
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void WaitForServerReady() {
        // 等待服务器准备就绪
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    Options server_options_;
    Options client_options_;
    std::unique_ptr<HTTPTransporter> server_transporter_;
    std::unique_ptr<HTTPTransporter> client_transporter_;
};

// 测试基本启动和停止功能
TEST_F(HTTPTransporterTest, BasicStartStop) {
    // 初始状态应该是未运行
    EXPECT_FALSE(server_transporter_->IsRunning());

    // 启动服务器
    bool start_result = server_transporter_->Start(server_options_);
    EXPECT_TRUE(start_result);
    SPDLOG_INFO("Server started");

    WaitForServerReady();

    // 检查运行状态
    EXPECT_TRUE(server_transporter_->IsRunning());
    SPDLOG_INFO("Server is running");

    // 检查本地地址和端口
    EXPECT_EQ(server_transporter_->GetLocalHost(), TEST_HOST);
    EXPECT_EQ(server_transporter_->GetBindPort(), TEST_PORT);
    SPDLOG_INFO("Server local host: {}", server_transporter_->GetLocalHost());
    SPDLOG_INFO("Server local port: {}", server_transporter_->GetBindPort());

    // 停止服务器
    server_transporter_->Stop();
    EXPECT_FALSE(server_transporter_->IsRunning());
}

// 测试处理器注册和HTTP消息处理
TEST_F(HTTPTransporterTest, HandlerRegistrationAndMessageHandling) {
    // 用于测试的共享状态
    std::atomic<bool> handler_called{false};
    std::string received_message;
    std::mutex msg_mutex;
    std::condition_variable msg_cv;

    const std::string test_path = "test_endpoint";
    const std::string test_message = "Hello, World!";

    // 注册处理器
    server_transporter_->RegisterHandler(
        test_path, [&](const std::string& request, const void* message_data, size_t message_size) {
            std::lock_guard<std::mutex> lock(msg_mutex);
            received_message = std::string(static_cast<const char*>(message_data), message_size);
            handler_called = true;
            msg_cv.notify_one();
            return ResponseStatus{true, "Success", ExtendInfo{}};
        });

    // 启动服务器
    EXPECT_TRUE(server_transporter_->Start(server_options_));
    WaitForServerReady();

    // 发送消息
    bool send_result = server_transporter_->Send(
        test_path, test_message.c_str(), test_message.size(), TEST_HOST, TEST_PORT, nullptr);
    EXPECT_TRUE(send_result);

    // 等待处理器被调用
    std::unique_lock<std::mutex> lock(msg_mutex);
    bool notified = msg_cv.wait_for(lock, std::chrono::seconds(5), [&] { return handler_called.load(); });

    EXPECT_TRUE(notified);
    EXPECT_TRUE(handler_called);
    EXPECT_EQ(received_message, test_message);
}

// 测试多个处理器注册
TEST_F(HTTPTransporterTest, MultipleHandlerRegistration) {
    std::atomic<int> handler1_calls{0};
    std::atomic<int> handler2_calls{0};
    std::mutex call_mutex;
    std::condition_variable call_cv;

    const std::string path1 = "endpoint1";
    const std::string path2 = "endpoint2";

    // 注册第一个处理器
    server_transporter_->RegisterHandler(
        path1, [&](const std::string& request, const void* message_data, size_t message_size) {
            handler1_calls++;
            call_cv.notify_one();
            return ResponseStatus{true, "Handler1", ExtendInfo{}};
        });

    // 注册第二个处理器
    server_transporter_->RegisterHandler(
        path2, [&](const std::string& request, const void* message_data, size_t message_size) {
            handler2_calls++;
            call_cv.notify_one();
            return ResponseStatus{true, "Handler2", ExtendInfo{}};
        });

    // 启动服务器
    EXPECT_TRUE(server_transporter_->Start(server_options_));
    WaitForServerReady();

    // 向两个端点发送消息
    EXPECT_TRUE(server_transporter_->Send(path1, "test1", 4, TEST_HOST, TEST_PORT, nullptr));
    EXPECT_TRUE(server_transporter_->Send(path2, "test2", 4, TEST_HOST, TEST_PORT, nullptr));

    // 等待处理器被调用
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_EQ(handler1_calls.load(), 1);
    EXPECT_EQ(handler2_calls.load(), 1);
}

// 测试重复注册处理器的警告
TEST_F(HTTPTransporterTest, DuplicateHandlerRegistration) {
    const std::string test_path = "duplicate";

    // 注册第一个处理器
    server_transporter_->RegisterHandler(
        test_path, [&](const std::string& request, const void* message_data, size_t message_size) {
            return ResponseStatus{true, "First", ExtendInfo{}};
        });

    // 注册重复的处理器（应该产生警告，但不会失败）
    server_transporter_->RegisterHandler(
        test_path, [&](const std::string& request, const void* message_data, size_t message_size) {
            return ResponseStatus{true, "Second", ExtendInfo{}};
        });

    // 启动服务器应该成功
    bool start_result = server_transporter_->Start(server_options_);
    SPDLOG_INFO("Start result: {}", start_result);
    EXPECT_TRUE(start_result);
}

// 测试发送HTTP消息到远程服务器
TEST_F(HTTPTransporterTest, SendHttpMessageToRemoteServer) {
    std::atomic<bool> message_received{false};
    std::string received_body;
    std::mutex recv_mutex;
    std::condition_variable recv_cv;

    const std::string test_path = "remote_test";
    const std::string test_message = "Remote message";

    // 设置服务器端处理器
    server_transporter_->RegisterHandler(
        test_path, [&](const std::string& request, const void* message_data, size_t message_size) {
            std::lock_guard<std::mutex> lock(recv_mutex);
            received_body = std::string(static_cast<const char*>(message_data), message_size);
            message_received = true;
            recv_cv.notify_one();
            return ResponseStatus{true, "Received", ExtendInfo{}};
        });

    // 启动服务器
    EXPECT_TRUE(server_transporter_->Start(server_options_));
    WaitForServerReady();

    // 启动客户端
    EXPECT_TRUE(client_transporter_->Start(client_options_));
    WaitForServerReady();

    // 从客户端发送消息到服务器
    bool send_result = client_transporter_->Send(
        test_path, test_message.c_str(), test_message.size(), TEST_HOST, TEST_PORT, nullptr);
    EXPECT_TRUE(send_result);

    // 等待消息被接收
    std::unique_lock<std::mutex> lock(recv_mutex);
    bool notified = recv_cv.wait_for(lock, std::chrono::seconds(5), [&] { return message_received.load(); });

    EXPECT_TRUE(notified);
    EXPECT_EQ(received_body, test_message);
}

// 测试JSON消息处理
TEST_F(HTTPTransporterTest, JsonMessageHandling) {
    std::atomic<bool> json_received{false};
    std::string received_json;
    std::mutex json_mutex;
    std::condition_variable json_cv;

    const std::string test_path = "json_test";
    const std::string test_json = R"({"key": "value", "number": 42})";

    // 注册JSON处理器
    server_transporter_->RegisterHandler(
        test_path, [&](const std::string& request, const void* message_data, size_t message_size) {
            std::lock_guard<std::mutex> lock(json_mutex);
            received_json = std::string(static_cast<const char*>(message_data), message_size);
            json_received = true;
            json_cv.notify_one();
            return ResponseStatus{true, "Success", ExtendInfo{}};
        });

    // 启动服务器
    EXPECT_TRUE(server_transporter_->Start(server_options_));
    WaitForServerReady();

    // 发送JSON消息
    bool send_result
        = server_transporter_->Send(test_path, test_json.c_str(), test_json.size(), TEST_HOST, TEST_PORT, nullptr);
    EXPECT_TRUE(send_result);

    // 等待JSON被接收
    std::unique_lock<std::mutex> lock(json_mutex);
    bool notified = json_cv.wait_for(lock, std::chrono::seconds(5), [&] { return json_received.load(); });

    EXPECT_TRUE(notified);
    EXPECT_EQ(received_json, test_json);
}

// 测试错误处理
TEST_F(HTTPTransporterTest, ErrorHandling) {
    // 测试发送到不存在的服务器
    EXPECT_TRUE(client_transporter_->Start(client_options_));
    bool send_result = client_transporter_->Send("test", "test", 4, "127.0.0.1", 9999, nullptr);
    EXPECT_FALSE(send_result);

    // 测试发送到无效的路径
    EXPECT_TRUE(server_transporter_->Start(server_options_));
    WaitForServerReady();

    bool invalid_path_result = client_transporter_->Send("nonexistent", "test", 4, TEST_HOST, TEST_PORT, nullptr);
    // 这应该失败，因为没有注册处理器
    EXPECT_FALSE(invalid_path_result);
}

// 测试并发消息处理
TEST_F(HTTPTransporterTest, ConcurrentMessageHandling) {
    std::atomic<int> total_messages{0};
    std::mutex count_mutex;
    std::condition_variable count_cv;

    const std::string test_path = "concurrent_test";
    const int num_messages = 10;

    // 注册处理器
    server_transporter_->RegisterHandler(
        test_path, [&](const std::string& request, const void* message_data, size_t message_size) {
            std::lock_guard<std::mutex> lock(count_mutex);
            total_messages++;
            if (total_messages == num_messages) {
                count_cv.notify_one();
            }
            return ResponseStatus{true, "OK", ExtendInfo{}};
        });

    // 启动服务器
    EXPECT_TRUE(server_transporter_->Start(server_options_));
    WaitForServerReady();

    // 并发发送多个消息
    std::vector<std::thread> threads;
    for (int i = 0; i < num_messages; ++i) {
        threads.emplace_back([&, i]() {
            std::string message = "Message " + std::to_string(i);
            server_transporter_->Send(test_path, message.c_str(), message.size(), TEST_HOST, TEST_PORT, nullptr);
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 等待所有消息被处理
    std::unique_lock<std::mutex> lock(count_mutex);
    bool all_received
        = count_cv.wait_for(lock, std::chrono::seconds(10), [&] { return total_messages.load() == num_messages; });

    EXPECT_TRUE(all_received);
    EXPECT_EQ(total_messages.load(), num_messages);
}

// 测试服务器重启
TEST_F(HTTPTransporterTest, ServerRestart) {
    // 第一次启动
    EXPECT_TRUE(server_transporter_->Start(server_options_));
    WaitForServerReady();
    EXPECT_TRUE(server_transporter_->IsRunning());

    // 停止服务器
    server_transporter_->Stop();
    EXPECT_FALSE(server_transporter_->IsRunning());

    // 等待一段时间确保端口被释放
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // 重新启动
    EXPECT_TRUE(server_transporter_->Start(server_options_));
    WaitForServerReady();
    EXPECT_TRUE(server_transporter_->IsRunning());
}
} // namespace astate
