#include "http_transporter.h"

#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include <httplib.h>

#include <spdlog/spdlog.h>

#include "common/network_utils.h"
#include "common/option.h"
#include "common/retry/counting_retry.h"
#include "common/retry/retry_utils.h"
#include "common/thread_pool.h"
#include "transport/base_transport.h"

namespace astate {

constexpr const char* PING_REQUEST = "ping";

HTTPTransporter::~HTTPTransporter() {
    // Ensure cleanup happens in destructor
    if (is_running_) {
        Stop();
    }
}

bool HTTPTransporter::Start(const Options& options) {
    SPDLOG_INFO("Starting HTTP service...");

    options_ = options;

    CountingRetry retry_policy(3);

    try {
        RetryUtils::Retry(
            "check HTTP service",
            [this, &retry_policy, &options]() {
                StartHttpService(options);
                SPDLOG_INFO("Checking HTTP service health, attempt #{}", retry_policy.GetAttemptCount());
                if (!CheckHttpService()) {
                    SPDLOG_WARN("HTTP service health check failed on attempt #{}", retry_policy.GetAttemptCount());
                    // release resources
                    Stop();
                    throw std::runtime_error("HTTP service health check failed");
                }

                SPDLOG_INFO("HTTP service start passed on attempt #{}", retry_policy.GetAttemptCount());
                return;
            },
            retry_policy);

        SPDLOG_INFO("HTTP service start completed successfully");
        return true;

    } catch (const std::exception& e) {
        SPDLOG_ERROR("HTTP service start failed after {} attempts: {}", retry_policy.GetAttemptCount(), e.what());
        return false;
    }
}

bool HTTPTransporter::StartHttpService(const Options& options) {
    // Get options
    local_host_ = GetLocalHostnameOrIP();

    // Initialize HTTP server and thread pool
    http_server_ = std::make_unique<httplib::Server>();
    receive_thread_pool_ = std::make_unique<ThreadPool>(GetOptionValue<int>(options, TRANSFER_ENGINE_READ_THREAD_NUM));

    // Register handlers to HTTP server BEFORE starting the server thread
    for (const auto& handler : handlers_) {
        const std::string path = ToHttpPath(handler.first);
        const Handler func = handler.second;

        // Register both GET and POST handlers for each path
        http_server_->Get(path, [func](const httplib::Request& req, httplib::Response& res) {
            ResponseStatus status = func(ToRequestName(req.path), CharToVoid(req.body.c_str()), req.body.size());
            res.status = status.success ? 200 : 500;
            res.body = status.status_message;
        });

        http_server_->Post(path, [func](const httplib::Request& req, httplib::Response& res) {
            ResponseStatus status = func(ToRequestName(req.path), CharToVoid(req.body.c_str()), req.body.size());
            res.status = status.success ? 200 : 500;
            res.body = status.status_message;
        });
    }
    // Register ping handler
    http_server_->Get(PING_REQUEST, [](const httplib::Request& req, httplib::Response& res) {
        astate::HTTPTransporter::HandlePing(req, res);
        return;
    });

    // Setup server based on dynamic bind flag
    bool fixed_port = GetOptionValue<bool>(options, TRANSFER_ENGINE_SERVICE_FIXED_PORT);
    if (!fixed_port) {
        // Setup server with port retry mechanism
        if (!SetupServerWithRetry()) {
            SPDLOG_ERROR("Failed to setup HTTP server after retry");
            return false;
        }
    } else {
        // 从Options读取端口
        int port = kPortStart;
        auto it = options.find(TRANSFER_ENGINE_SERVICE_PORT);
        if (it != options.end()) {
            try {
                port = std::stoi(it->second);
            } catch (...) {
                SPDLOG_WARN(
                    "Invalid TRANSFER_ENGINE_SERVICE_PORT, fallback to "
                    "default: {}",
                    kPortStart);
                port = kPortStart;
            }
        } else {
            SPDLOG_WARN("TRANSFER_ENGINE_SERVICE_PORT not set, fallback to default: {}", kPortStart);
        }
        local_port_ = port;
        SPDLOG_INFO("HTTPTransporter using fixed port {} (dynamic bind disabled)", local_port_);
        // 启动监听线程
        server_thread_ = std::thread([this, port]() {
            SPDLOG_INFO("HTTP server listen on {}:{}", local_host_, port);
            http_server_->listen(local_host_, port);
        });
    }

    http_server_->wait_until_ready();
    is_running_ = true;
    SPDLOG_INFO("HTTPTransporter started on port {}", local_port_);
    return true;
}

void HTTPTransporter::Stop() {
    // Stop HTTP server regardless of dynamic bind setting
    if (http_server_) {
        http_server_->stop();
    }

    // Join server thread if it's running
    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    // Wait for receive thread pool tasks to complete
    if (receive_thread_pool_) {
        receive_thread_pool_->WaitForTasks();
    }

    is_running_ = false;
    SPDLOG_INFO("HTTPTransporter stopped.");
}

bool HTTPTransporter::Send(
    const std::string& request_name,
    const void* send_data,
    size_t send_size,
    const std::string& remote_host,
    int remote_port,
    const ExtendInfo* /*extend_info*/) {
    std::string http_path = ToHttpPath(request_name);
    if (http_path.empty()) {
        SPDLOG_ERROR("HTTP path is empty");
        return false;
    }

    int retry_count = GetOptionValue<int>(options_, TRANSPORT_SEND_RETRY_COUNT);
    int retry_sleep_ms = GetOptionValue<int>(options_, TRANSPORT_SEND_RETRY_SLEEP_MS);
    CountingAndSleepRetryPolicy policy(retry_count, retry_sleep_ms);
    auto remote_addr = remote_host + ":" + std::to_string(remote_port);

    auto send_func = [&]() -> bool {
        httplib::Client client(remote_host, remote_port);
        std::string data(static_cast<const char*>(send_data), send_size);
        SPDLOG_INFO("Send message: {} to {}: {}", http_path, remote_addr, data);
        auto res = client.Post(http_path, data, "application/json; charset=utf-8");
        if (res && res->status == 200) {
            SPDLOG_INFO("Send message: {} to {} success", http_path, remote_addr);
            return true;
        }
        if (res) {
            SPDLOG_ERROR(
                "Failed to send message, status: {}, body: {}, url: {}/{}",
                res->status,
                res->body,
                remote_addr,
                http_path);
        } else {
            SPDLOG_ERROR(
                "Failed to send message: No response received (connection "
                "failed), url: {}/{}",
                remote_addr,
                http_path);
        }
        // retry by throw exception
        throw std::runtime_error("Failed to send message");
    };

    try {
        return RetryUtils::Retry<bool>("HTTPTransporter::Send", send_func, policy);
    } catch (const NonRetryableException& e) {
        SPDLOG_ERROR("Non-retryable error: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to send message after retry: {}", e.what());
        return false;
    }
    return true;
}

void HTTPTransporter::HandlePing(const httplib::Request& req, httplib::Response& res) {
    try {
        SPDLOG_INFO("Received ping request from: {}", req.get_header_value("Host"));

        res.status = 200;
        res.body = "pong";
        res.set_header("Content-Type", "text/plain");

        SPDLOG_INFO("Ping request handled successfully");
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to handle ping request: {}", e.what());
        res.status = 500;
        res.body = "Internal server error";
    }
}

bool HTTPTransporter::SetupServerWithRetry() {
    static constexpr int kBindPortMaxRetry = 100;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000);
    int random_offset = dis(gen);
    int base_port = kPortStart + random_offset;
    bool bind_success = false;
    int attempt = 0;

    while (attempt < kBindPortMaxRetry && !bind_success) {
        int current_port = base_port + attempt;
        SPDLOG_INFO(
            "Attempt {}/{} - Trying to bind HTTP server on port {}", (attempt + 1), kBindPortMaxRetry, current_port);

        try {
            // Test if port is available by trying to bind
            auto bind_promise = std::make_shared<std::promise<bool>>();
            std::future<bool> bind_future = bind_promise->get_future();

            server_thread_ = std::thread([this, current_port, bind_promise]() {
                bool success = http_server_->bind_to_port(local_host_, current_port);
                if (success) {
                    SPDLOG_INFO("HttpThread: HTTP server started on {}:{}", local_host_, current_port);
                    bind_promise->set_value(success);
                }
                http_server_->listen_after_bind();
            });

            // Wait a moment to check if binding was successful, bind cost 0.7s when local machine test
            if (bind_future.wait_for(std::chrono::milliseconds(10000)) == std::future_status::ready) {
                local_port_ = current_port; // set local_port_ after binding
                bind_success = bind_future.get();
                if (bind_success) {
                    SPDLOG_INFO("Successfully bound HTTP server on port {}", current_port);
                    break;
                }
            }

            if (!bind_success) {
                SPDLOG_WARN("Failed to bind HTTP server on port {}", current_port);
                if (server_thread_.joinable()) {
                    http_server_->stop();
                    server_thread_.join();
                }
                http_server_.reset();
                http_server_ = std::make_unique<httplib::Server>();
                attempt++;
            }
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Exception while binding to port {}: {}", current_port, e.what());
            if (server_thread_.joinable()) {
                http_server_->stop();
                server_thread_.join();
            }
            http_server_.reset();
            http_server_ = std::make_unique<httplib::Server>();
            attempt++;
        }
    }

    if (!bind_success) {
        SPDLOG_ERROR(
            "Failed to bind HTTP server after {} attempts, tried ports {} to "
            "{}",
            kBindPortMaxRetry,
            base_port,
            (base_port + kBindPortMaxRetry - 1));
        return false;
    }

    return true;
}

bool HTTPTransporter::CheckHttpService() const {
    try {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        std::string local_host = GetLocalHost();
        int local_port = GetBindPort();
        std::string ping_url = "http://" + local_host + ":" + std::to_string(local_port) + PING_REQUEST;

        SPDLOG_INFO("Checking HTTP service at: {}", ping_url);

        httplib::Client client(local_host, local_port);
        client.set_connection_timeout(5);
        client.set_read_timeout(5);

        auto result = client.Get(PING_REQUEST);

        if (!result) {
            SPDLOG_ERROR("Failed to connect to HTTP service: {}", ping_url);
            return false;
        }

        if (result->status != 200) {
            SPDLOG_ERROR("HTTP service returned non-200 status: {}, body: {}", result->status, result->body);
            return false;
        }

        SPDLOG_INFO("HTTP service health check passed");
        return true;

    } catch (const std::exception& e) {
        SPDLOG_ERROR("Exception during HTTP service health check: {}", e.what());
        return false;
    }
}
} // namespace astate
