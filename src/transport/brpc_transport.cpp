#include "transport/brpc_transport.h"

#include <chrono>
#include <cstdlib>
#include <thread>

#include <brpc/channel.h>
#include <brpc/server.h>
#include <gflags/gflags.h>
#include <spdlog/spdlog.h>

#include "transport/brpc_transfer_service.h"

namespace astate {
BrpcTransport::BrpcTransport()
    : server_(new brpc::Server()),
      service_impl_(new RpcTransferServiceImpl()) {
}
BrpcTransport::~BrpcTransport() {
    Stop();
}

bool BrpcTransport::Start(const Options& options) {
    google::SetCommandLineOption("log_level", "WARNING");
    google::SetCommandLineOption("log_verbose", "0");

    if (is_running_) {
        SPDLOG_WARN("BrpcTransport: Already running");
        return true;
    }

    max_retries_ = astate::GetOptionValue<int>(options, astate::BRPC_TRANSPORT_MAX_RETRIES);
    timeout_ms_ = astate::GetOptionValue<int>(options, astate::BRPC_TRANSPORT_TIMEOUT_MS);

    SPDLOG_INFO("BrpcTransport: Config loaded - max_retries: {}, timeout_ms: {}", max_retries_, timeout_ms_);

    for (auto& handler : handlers_) {
        service_impl_->RegisterHandler(handler.first, handler.second);
    }

    server_->AddService(service_impl_.get(), brpc::SERVER_DOESNT_OWN_SERVICE);

    brpc::ServerOptions server_options;
    server_options.max_concurrency = 1024;
    int port = 0;
    bool fixed_port = astate::GetOptionValue<bool>(options, "TRANSFER_ENGINE_SERVICE_FIXED_PORT");
    if (fixed_port) {
        port = astate::GetOptionValue<int>(options, "TRANSFER_ENGINE_SERVICE_PORT");
        if (port <= 0) {
            port = kPortStart;
        }
        auto ret = server_->Start(port, &server_options);
        if (ret != 0) {
            SPDLOG_ERROR("BrpcTransport failed to start on port {}", port);
            return false;
        }
    } else {
        port = kPortStart + (rand() % 1000);
        brpc::PortRange port_range(port, port + kBindPortMaxRetry);
        auto ret = server_->Start(port_range, &server_options);
        if (ret != 0) {
            SPDLOG_ERROR("BrpcTransport failed to start on port {}", port);
            return false;
        }
    }
    if (server_->IsRunning()) {
        port = server_->listen_address().port;
        bind_port_ = port;
        SPDLOG_INFO("BrpcTransport started on port {}", endpoint2str(server_->listen_address()).c_str());
        is_running_ = true;

        server_thread_ = std::thread([this]() {
            server_->RunUntilAskedToQuit();
            SPDLOG_INFO("BrpcTransport server thread stopped");
        });

        cleanup_running_ = true;
        cleanup_thread_ = std::thread([this]() {
            while (cleanup_running_.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(30));
                if (cleanup_running_.load()) {
                    CleanupInvalidConnections();
                }
            }
            SPDLOG_INFO("BrpcTransport cleanup thread stopped");
        });

        std::ostringstream id_oss;
        id_oss << server_thread_.get_id();
        SPDLOG_INFO("BrpcTransport started on port {} with thread {}", bind_port_, id_oss.str());
        // 等待server启动
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return true;
    }
    SPDLOG_ERROR("BrpcTransport failed to start on port {}", port);
    return false;
}

void BrpcTransport::Stop() {
    if (!is_running_) {
        SPDLOG_INFO("BrpcTransport: Already stopped");
        return;
    }

    SPDLOG_INFO("BrpcTransport: Stopping server...");

    // 设置停止标志
    is_running_ = false;

    // 停止清理线程
    cleanup_running_ = false;
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }

    if (server_) {
        SPDLOG_INFO("BrpcTransport: Calling server_->Stop(1s)...");
        if (server_->Stop(1000) == 0) {
            SPDLOG_WARN("BrpcTransport: server_->Stop(1s) failed");
        }

        pthread_cancel(server_thread_.native_handle());
        SPDLOG_INFO("BrpcTransport: Waiting for server thread to join...");
        if (server_thread_.joinable()) {
            server_thread_.join();
        }

        SPDLOG_INFO("BrpcTransport: Calling server_->Join()...");
        if (server_->Join() != 0) {
            SPDLOG_WARN("BrpcTransport: server_->Join() failed");
        }
    }

    SPDLOG_INFO("BrpcTransport: Server stopped successfully");

    // 清理stub缓存
    {
        std::lock_guard<std::mutex> lock(stub_mutex_);
        SPDLOG_INFO("BrpcTransport: Clearing {} cached stubs", stub_cache_.size());
        stub_cache_.clear();
    }

    SPDLOG_INFO("BrpcTransport: Stop completed");
}

int BrpcTransport::GetBindPort() const {
    return bind_port_;
}

astate::proto::RpcTransferService_Stub* BrpcTransport::GetOrCreateStub(const std::string& server_addr) {
    std::lock_guard<std::mutex> lock(stub_mutex_);


    auto it = stub_cache_.find(server_addr);
    if (it != stub_cache_.end()) {
        return it->second.second.get();
    }

    auto channel = std::make_unique<brpc::Channel>();
    brpc::ChannelOptions opts;
    opts.timeout_ms = 10000;
    opts.connect_timeout_ms = 10000;
    opts.max_retry = 5;
    opts.connection_group = "astate_brpc_group";

    if (channel->Init(server_addr.c_str(), nullptr, &opts) != 0) {
        SPDLOG_ERROR("BrpcTransport: channel init failed for {}", server_addr);
        return nullptr;
    }

    auto stub = std::make_unique<astate::proto::RpcTransferService_Stub>(channel.get());

    SPDLOG_INFO("BrpcTransport: created new stub for {}", server_addr);

    astate::proto::RpcTransferService_Stub* stub_ptr = stub.get();
    stub_cache_[server_addr] = std::make_pair(std::move(channel), std::move(stub));

    return stub_ptr;
}

bool BrpcTransport::Send(
    const std::string& request_name,
    const void* send_data,
    size_t send_size,
    const std::string& remote_host,
    int remote_port,
    const ExtendInfo* /*extend_info*/) {
    std::string server_addr = remote_host + ":" + std::to_string(remote_port);

    astate::proto::RpcTransferService_Stub* stub = GetOrCreateStub(server_addr);
    if (stub == nullptr) {
        SPDLOG_ERROR("BrpcTransport: failed to get stub for {}", server_addr);
        return false;
    }

    for (int retry = 0; retry < max_retries_; ++retry) {
        brpc::Controller cntl;

        cntl.set_timeout_ms(timeout_ms_);

        astate::proto::TransferData req;
        req.set_request_name(request_name);
        req.set_data(std::string(static_cast<const char*>(send_data), send_size));

        astate::proto::StatusReply resp;

        if (retry > 0) {
            SPDLOG_INFO(
                "BrpcTransport sending request: {} to {} with {} bytes "
                "(attempt {}/{})",
                request_name,
                server_addr,
                send_size,
                (retry + 1),
                max_retries_);
        }

        if (stub == nullptr) {
            SPDLOG_ERROR("BrpcTransport: Invalid stub pointer for {}", server_addr);
            return false;
        }

        stub->Send(&cntl, &req, &resp, nullptr);

        if (cntl.Failed()) {
            SPDLOG_WARN(
                "BrpcTransport send failed (attempt {}/{}): {} to {}",
                (retry + 1),
                max_retries_,
                cntl.ErrorText(),
                server_addr);

            if (cntl.ErrorCode() != 0) {
                SPDLOG_INFO(
                    "Connection error detected, removing stub from cache for "
                    "{}",
                    server_addr);
                {
                    std::lock_guard<std::mutex> lock(stub_mutex_);
                    stub_cache_.erase(server_addr);
                }

                stub = GetOrCreateStub(server_addr);
                if (stub == nullptr) {
                    SPDLOG_ERROR("BrpcTransport: failed to recreate stub for {}", server_addr);
                    return false;
                }
            }

            if (retry == max_retries_ - 1) {
                SPDLOG_ERROR(
                    "BrpcTransport send failed after {} attempts: {} to {}",
                    max_retries_,
                    cntl.ErrorText(),
                    server_addr);
                return false;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        bool success = resp.code() == 0;
        if (!success) {
            SPDLOG_INFO(
                "BrpcTransport send {} to {}, response: {}",
                (success ? "success" : "failed"),
                server_addr,
                resp.message());
        }
        return success;
    }

    return false;
}

void BrpcTransport::CleanupInvalidConnections() {
    std::lock_guard<std::mutex> lock(stub_mutex_);

    if (stub_cache_.empty()) {
        return;
    }

    auto it = stub_cache_.begin();
    while (it != stub_cache_.end()) {
        ++it;
    }

    SPDLOG_INFO("BrpcTransport: Connection cleanup completed, {} stubs remaining", stub_cache_.size());
}
} // namespace astate
