
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <brpc/channel.h>
#include <brpc/server.h>

#include "protocol/gen/transfer_service.pb.h"
#include "transport/base_transport.h"
#include "transport/brpc_transfer_service.h"

namespace astate {

class BrpcTransport : public BaseControlTransport {
 public:
    BrpcTransport();

    ~BrpcTransport() override;

    bool Start(const Options& options) override;
    void Stop() override;
    int GetBindPort() const override;

    bool Send(
        const std::string& request_name,
        const void* send_data,
        size_t send_size,
        const std::string& remote_host,
        int remote_port,
        const ExtendInfo* extend_info) override;

 private:
    std::unique_ptr<brpc::Server> server_;
    std::unique_ptr<RpcTransferServiceImpl> service_impl_;
    int bind_port_ = 0;
    std::thread server_thread_;

    mutable std::mutex stub_mutex_;
    std::unordered_map<
        std::string,
        std::pair<std::unique_ptr<brpc::Channel>, std::unique_ptr<astate::proto::RpcTransferService_Stub>>>
        stub_cache_;

    std::thread cleanup_thread_;
    std::atomic<bool> cleanup_running_{false};
    std::chrono::steady_clock::time_point last_cleanup_time_;

    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_health_check_;
    std::unordered_map<std::string, int> consecutive_failures_;
    std::atomic<int> total_requests_{0};
    std::atomic<int> failed_requests_{0};

    static constexpr int kMaxConsecutiveFailures = 3;
    static constexpr int kHealthCheckIntervalMs = 60000;
    static constexpr int kCircuitBreakerTimeoutMs = 60000;

    int max_retries_ = 90;
    int timeout_ms_ = 10000;

    astate::proto::RpcTransferService_Stub* GetOrCreateStub(const std::string& server_addr);

    void CleanupInvalidConnections();
};

} // namespace astate
