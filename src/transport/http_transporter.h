#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include <httplib.h>

#include "common/option.h"
#include "common/queue_utils.h"
#include "common/thread_pool.h"
#include "transport/base_transport.h"

namespace astate {

class HTTPTransporter : public BaseControlTransport {
 public:
    HTTPTransporter() = default;
    ~HTTPTransporter() override;

    [[nodiscard]] bool Start(const Options& options) override;

    void Stop() override;

    [[nodiscard]] bool Send(
        const std::string& request_name,
        const void* send_data,
        size_t send_size,
        const std::string& remote_host,
        int remote_port,
        const ExtendInfo* extend_info) override;

    int GetBindPort() const override { return local_port_; }

    std::string GetLocalHost() const { return local_host_; }

    static void HandlePing(const httplib::Request& req, httplib::Response& res);

    bool StartHttpService(const Options& options);
    bool CheckHttpService() const;

 private:
    Options options_;

    std::string local_host_;
    int local_port_{0};

    std::unique_ptr<httplib::Server> http_server_;
    std::thread server_thread_;

    std::unique_ptr<ThreadPool> receive_thread_pool_;

    // path -> message queue
    std::unordered_map<std::string, MessageQueue<httplib::Request>> receive_queues_;
    bool SetupServerWithRetry();
};

inline static std::string ToHttpPath(const std::string& request_name) {
    // "request_name" -> "/request_name", .e.g. "ping" -> "/ping"
    return "/" + request_name;
}

inline static std::string ToRequestName(const std::string& path) {
    // "/request_name" -> "request_name", .e.g. "/ping" -> "ping"
    return path.substr(1);
}

inline static const char* VoidToChar(const void* data) {
    return static_cast<const char*>(data);
}

inline static const void* CharToVoid(const char* data) {
    return static_cast<const void*>(data);
}

static std::string GetHttpPathFromExtendInfo(const ExtendInfo* extend_info) {
    // HTTP Transport Extend info:[http_path]
    if (extend_info == nullptr || extend_info->size() == 0) {
        SPDLOG_ERROR("Extend info is null or empty");
        return "";
    }
    return std::any_cast<const std::string&>(extend_info->at(0));
}

inline static ExtendInfo SetHttpPathToExtendInfo(const std::string& path) {
    ExtendInfo extend_info;
    extend_info.emplace_back(path);
    return extend_info;
}

} // namespace astate
