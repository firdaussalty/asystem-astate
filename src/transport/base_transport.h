#pragma once

#include <any>
#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "common/option.h"
#include "core/atensor.h"

namespace astate {


// Callback functions for send and receive
using SendCallback = std::function<bool(const void*, size_t)>;
using ReceiveCallback = std::function<bool(const void*, size_t)>;

using ExtendInfo = std::vector<std::any>;

/*
 * BaseTransport is an abstract class that provides a common interface
 * definition for all data plane transport services.
 */
class BaseDataTransport {
 public:
    virtual ~BaseDataTransport() = default;

    /*
     * Start the transport service.
     * @param config: The configuration of the transport service.
     * @return: True if the transport service is started successfully, false
     * otherwise.
     */
    virtual bool Start(const Options& options, const AParallelConfig& parallel_config) = 0;

    /*
     * Stop the transport service.
     */
    virtual void Stop() = 0;

    /*
     * Check if the transport service is running.
     * @return: True if the transport service is running, false otherwise.
     */
    [[nodiscard]] bool IsRunning() const { return is_running_; }

    /*
     * Get the port that the transport service is bound to.
     * @return: The port number, or 0 if not bound.
     */
    [[nodiscard]] virtual int GetBindPort() const = 0;

    ////////////////////// Sync send and receive //////////////////////
    /*
     * Send data to the remote endpoint synchronously.
     * @param send_data: The local data to send.
     * @param send_size: The size of the data to send.
     * @param remote_host: The host of the remote endpoint.
     * @param remote_port: The port of the remote endpoint.
     * @param extend_info: The extend information of the transport.
     * @return: True if the data is sent successfully, false otherwise.
     */
    [[nodiscard]] virtual bool Send(
        const void* send_data,
        size_t send_size,
        const std::string& remote_host,
        int remote_port,
        const ExtendInfo* extend_info)
        = 0;

    /*
     * Receive data from the remote endpoint synchronously.
     * @param recv_data: The local address to receive.
     * @param recv_size: The size of the data to receive.
     * @param remote_host: The host of the remote endpoint.
     * @param remote_port: The port of the remote endpoint.
     * @param extend_info: The extend information of the transport.
     * @return: The size of the data to receive.
     */
    [[nodiscard]] virtual bool Receive(
        const void* recv_data,
        size_t recv_size,
        const std::string& remote_host,
        int remote_port,
        const ExtendInfo* extend_info)
        = 0;

    ////////////////////// Async send and receive with callback
    /////////////////////////
    /*
     * Send data to the remote endpoint asynchronously.
     * @param send_data: The local address to send.
     * @param send_size: The size of the data to send.
     * @param remote_host: The host of the remote endpoint.
     * @param remote_port: The port of the remote endpoint.
     * @param extend_info: The extend information of the transport.
     * @param callback: The callback to be called after the data is sent.
     */
    virtual void AsyncSend(
        const void* send_data,
        size_t send_size,
        const std::string& remote_host,
        int remote_port,
        const ExtendInfo* extend_info,
        const SendCallback& callback)
        = 0;

    /*
     * Receive data from the remote endpoint asynchronously.
     * @param recv_data: The local address to receive.
     * @param recv_size: The size of the data to receive.
     * @param remote_host: The host of the remote endpoint.
     * @param remote_port: The port of the remote endpoint.
     * @param extend_info: The extend information of the transport.
     * @param callback: The callback to be called when the data is received.
     */
    virtual void AsyncReceive(
        const void* recv_data,
        size_t recv_size,
        const std::string& remote_host,
        int remote_port,
        const ExtendInfo* extend_info,
        const ReceiveCallback& callback)
        = 0;

 protected:
    // Whether the transport service is running
    volatile bool is_running_{false};
    static constexpr int kBindPortMaxRetry = 100;
};

struct ResponseStatus {
    bool success;
    std::string status_message;
    ExtendInfo extend_info;
};

// Handler for control plane transport
using Handler = std::function<ResponseStatus(const std::string&, const void*, size_t)>;

/*
 * BaseControlTransport is an abstract class that provides a common interface
 * definition for all control plane transport services.
 */
class BaseControlTransport {
 public:
    virtual ~BaseControlTransport() = default;
    /*
     * Start the transport service.
     * @param config: The configuration of the transport service.
     * @return: True if the transport service is started successfully, false
     * otherwise.
     */
    virtual bool Start(const Options& options) = 0;

    /*
     * Stop the transport service.
     */
    virtual void Stop() = 0;

    /*
     * Check if the transport service is running.
     * @return: True if the transport service is running, false otherwise.
     */
    bool IsRunning() const { return is_running_; }

    /*
     * Get the port that the transport service is bound to.
     * @return: The port number, or 0 if not bound.
     */
    virtual int GetBindPort() const = 0;

    /*
     * Register a handler for a specific request.
     * @param request_name: The name of the request.
     * @param handler: The handler registered to process the request.
     */
    void RegisterHandler(const std::string& request_name, const Handler& handler) {
        auto it = handlers_.find(request_name);
        if (it != handlers_.end()) {
            SPDLOG_WARN("Http handler already registered for request: {}", request_name);
        } else {
            handlers_.insert({request_name, handler});
            SPDLOG_INFO("Http handler registered for request: {}", request_name);
        }
    };

    /*
     * Get a handler for a specific request.
     * @param request_name: The name of the request.
     * @return: The handler for the request.
     */
    Handler GetHandler(const std::string& request_name) {
        auto it = handlers_.find(request_name);
        if (it != handlers_.end()) {
            return it->second;
        }
        return nullptr;
    }

    /*
     * Send control message to the remote endpoint synchronously.
     * @param request_name: The name of the request.
     * @param send_data: The local data to send.
     * @param send_size: The size of the data to send.
     * @param remote_host: The host of the remote endpoint.
     * @param remote_port: The port of the remote endpoint.
     * @param extend_info: The extend information of the transport.
     * @return: True if the data is sent successfully, false otherwise.
     */
    [[nodiscard]] virtual bool Send(
        const std::string& request_name,
        const void* send_data,
        size_t send_size,
        const std::string& remote_host,
        int remote_port,
        const ExtendInfo* extend_info)
        = 0;

 protected:
    // Whether the transport service is running
    volatile bool is_running_{false};
    static constexpr int kBindPortMaxRetry = 100;
    static constexpr int kPortStart = 52010;

    std::unordered_map<std::string, Handler> handlers_;
};

} // namespace astate
