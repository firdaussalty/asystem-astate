#include "discovery/http_config_center.h"

#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <httplib.h>

#include <spdlog/spdlog.h>

#include "common/retry/retry_utils.h"
#include "common/retry/timeout_retry.h"

namespace astate {

HttpConfigCenter::HttpConfigCenter() {
    LoadConfigFromEnv();
    BuildUrls();
}

HttpConfigCenter::HttpConfigCenter(std::string server_ip, int server_port)
    : server_ip_(std::move(server_ip)),
      server_port_(server_port) {
    BuildUrls();
}

HttpConfigCenter::~HttpConfigCenter() = default;

std::unique_ptr<httplib::Client> HttpConfigCenter::CreateClient() const {
    auto client = std::make_unique<httplib::Client>(server_ip_, server_port_);
    client->set_connection_timeout(kHttpConnectTimeout);
    client->set_read_timeout(kHttpReadTimeout);
    return client;
}

void HttpConfigCenter::LoadConfigFromEnv() {
    const char* ip_env = std::getenv("ASYSTEM_META_SERVER_IP");
    const char* port_env = std::getenv("ASYSTEM_META_SERVER_PORT");

    if (ip_env != nullptr) {
        server_ip_ = std::string(ip_env);
        SPDLOG_INFO("ASYSTEM_META_SERVER_IP: {}", server_ip_);
    } else {
        SPDLOG_ERROR("ASYSTEM_META_SERVER_IP not set, exit.");
        throw std::runtime_error("ASYSTEM_META_SERVER_IP not set in ENV");
    }

    if (port_env != nullptr) {
        try {
            server_port_ = std::stoi(port_env);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("ASYSTEM_META_SERVER_PORT not set, exit.");
            throw std::runtime_error("ASYSTEM_META_SERVER_PORT not set in ENV");
        }
    }
}

std::string HttpConfigCenter::BuildUrl(const std::string& path) const {
    std::ostringstream oss;
    oss << base_url_ << path;
    return oss.str();
}

bool HttpConfigCenter::SetConfig(const std::string& key, const std::string& value) {
    try {
        TimeoutRetry retry(30000, 1000);

        RetryUtils::Retry(
            "set config",
            [&]() {
                std::string response;
                bool success = MakeHttpRequest("PUT", key, value, response);
                SPDLOG_INFO("Successfully set config: {} = '{}'", key, value);
            },
            retry);
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR(
            "Failed to set config after retries: {}, error: {}, server_addr: "
            "{}:{}",
            key,
            e.what(),
            server_ip_,
            server_port_);
        throw;
    }
}

bool HttpConfigCenter::GetConfig(const std::string& key, std::string& value) {
    try {
        // 使用TimeoutRetry: 最多重试10次，每次重试间隔1秒，总时间10秒
        TimeoutRetry retry(10000, 1000);
        bool ret = false;

        RetryUtils::Retry(
            "get config",
            [&]() {
                std::string response;
                bool success = MakeHttpRequest("GET", key, "", response);

                ret = success;
                if (success) {
                    value = response;
                    SPDLOG_ERROR("Successfully got config: {} = {}", key, value);
                } else {
                    SPDLOG_ERROR("Key not found: {}", key);
                }
            },
            retry);
        SPDLOG_INFO("get_config: {} = '{}' got: {}", key, value, (bool)ret);
        return ret;
    } catch (const std::exception& e) {
        SPDLOG_ERROR(
            "Failed to get config after retries: {}, error: {}, server_addr: "
            "{}:{}",
            key,
            e.what(),
            server_ip_,
            server_port_);
        throw;
    }
}

bool HttpConfigCenter::RemoveConfig(const std::string& key) {
    try {
        // 使用TimeoutRetry: 最多重试10次，每次重试间隔1秒
        TimeoutRetry retry(10000, 1000);

        RetryUtils::Retry(
            "remove config",
            [&]() {
                std::string response;
                MakeHttpRequest("DELETE", key, "", response);
                SPDLOG_INFO("Successfully removed config: {}", key);
            },
            retry);
        return true;
    } catch (const std::exception& e) {
        SPDLOG_ERROR(
            "Failed to remove config after retries: {}, error: {}, "
            "server_addr: {}:{}",
            key,
            e.what(),
            server_ip_,
            server_port_);
        throw;
    }
}

bool HttpConfigCenter::MakeHttpRequest(
    const std::string& method, const std::string& key, const std::string& value, std::string& response) {
    auto client = CreateClient();
    if (!client) {
        throw astate::NonRetryableException("Failed to create HTTP client");
    }

    httplib::Result result;
    try {
        if (method == "PUT") {
            std::string url = put_base_url_ + "/" + key;
            result = client->Put(url, value, "application/octet-stream");
        } else if (method == "GET") {
            std::string url = get_base_url_ + "/" + key;
            result = client->Get(url);
        } else if (method == "DELETE") {
            std::string url = delete_base_url_ + "/" + key;
            result = client->Delete(url);
        } else {
            throw astate::NonRetryableException("Unsupported HTTP method: " + method);
        }

        auto error = result.error();

        // 根据错误类型决定是否可重试
        if (error != httplib::Error::Success) {
            std::string error_msg = "HTTP request failed for key: " + key + " - " + httplib::to_string(error);
            switch (error) {
                // 这些错误通常不应该重试
                case httplib::Error::BindIPAddress:
                case httplib::Error::UnsupportedMultipartBoundaryChars:
                case httplib::Error::Compression:
                case httplib::Error::SSLConnection:
                case httplib::Error::SSLLoadingCerts:
                case httplib::Error::SSLServerVerification:
                    SPDLOG_ERROR("{} (non-retryable error: {})", error_msg, httplib::to_string(error));
                    throw astate::NonRetryableException(error_msg + " - " + httplib::to_string(error));
                default:
                    SPDLOG_ERROR("{} (retryable error: {})", error_msg, httplib::to_string(error));
                    throw HttpRetryableException(error_msg + " - " + httplib::to_string(error));
            }
        }

        response = result->body;
        if (result->status == 200) {
            return true;
        }
        if (result->status == 404) {
            SPDLOG_INFO("Config not found key(404): {}", key);
            return false;
        }
        if (result->status >= 500 && result->status < 600) {
            std::string error_msg = "Server error (status " + std::to_string(result->status) + ") for key: " + key
                + ", response: " + result->body;
            SPDLOG_WARN("{}", error_msg);
            throw HttpRetryableException(error_msg);
        }
        if (result->status == 408 || result->status == 429) {
            std::string error_msg = "Temporary error (status " + std::to_string(result->status) + ") for key: " + key
                + ", response: " + result->body;
            SPDLOG_WARN("{}", error_msg);
            throw HttpRetryableException(error_msg);
        }
        std::string error_msg = "Client error (status " + std::to_string(result->status) + ") for key: " + key
            + ", response: " + result->body;
        SPDLOG_ERROR("{}", error_msg);
        throw astate::NonRetryableException(error_msg);

    } catch (const HttpRetryableException&) {
        throw;
    } catch (const astate::NonRetryableException&) {
        throw;
    } catch (const std::exception& e) {
        std::string error_msg = "HTTP request exception for key: " + key + " - " + e.what();
        SPDLOG_ERROR("{}", error_msg);

        throw HttpRetryableException(error_msg);
    }
}

void HttpConfigCenter::BuildUrls() {
    base_url_ = "http://" + server_ip_ + ":" + std::to_string(server_port_) + "/v1";
    put_base_url_ = BuildUrl("/put_binary");
    get_base_url_ = BuildUrl("/get_binary");
    delete_base_url_ = BuildUrl("/delete");
}

} // namespace astate
