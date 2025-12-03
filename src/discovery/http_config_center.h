#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <httplib.h>

#include "discovery/service_discovery.h"

namespace astate {

class HttpRetryableException : public std::runtime_error {
 public:
    explicit HttpRetryableException(const std::string& msg)
        : std::runtime_error(msg) {}
};

class HttpConfigCenter : public ConfigCenter {
 public:
    HttpConfigCenter();
    HttpConfigCenter(std::string server_ip, int server_port);
    ~HttpConfigCenter() override;

    HttpConfigCenter(const HttpConfigCenter&) = delete;
    HttpConfigCenter& operator=(const HttpConfigCenter&) = delete;

    bool GetConfig(const std::string& key, std::string& value) override;
    bool SetConfig(const std::string& key, const std::string& value) override;
    bool RemoveConfig(const std::string& key) override;

    void LoadConfigFromEnv();

 private:
    bool
    MakeHttpRequest(const std::string& method, const std::string& key, const std::string& value, std::string& response);

    void BuildUrls();
    [[nodiscard]] std::string BuildUrl(const std::string& path) const;

    [[nodiscard]] std::unique_ptr<httplib::Client> CreateClient() const;

    static constexpr int kHttpConnectTimeout = 60;
    static constexpr int kHttpReadTimeout = 60;
    std::string server_ip_;
    int server_port_ = 0;
    std::string base_url_;
    std::string put_base_url_;
    std::string get_base_url_;
    std::string delete_base_url_;
};

} // namespace astate
