#pragma once

#include <cstdlib>
#include <cstring>

#include <ifaddrs.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <spdlog/spdlog.h>
#include <sys/socket.h>

namespace astate {

inline std::string GetLocalHostnameOrIP() {
    // 1. POD_IP
    const char* pod_ip = std::getenv("POD_IP");
    if ((pod_ip != nullptr) && strlen(pod_ip) > 0) {
        SPDLOG_INFO("Using POD_IP from environment: {}", pod_ip);
        return pod_ip;
    }

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock >= 0) {
        struct sockaddr_in addr {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(80);
        addr.sin_addr.s_addr = inet_addr("8.8.8.8");

        if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0) {
            socklen_t addr_len = sizeof(addr);
            if (getsockname(sock, reinterpret_cast<struct sockaddr*>(&addr), &addr_len) == 0) {
                std::string ip_str(INET_ADDRSTRLEN, 0);
                if (inet_ntop(AF_INET, &addr.sin_addr, ip_str.data(), INET_ADDRSTRLEN) != nullptr) {
                    close(sock);
                    SPDLOG_INFO("Using local IPv4 address: {}", ip_str);
                    return ip_str;
                }
            }
        }
        close(sock);
    }

    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) != -1) {
        for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == nullptr) {
                continue;
            }

            // IPV4
            if (ifa->ifa_addr->sa_family == AF_INET) {
                auto* addr_in = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
                std::string ip_str(INET_ADDRSTRLEN, 0);
                inet_ntop(AF_INET, &(addr_in->sin_addr), ip_str.data(), INET_ADDRSTRLEN);

                if (strcmp(ip_str.data(), "127.0.0.1") != 0) {
                    freeifaddrs(ifaddr);
                    SPDLOG_INFO("Using local IPv4 address: {}", ip_str);
                    return ip_str;
                }
            }
        }
        freeifaddrs(ifaddr);
    }

    const char* host = std::getenv("NODE_IP");
    if ((host != nullptr) && strlen(host) > 0) {
        SPDLOG_INFO("Using NODE_IP from environment: {}", host);
        return host;
    }

    const char* hostname = std::getenv("ALIPAY_POD_NAME");
    if ((hostname != nullptr) && strlen(hostname) > 0) {
        SPDLOG_INFO("Using ALIPAY_POD_NAME from environment: {}", hostname);
        return hostname;
    }

    throw std::runtime_error("No valid IP found");
}

} // namespace astate