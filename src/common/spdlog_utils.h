#pragma once

#include <unistd.h>

#include <spdlog/async.h>
#include <spdlog/common.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include "common/option.h"

namespace astate {

static std::mutex logger_mutex; // Global logger mutex
static std::shared_ptr<spdlog::logger> logger = nullptr; // Global logger

// Get log level from options
inline spdlog::level::level_enum GetLogLevel(const Options& options) {
    spdlog::level::level_enum ret = spdlog::level::info; // default log level

    auto log_level = GetOptionValue<std::string>(options, ASTATE_LOG_LEVEL);
    if (log_level == "DEBUG") {
        ret = spdlog::level::debug;
    } else if (log_level == "INFO") {
        ret = spdlog::level::info;
    } else if (log_level == "WARNING") {
        ret = spdlog::level::warn;
    } else if (log_level == "ERROR") {
        ret = spdlog::level::err;
    } else {
        SPDLOG_ERROR("Unknown log level: {}", log_level);
    }
    return ret;
}

/*
 * Initialize spdlog
 * @param app_name: the name of the application
 * @param options: the options of the application
 */
inline void InitSpdlog(std::string app_name, const Options& options) {
    // Init async thread pool
    spdlog::init_thread_pool(8192, 1);

    // Create async multi-sink logger
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (logger == nullptr) {
        logger = std::make_shared<spdlog::async_logger>(
            app_name, // logger name
            spdlog::sinks_init_list{}, // init with empty sinks list
            spdlog::thread_pool(), // async thread pool
            spdlog::async_overflow_policy::block // When the queue is full, block
        );
    }

    // Add sinks to the logger by configuration
    std::vector<spdlog::sink_ptr> sinks;
    if (GetOptionValue<bool>(options, ASTATE_LOG_TO_CONSOLE)) {
        // Add stdout sink
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_mt>());
    }

    if (GetOptionValue<bool>(options, ASTATE_LOG_TO_FILE)) {
        // Add daily file sink (cut a new file at 00:00 every day, keep the last specific days)
        auto log_dir = GetOptionValue<std::string>(options, ASTATE_LOG_DIR);
        std::string log_name = log_dir + "/" + app_name + "." + std::to_string(getpid()); // logdir/app_name.pid
        int max_file_days = GetOptionValue<int>(options, ASTATE_LOG_MAX_FILE_DAYS);
        auto daily_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_name, 0, 0, false, max_file_days);
        sinks.push_back(daily_sink);
    }
    logger->sinks().insert(logger->sinks().end(), sinks.begin(), sinks.end());

    // Set global default logger
    spdlog::set_default_logger(logger);
    spdlog::set_level(GetLogLevel(options)); // Global minimum log level
    spdlog::flush_on(spdlog::level::info); // info and above immediately flush
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [pid %P] [thread %t] [%l] [%s:%#] %v");

    SPDLOG_INFO("Initialized spdlog for {}", app_name);
}

} // namespace astate
