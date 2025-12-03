#pragma once

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>

#include <unistd.h>

#include <glog/logging.h>

#include "common/macro_utils.h"
#include "common/option.h"

namespace astate {

class AstateLogSink : public google::LogSink {
 public:
    DISALLOW_COPY_AND_MOVE(AstateLogSink);

    AstateLogSink(const std::string& filename_pattern, const std::string& special_log_path)
        : regex_pattern_(filename_pattern, std::regex::icase), // ignore case
          special_log_path_(special_log_path) {
        if (!std::filesystem::exists(special_log_path)) {
            std::filesystem::create_directories(std::filesystem::path(special_log_path_).parent_path());
        }
        ofs_.open(special_log_path_, std::ios::app); // open with append mode
    }

    ~AstateLogSink() override {
        if (ofs_.is_open()) {
            ofs_.close();
        }
    }

    // Write log to the log file configured by special_log_path_.
    // Note: the logs are also written to the default log file.
    void send(
        google::LogSeverity severity,
        const char* full_filename,
        const char* base_filename,
        int line,
        const google::LogMessageTime& log_time,
        const char* message,
        size_t message_len) override {
        // if base_filename matches our regex rule
        if (std::regex_search(full_filename, regex_pattern_)) {
            std::lock_guard<std::mutex> lock(write_mutex_);
            ofs_ << "[" << SeverityName(severity) << "] "
                 << "[" << FormatTime(log_time) << "] "
                 << "[" << base_filename << ":" << line << "] " << std::string(message, message_len) << "\n";
            ofs_.flush();
        }
        // otherwise, do not process (let glog handle the default path logic)
    }

 private:
    static std::string SeverityName(google::LogSeverity log_severity) {
        switch (log_severity) {
            case google::GLOG_INFO:
                return "INFO";
            case google::GLOG_WARNING:
                return "WARN";
            case google::GLOG_ERROR:
                return "ERROR";
            case google::GLOG_FATAL:
                return "FATAL";
            default:
                break;
        }
        return "UNKNOWN";
    }

    static std::string FormatTime(const google::LogMessageTime& time) {
        std::ostringstream oss;
        oss << std::setw(2) << std::setfill('0') << time.month() << "-" << std::setw(2) << std::setfill('0')
            << time.day() << " " << std::setw(2) << std::setfill('0') << time.hour() << ":" << std::setw(2)
            << std::setfill('0') << time.min() << ":" << std::setw(2) << std::setfill('0') << time.sec() << "."
            << std::setw(6) << std::setfill('0') << time.usec();
        return oss.str();
    }

    std::regex regex_pattern_;
    std::string special_log_path_;
    std::ofstream ofs_;
    std::mutex write_mutex_;
};

static std::atomic<bool> log_initialized = false;
static std::atomic<bool> flush_thread_running = false;
static std::unique_ptr<std::thread> flush_thread;

class AsyncFlushThreadManager {
 public:
    ~AsyncFlushThreadManager() {
        if (flush_thread_running.load()) {
            flush_thread_running.store(false);

            if (flush_thread && flush_thread->joinable()) {
                flush_thread->join();
            }

            flush_thread.reset();
        }
    }
};

static AsyncFlushThreadManager flush_thread_manager;

#define DEFAULT_FLUSH_INTERVAL_SECONDS 10

#define DEFAUTL_LOG_CLEANER_DAYS 3

inline void StartAsyncFlushThread(int flush_interval_seconds);
inline void StopAsyncFlushThread();
inline bool IsAsyncFlushThreadRunning();

inline void InitGlog(int argc, char* argv[], const Options& options = Options()) {
    if (log_initialized.exchange(true)) {
        return;
    }

    google::ParseCommandLineFlags(&argc, &argv, true);

    std::string log_level;
    if (!google::GetCommandLineOption("minloglevel", &log_level)) {
        FLAGS_minloglevel = google::INFO;
    }

    std::string log_dir;
    if (!google::GetCommandLineOption("log_dir", &log_dir) || log_dir.empty()) {
        log_dir = GetOptionValue<std::string>(options, ASTATE_LOG_DIR);
    }
    std::filesystem::create_directories(log_dir);
    FLAGS_log_dir = log_dir;

    std::string logtostderr;
    if (google::GetCommandLineOption("logtostderr", &logtostderr)) {
        if (logtostderr == "true" || logtostderr == "1" || logtostderr == "yes") {
            FLAGS_logtostderr = false;
            FLAGS_stderrthreshold = google::INFO;
        }
    }

    std::string log_flush_interval;
    if (!google::GetCommandLineOption("logbufsecs", &log_flush_interval)) {
        FLAGS_logbufsecs = 1;
    }

    std::chrono::minutes overdue_days = std::chrono::minutes(DEFAUTL_LOG_CLEANER_DAYS * 24 * 60);
    google::EnableLogCleaner(overdue_days.count());
    google::FlushLogFiles(google::INFO);

    // Init glog.
    std::string app_name;
    if (!google::GetCommandLineOption("app_name", &app_name) || app_name.empty()) {
        app_name = "astate__storage";
    }
    google::InitGoogleLogging(app_name.c_str());

    if (GetOptionValue<bool>(options, ASTATE_ENABLE_GLOG_EXT)) {
        std::string log_path
            = GetOptionValue<std::string>(options, ASTATE_LOG_DIR) + "/astate." + std::to_string(getpid()) + ".log";
        google::AddLogSink(new AstateLogSink(R"(master.*)", log_path));
        LOG(INFO) << "Added glog ext sink for master.*";
    }

    StartAsyncFlushThread(DEFAULT_FLUSH_INTERVAL_SECONDS);
}

inline void StartAsyncFlushThread(int flush_interval_seconds = DEFAULT_FLUSH_INTERVAL_SECONDS) {
    if (flush_thread_running.exchange(true)) {
        return;
    }

    flush_thread = std::make_unique<std::thread>([flush_interval_seconds]() {
        LOG(INFO) << "Async glog flush thread started with interval: " << flush_interval_seconds << " seconds";

        auto last_flush_time = std::chrono::steady_clock::now();

        while (flush_thread_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_flush_time);

            if (elapsed.count() >= flush_interval_seconds && flush_thread_running.load()) {
                google::FlushLogFiles(google::INFO);
                last_flush_time = current_time;
            }
        }

        LOG(INFO) << "Async glog flush thread stopped";
    });

    LOG(INFO) << "Async glog flush thread started with interval: " << flush_interval_seconds << " seconds";
}

inline void StopAsyncFlushThread() {
    if (!flush_thread_running.exchange(false)) {
        return;
    }

    if (flush_thread && flush_thread->joinable()) {
        auto timeout = std::chrono::seconds(1);
        auto start_time = std::chrono::steady_clock::now();

        while (flush_thread->joinable() && std::chrono::steady_clock::now() - start_time < timeout) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (flush_thread->joinable()) {
            flush_thread->join();
        }
    }

    flush_thread.reset();
    LOG(INFO) << "Async glog flush thread stopped";
}

inline bool IsAsyncFlushThreadRunning() {
    return flush_thread_running.load();
}

} // namespace astate
