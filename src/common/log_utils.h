#pragma once

#include <utility>

#include <gflags/gflags.h>
#include <spdlog/spdlog.h>

#include "common/glog_utils.h"
#include "common/option.h"
#include "common/spdlog_utils.h"

namespace astate {

#define ASTATE_APP_NAME "astate_storage";

static void InitAstateLog(int argc, char* argv[], const Options& options = Options()) {
    auto log_backend = GetOptionValue<std::string>(options, ASTATE_LOG_BACKEND);
    if (log_backend == "GLOG") {
        InitGlog(argc, argv, options);
    } else if (log_backend == "SPDLOG") {
        std::string app_name;
        if (!google::GetCommandLineOption("app_name", &app_name) || app_name.empty()) {
            app_name = ASTATE_APP_NAME;
        }
        InitSpdlog(app_name, options);
    } else {
        SPDLOG_ERROR("Invalid log backend: {}", log_backend);
    }
}

static void InitAstateLog(std::string app_name, const Options& options) {
    auto log_backend = GetOptionValue<std::string>(options, ASTATE_LOG_BACKEND);
    if (log_backend == "SPDLOG") {
        InitSpdlog(std::move(app_name), options);
    } else if (log_backend == "GLOG") {
        std::string log_name = ASTATE_APP_NAME;
        char* argv[] = {log_name.data()};
        InitGlog(1, argv, options);
    } else {
        SPDLOG_ERROR("Invalid log backend: {}", log_backend);
    }
}

} // namespace astate
