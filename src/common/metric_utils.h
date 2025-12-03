#pragma once

#include <utility>

#include <spdlog/spdlog.h>

#include "common/option.h"

namespace astate {

/*
 * PerfMetricsController is a helper class to control performance metrics logging based on sampling parameters.
 */
class PerfMetricsController {
 public:
    PerfMetricsController() = delete;

    PerfMetricsController(std::string component_name, const Options& options)
        : component_name_(std::move(component_name)),
          enable_perf_metrics_(GetOptionValue<bool>(options, TRANSFER_ENGINE_ENABLE_PERF_METRICS)),
          perf_sample_rate_(GetOptionValue<int>(options, TRANSFER_ENGINE_SAMPLE_RATE)),
          perf_sample_step_(GetOptionValue<int>(options, TRANSFER_ENGINE_SAMPLE_STEP)) {
        SPDLOG_INFO(
            "PerfMetricsController[{}] initialized: enable_perf_metrics={}, "
            "sample_rate={}, sample_step={}",
            component_name_,
            enable_perf_metrics_,
            perf_sample_rate_,
            perf_sample_step_);
    }

    [[nodiscard]] bool IsPerfMetricsEnabled() const { return enable_perf_metrics_; }

    bool ShouldLogPerfMetric(int64_t seq_id) {
        if (!enable_perf_metrics_) {
            return false;
        }
        if ((seq_id == perf_sample_step_) || (++perf_read_count_ % perf_sample_rate_ == 0)) {
            perf_read_count_ = 0; // reset counter after logging
            return true;
        }
        return false;
    }

 private:
    std::string component_name_;

    bool enable_perf_metrics_;

    // Perf metrics sampling: two parameters to control sampling behavior currenlty.
    int perf_sample_rate_; // 1/sample_rate_ requests for perf metrics
    int perf_sample_step_; // sample Nth step
    std::atomic_int perf_read_count_{0};
};

} // namespace astate
