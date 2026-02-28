#ifndef ESKF_BASELINE_STATS_HPP_
#define ESKF_BASELINE_STATS_HPP_

#include <chrono>
#include <cstdint>
#include <deque>

#include "eskf_baseline/definitions.hpp"

namespace eskf {

struct Totals {
  uint64_t predict_ok = 0;
  uint64_t predict_fail = 0;

  uint64_t correct_ok = 0;
  uint64_t correct_reject = 0;
  uint64_t correct_fail = 0;  // non-success, non-reject (future-proof)

  uint64_t process_wakeups = 0;
  uint64_t process_iters = 0;
};

struct TimeTotals {
  int64_t process_ns = 0;
  int64_t predict_ns = 0;
  int64_t correct_ns = 0;
  int64_t rebuild_ns = 0;
};

struct StatsSample {
  double wall_t = 0.0;  // steady-clock seconds
  Totals totals;
  TimeTotals time_totals;
};

struct WindowStats {
  double window_s = 0.0;
  double sample_age_s = 0.0;

  // rates
  double process_hz = 0.0;
  double predict_ok_hz = 0.0;
  double predict_fail_hz = 0.0;
  double correct_ok_hz = 0.0;
  double correct_reject_hz = 0.0;

  // cpu fractions of wall time in window
  double process_cpu = 0.0;
  double predict_cpu = 0.0;
  double correct_cpu = 0.0;
  double rebuild_cpu = 0.0;

  // mean cost
  double mean_predict_us = 0.0;
  double mean_correct_us = 0.0;
  double mean_process_us = 0.0;
};

// namespace

class StatsContainer {
 public:
  WindowStats computeWindowStats() const;

  template <auto Mem, typename... Args>
  void updateNs(const std::chrono::duration<Args...>& d) {
    time_totals_.*Mem +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
  }

  template <typename... R>
  void updatePredictNs(const std::chrono::duration<R...>& d) {
    updateNs<&TimeTotals::predict_ns>(d);
  }

  template <typename... R>
  void updateCorrectNs(const std::chrono::duration<R...>& d) {
    updateNs<&TimeTotals::correct_ns>(d);
  }

  template <typename... R>
  void updateProcessNs(const std::chrono::duration<R...>& d) {
    updateNs<&TimeTotals::process_ns>(d);
  }

  template <typename... R>
  void updateRebuildNs(const std::chrono::duration<R...>& d) {
    updateNs<&TimeTotals::rebuild_ns>(d);
  }

  void updateCorrectOutcome(Errc ec);

  void updatePredictOutcome(Errc ec);

  void recordStatsSample();

  void updateProcessWakeupCount() { ++totals_.process_wakeups; }

  void updateProcessIterCount() { ++totals_.process_iters; }

 private:
  double steadyNowSec() const;

  double stats_window_s_ = 5.0;
  double stats_margin_s_ = 0.5;

  mutable Totals totals_;
  mutable TimeTotals time_totals_;
  std::deque<StatsSample> stats_samples_;
};

}  // namespace eskf

#endif  // ESKF_BASELINE_STATS_HPP_
