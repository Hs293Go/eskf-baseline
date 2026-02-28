#include "eskf_baseline/stats.hpp"

#include <algorithm>

namespace eskf {

using std::ranges::lower_bound;

WindowStats StatsContainer::computeWindowStats() const {
  WindowStats out;

  if (stats_samples_.size() < 2) {
    return out;
  }

  const double now = steadyNowSec();
  const double target = now - stats_window_s_;

  const auto& newest = stats_samples_.back();

  // find first sample with wall_t >= target
  auto it = lower_bound(stats_samples_, target, {}, &StatsSample::wall_t);

  // choose oldest sample <= target if possible
  const StatsSample* oldest = nullptr;
  if (it == stats_samples_.begin()) {
    oldest = &*it;  // best we can do
  } else if (it == stats_samples_.end()) {
    oldest = &stats_samples_.front();  // should be rare with trimming
  } else {
    oldest = &*std::prev(it);
  }

  const double dt = newest.wall_t - oldest->wall_t;
  if (dt <= 1e-9) {
    return out;
  }

  out.window_s = dt;
  out.sample_age_s = now - newest.wall_t;

  const auto& o = oldest->totals;
  const auto& n = newest.totals;

  const auto& tot_o = oldest->time_totals;
  const auto& tot_n = newest.time_totals;

  const double dpred_ok = static_cast<double>(n.predict_ok - o.predict_ok);
  const double dpred_fail =
      static_cast<double>(n.predict_fail - o.predict_fail);
  const double dcorr_ok = static_cast<double>(n.correct_ok - o.correct_ok);
  const double dcorr_reject =
      static_cast<double>(n.correct_reject - o.correct_reject);
  const double dproc = static_cast<double>(n.process_iters - o.process_iters);

  const auto dpred_ns =
      static_cast<double>(tot_n.predict_ns - tot_o.predict_ns);
  const auto dcorr_ns =
      static_cast<double>(tot_n.correct_ns - tot_o.correct_ns);
  const auto dproc_ns =
      static_cast<double>(tot_n.process_ns - tot_o.process_ns);
  const auto drbld_ns =
      static_cast<double>(tot_n.rebuild_ns - tot_o.rebuild_ns);

  out.predict_ok_hz = dpred_ok / dt;
  out.predict_fail_hz = dpred_fail / dt;
  out.correct_ok_hz = dcorr_ok / dt;
  out.correct_reject_hz = dcorr_reject / dt;
  out.process_hz = dproc / dt;

  const double denom_ns = dt * 1e9;
  out.predict_cpu = dpred_ns / denom_ns;
  out.correct_cpu = dcorr_ns / denom_ns;
  out.process_cpu = dproc_ns / denom_ns;
  out.rebuild_cpu = drbld_ns / denom_ns;

  const double dpred_calls = dpred_ok + dpred_fail;
  const double dcorr_calls = dcorr_ok + dcorr_reject;  // + fail if you want
  if (dpred_calls > 0) {
    out.mean_predict_us = (dpred_ns / dpred_calls) / 1e3;
  }
  if (dcorr_calls > 0) {
    out.mean_correct_us = (dcorr_ns / dcorr_calls) / 1e3;
  }
  if (dproc > 0) {
    out.mean_process_us = (dproc_ns / dproc) / 1e3;
  }

  return out;
}

void StatsContainer::recordStatsSample() {
  const auto now = steadyNowSec();
  stats_samples_.emplace_back(now, totals_, time_totals_);

  const double keep_from = now - (stats_window_s_ + stats_margin_s_);
  while (!stats_samples_.empty() && stats_samples_.front().wall_t < keep_from) {
    stats_samples_.pop_front();
  }
}

double StatsContainer::steadyNowSec() const {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void StatsContainer::updateCorrectOutcome(Errc ec) {
  if (IsSuccess(ec)) {
    ++totals_.correct_ok;
  } else if (IsReject(ec)) {
    ++totals_.correct_reject;
  } else {
    ++totals_.correct_fail;
  }
}

void StatsContainer::updatePredictOutcome(Errc ec) {
  if (IsSuccess(ec)) {
    ++totals_.predict_ok;
  } else {
    ++totals_.predict_fail;
  }
}
}  // namespace eskf
