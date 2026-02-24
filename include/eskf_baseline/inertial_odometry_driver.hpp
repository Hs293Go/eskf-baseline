#ifndef ESKF_BASELINE_INERTIAL_ODOMETRY_DRIVER_HPP_
#define ESKF_BASELINE_INERTIAL_ODOMETRY_DRIVER_HPP_

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace eskf {

template <typename T>
concept TimeStamped = requires(T t) {
  { t.t } -> std::convertible_to<double>;
};

using std::ranges::lower_bound;
using std::ranges::upper_bound;

template <typename It>
It prevOrBegin(It it, It begin) {
  // Use a cleverer branchless version in the helper
  return std::prev(it, static_cast<int>(it != begin));
}

template <TimeStamped T>
struct Checkpoint {
  std::deque<T> ckpts_;
  double period = 0.02;
  double max_age = 10.0;

  void setSingle(const T& c) { ckpts_.assign(1, c); }

  void tryPush(const T& c) {
    if (ckpts_.empty() || (c.t - ckpts_.back().t) >= period) {
      ckpts_.push_back(c);
    }
  }

  T get(double t) {
    auto it = upper_bound(ckpts_, t, {}, &T::t);  // first > t
    return *prevOrBegin(it, ckpts_.begin());
  }

  std::optional<T> tryGet(double t) {
    auto it = upper_bound(ckpts_, t, {}, &T::t);  // first > t
    if (it == ckpts_.begin()) {
      return std::nullopt;
    }
    return *prev(it);
  }

  void eraseUntil(double keep_from) {
    if (ckpts_.empty()) {
      return;
    }
    auto keep_it = lower_bound(ckpts_, keep_from, {}, &T::t);
    // Keep one checkpoint before keep_from if it exists
    keep_it = prevOrBegin(keep_it, ckpts_.begin());

    ckpts_.erase(ckpts_.begin(), keep_it);
  }

  void eraseAfter(double t) {
    if (ckpts_.empty()) {
      return;
    }
    // upper_bound returns first crumb with t > meas.t.
    const auto it = upper_bound(ckpts_, t, {}, &T::t);
    // Since meas.t >= crumbs.front().t, upper_bound cannot return
    // crumbs.begin()
    ckpts_.erase(it, ckpts_.end());
  }

  bool empty() const { return ckpts_.empty(); }
};

template <typename T>
concept HasEstimationQuantities = requires {
  requires TimeStamped<typename T::Context>;
  requires TimeStamped<typename T::Measurement>;
  requires TimeStamped<typename T::Input>;
};

template <typename T>
concept KalmanFilterAlgorithm =
    HasEstimationQuantities<T> &&
    requires(T self, T::Context& ctx, T::Measurement meas, T::Input u) {
      { self.timeUpdate(ctx, u, 0.01) } -> std::convertible_to<bool>;
      { self.measurementUpdate(ctx, meas) } -> std::same_as<void>;
    };

template <KalmanFilterAlgorithm Algorithm>
class InertialOdometryDriver {
  using Context = typename Algorithm::Context;
  using Measurement = typename Algorithm::Measurement;
  using Input = typename Algorithm::Input;

  Algorithm alg_;

  // Time-ordered measurement history
  std::deque<Measurement> meas_hist_;

  // measurement frontier: all prior measurements are fused into the posterior
  double processed_up_to_t_ = -std::numeric_limits<double>::infinity();

  // next measurement in meas_hist_ to be processed in normal streaming mode
  size_t meas_next_idx_ = 0;

  std::deque<Input> imus_;

  Context prio;
  Context post;

  // time-ordered checkpoint history covering (head_t - max_age) until head_t
  Checkpoint<Context> ckpts_;
  double max_ckpt_age_ = 10.0;

  std::shared_mutex mtx;
  std::condition_variable_any cv;
  std::jthread thread;

  struct RebuildPlan {
    // Snapshot inputs for rebuild (stable while replay runs)
    std::vector<Input> imus;
    // Where we are in the IMU stream (time-ordered)
    // index of first imu with t > ctx.t  (next event)
    size_t imu_it_next = 0;

    // held input for current interval
    // Safety: This should point into plan.imus only
    Input const* u_zoh = nullptr;

    std::vector<Measurement> meas;

    // Next measurement to replay
    size_t meas_idx = 0;

    double last_meas_applied_t = -std::numeric_limits<double>::infinity();

    // Rebuilt checkpoints
    Checkpoint<Context> new_ckpts = {.period = 0.02};

    Context ctx;  // current in-progress context during replay

    double target_head_t = 0.0;
  };
  std::optional<RebuildPlan> rebuild_;
  std::optional<double> late_meas_trigger_t_;

 public:
  void start() {
    prio = post;
    ckpts_.setSingle(post);
    thread = std::jthread([this](std::stop_token stop) { process(stop); });
  }

  void push_imu(Input imu) {
    std::scoped_lock lock(mtx);

    imus_.insert(upper_bound(imus_, imu.t, {}, &Input::t), imu);
    cv.notify_all();
  }

  void push_pose(Measurement meas) {
    std::scoped_lock lock(mtx);
    auto insert_pt = upper_bound(meas_hist_, meas.t, {}, &Measurement::t);
    auto insert_idx = std::distance(meas_hist_.begin(), insert_pt);
    meas_hist_.insert(insert_pt, meas);

    if (std::cmp_less_equal(insert_idx, meas_next_idx_)) {
      ++meas_next_idx_;
    }
    if (meas.t < processed_up_to_t_) {
      // We don't start rebuild here (keep push fast); process() will start it.
      // But we do wake it up.
      if (!late_meas_trigger_t_ || meas.t < late_meas_trigger_t_.value()) {
        late_meas_trigger_t_ = meas.t;
      }
    }

    cv.notify_all();
  }

  void pruneHistory(double head_t) {
    const double keep_from = head_t - max_ckpt_age_;

    // Prune IMUs older than keep_from
    if (!imus_.empty()) {
      imus_.erase(imus_.begin(), lower_bound(imus_, keep_from, {}, &Input::t));
    }

    // Prune measurements older than keep_from
    if (!meas_hist_.empty()) {
      const auto keep_it =
          lower_bound(meas_hist_, keep_from, {}, &Measurement::t);
      const auto erased = std::distance(meas_hist_.begin(), keep_it);
      meas_hist_.erase(meas_hist_.begin(), keep_it);
      meas_next_idx_ = meas_next_idx_ > erased ? (meas_next_idx_ - erased) : 0;
    }

    // Prune checkpoints older than keep_from, but keep at least one
    ckpts_.eraseUntil(keep_from);
    if (ckpts_.empty()) {
      ckpts_.setSingle(post);
    }

    if (late_meas_trigger_t_ && late_meas_trigger_t_.value() < keep_from) {
      late_meas_trigger_t_.reset();
    }
  }

  void startRebuild(double trigger_t) {
    // PRE: mtx is held, imus_ non-empty, ckpts_ non-empty

    RebuildPlan plan;
    plan.last_meas_applied_t = processed_up_to_t_;

    const auto head_t = imus_.back().t;
    plan.target_head_t = head_t;

    // Choose seed checkpoint <= trigger_t (or earliest available)
    const auto seed = ckpts_.get(trigger_t);
    plan.ctx = seed;

    // Snapshot IMUs needed: include one sample before seed.t (for ZOH hold),
    // and include up to head_t.

    auto lb = lower_bound(imus_, seed.t, {}, &Input::t);
    lb = prevOrBegin(lb, imus_.begin());

    auto ub = upper_bound(imus_, head_t, {}, &Input::t);
    plan.imus.assign(lb, ub);

    // Snapshot measurements needed: all retained measurements in [seed.t,
    // head_t]

    auto m0 = lower_bound(meas_hist_, seed.t, {}, &Measurement::t);
    auto m1 = upper_bound(meas_hist_, head_t, {}, &Measurement::t);
    plan.meas.assign(m0, m1);

    // Initialize rebuilt checkpoints with the seed
    plan.new_ckpts.setSingle(seed);

    // Initialize IMU ZOH state at plan.ctx.t using upper_bound on plan.imus
    // (same semantics as your propagateImpl)

    auto it_next = upper_bound(plan.imus, plan.ctx.t, {}, &Input::t);
    plan.imu_it_next =
        static_cast<size_t>(std::distance(plan.imus.begin(), it_next));
    plan.u_zoh = &*prevOrBegin(it_next, plan.imus.begin());

    plan.meas_idx = 0;

    rebuild_ = std::move(plan);
  }

  Context peek(double time) {
    Context ctx;
    std::vector<Input> imus_copy;  // Vector for better cache locality; Maybe
                                   // even inplace_vector
    {
      std::shared_lock lock(mtx);  // Reader lock here; quickly unlocked since
      ctx = post;
      if (imus_.empty()) {
        return ctx;
      }

      if (time <= ctx.t) {
        return ctx;
      }

      const double head_t = imus_.back().t;
      const double target_t = std::min(time, head_t);

      auto it = lower_bound(imus_, ctx.t, {}, &Input::t);
      // include one sample before ctx.t for ZOH hold
      it = prevOrBegin(it, imus_.begin());
      auto sent =
          upper_bound(imus_, target_t, {}, &Input::t);  // first > target_t
      if (it == sent) {
        // No IMU samples in (ctx.t, target_t], return current state
        return ctx;
      }
      imus_copy.assign(it, sent);
      time = target_t;
    }
    peekTo(imus_copy, ctx, time);
    return ctx;
  }

  template <std::ranges::bidirectional_range R, typename Callback>
    requires std::same_as<std::ranges::range_value_t<R>, Input>
  void propagateImpl(const R& imus, Context& ctx, double target_t,
                     Callback&& callback) const {
    const auto imu_begin = std::ranges::begin(imus);
    const auto imu_end = std::ranges::end(imus);
    if (target_t <= ctx.t || imus.empty()) {
      return;
    }
    // Locate the first IMU sample with t > ctx.t.
    auto it = upper_bound(imus, ctx.t, {}, &Input::t);

    const auto* u_zoh = &*prevOrBegin(it, imu_begin);

    // Propagate in chunks between IMU samples and the target time
    while (ctx.t < target_t) {
      // The next event time is either the next IMU sample or the target time,
      // whicher is earlier
      const auto next_event_t =
          it != imu_end ? std::min(it->t, target_t) : target_t;
      const auto dt = next_event_t - ctx.t;

      // Run the KF predict equations if time has advanced
      if (alg_.timeUpdate(ctx, *u_zoh, dt)) {
        callback(std::as_const(ctx));
      }

      if (it == imu_end || ctx.t == target_t) {
        break;
      }
      u_zoh = &*it;
      ++it;
    }
  }

  void propagateTo(const std::deque<Input>& imus, Context& ctx,
                   double target_t) {
    propagateImpl(imus, ctx, target_t,
                  [this](const auto& c) { ckpts_.tryPush(c); });
  }

  void peekTo(std::span<const Input> imus, Context& ctx,
              double target_t) const {
    propagateImpl(imus, ctx, target_t, [](auto&& /*swallow*/) {});
  }

  void peekTo(const std::deque<Input>& imus, Context& ctx,
              double target_t) const {
    propagateImpl(imus, ctx, target_t, [](auto&& /*swallow*/) {});
  }

  void stepRebuild(int max_events) {
    // PRE: mtx held, rebuild_ engaged
    auto& plan = *rebuild_;

    for (int k = 0; k < max_events; ++k) {
      if (plan.ctx.t >= plan.target_head_t) break;

      const bool have_next_imu = plan.imu_it_next < plan.imus.size();
      const double next_imu_t =
          have_next_imu ? plan.imus[plan.imu_it_next].t : plan.target_head_t;

      const bool have_next_meas = plan.meas_idx < plan.meas.size();
      const double next_meas_t =
          have_next_meas ? plan.meas[plan.meas_idx].t : plan.target_head_t;

      const double next_t =
          std::min({next_imu_t, next_meas_t, plan.target_head_t});
      const double dt = next_t - plan.ctx.t;

      if (alg_.timeUpdate(plan.ctx, *plan.u_zoh, dt)) {
        plan.new_ckpts.tryPush(plan.ctx);
      }

      // If we hit a measurement time, apply all measurements at this time
      while (plan.meas_idx < plan.meas.size() &&
             plan.meas[plan.meas_idx].t <= plan.ctx.t) {
        assert(plan.meas[plan.meas_idx].t <= plan.ctx.t);

        alg_.measurementUpdate(plan.ctx, plan.meas[plan.meas_idx]);
        plan.last_meas_applied_t =
            std::max(plan.last_meas_applied_t, plan.meas[plan.meas_idx].t);
        ++plan.meas_idx;
        plan.new_ckpts.tryPush(plan.ctx);
      }

      // If we hit an IMU event time, advance ZOH hold
      if (have_next_imu && plan.imus[plan.imu_it_next].t <= plan.ctx.t) {
        plan.u_zoh = &plan.imus[plan.imu_it_next];
        ++plan.imu_it_next;
      }
    }

    // Commit if done
    if (plan.ctx.t >= plan.target_head_t) {
      post = plan.ctx;
      ckpts_ = std::move(plan.new_ckpts);

      // Update processed frontier: we have incorporated all measurements <=
      // head_t
      processed_up_to_t_ =
          std::max(processed_up_to_t_, plan.last_meas_applied_t);

      // Set meas_next_idx_ to first measurement with t > processed_up_to_t_
      auto it =
          upper_bound(meas_hist_, processed_up_to_t_, {}, &Measurement::t);
      meas_next_idx_ =
          static_cast<size_t>(std::distance(meas_hist_.begin(), it));

      pruneHistory(post.t);
      rebuild_.reset();
    }
  }

  enum class ProcessResult { kContinue, kExit };

  ProcessResult processOnce() {
    if (imus_.empty()) {
      return ProcessResult::kContinue;
    }
    const auto head_t = imus_.back().t;
    if (rebuild_) {
      stepRebuild(/*max_events=*/200);
      return ProcessResult::kContinue;
    }

    // Find the earliest measurement that is < processed_up_to_t_ (late
    // arrival) Minimal trigger: if any measurement exists with t <
    // processed_up_to_t_
    if (!rebuild_ && late_meas_trigger_t_.has_value()) {
      // Trigger at the oldest measurement in-window; you can also trigger
      // at the inserted time.
      startRebuild(late_meas_trigger_t_.value());
      late_meas_trigger_t_.reset();
      return ProcessResult::kContinue;
    }

    while (meas_next_idx_ < meas_hist_.size()) {
      const auto& meas = meas_hist_[meas_next_idx_];

      if (meas.t > head_t) {
        break;
      }

      if (meas.t < imus_.front().t) {
        ++meas_next_idx_;
        return ProcessResult::kContinue;
      }

      Context seed = ckpts_.get(meas.t);
      ckpts_.eraseAfter(meas.t);

      prio = seed;
      propagateTo(imus_, prio, meas.t);
      alg_.measurementUpdate(prio, meas);
      ckpts_.tryPush(prio);  // checkpoint after measurement update
      propagateTo(imus_, prio, head_t);

      post = prio;
      processed_up_to_t_ = std::max(processed_up_to_t_, meas.t);
      ++meas_next_idx_;

      pruneHistory(post.t);  // Retain bounded history
    }

    if (head_t > post.t) {
      prio = post;
      propagateTo(imus_, prio, imus_.back().t);
      post = prio;
      pruneHistory(post.t);
    }
    return ProcessResult::kContinue;
  }

  void process(std::stop_token stop) {
    auto& self = *this;
    while (!stop.stop_requested()) {
      std::unique_lock lock(mtx);
      cv.wait(lock, stop, [this, &stop] {
        return                                     // Wake conditions:
            stop.stop_requested()                  // 1. Exit if stopping
            || late_meas_trigger_t_.has_value()    // 2. Start rebuilding
            || rebuild_.has_value()                // 3. Rebuilding
            || meas_next_idx_ < meas_hist_.size()  // 4. Process measurements
            || (!imus_.empty() && imus_.back().t > post.t);  // 5. Process IMU
      });
      switch (processOnce()) {
        case ProcessResult::kContinue:
          break;  // Break from switch and continue loop
        case ProcessResult::kExit:
          return;
      }
    }
  }
};

}  // namespace eskf

#endif  // ESKF_BASELINE_INERTIAL_ODOMETRY_DRIVER_HPP_
