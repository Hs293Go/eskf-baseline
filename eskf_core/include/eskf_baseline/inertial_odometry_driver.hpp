#ifndef ESKF_BASELINE_INERTIAL_ODOMETRY_DRIVER_HPP_
#define ESKF_BASELINE_INERTIAL_ODOMETRY_DRIVER_HPP_

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
// Include just the headers we need to compile; run IWYU right before we ship

namespace eskf {
enum class Errc {
  kSuccess,
  kNoop,
  // Future proof: Distinguish between fatal and non fatal outliers
  kTimeStepTooLarge,  // anticipate
  kOutlierDetected,

  kFatalNonPositiveTimeStep,
  kFatalOutlierDetected,
  kFatalLinalgFailure,
  kFatalNonFiniteState,
  kFailure,
  kUnknown,
};

constexpr bool IsSuccess(Errc e) { return e == Errc::kSuccess; }

constexpr bool IsReject(Errc e) {
  switch (e) {
    case Errc::kTimeStepTooLarge:
      [[fallthrough]];
    case Errc::kOutlierDetected:
      return true;
    default:
      return false;
  }
}

constexpr bool IsFatal(Errc e) {
  switch (e) {
    case Errc::kFatalOutlierDetected:
      [[fallthrough]];
    case Errc::kFatalLinalgFailure:
      [[fallthrough]];
    case Errc::kFatalNonPositiveTimeStep:
      [[fallthrough]];
    case Errc::kFatalNonFiniteState:
      return true;
    default:
      return false;
  }
}

struct BasicErrorContext {
  Errc ec = Errc::kUnknown;         // Success must be explicitly set
  std::string_view custom_message;  // Static strings only

  std::string_view message() const noexcept { return custom_message; }
  Errc errorCode() const noexcept { return ec; }
};

template <typename R>
concept ErrorContext = requires(const R& r) {
  { r.message() } noexcept -> std::convertible_to<std::string_view>;
  { r.errorCode() } noexcept -> std::same_as<Errc>;
};

static_assert(ErrorContext<BasicErrorContext>,
              "SimpleErrorContext should satisfy ErrorContext");

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

struct Statistics {
  // ingress
  std::uint64_t imu_in = 0;
  std::uint64_t meas_in = 0;

  // process loop
  std::uint64_t process_calls = 0;
  std::uint64_t process_ns_total = 0;
  std::uint64_t process_ns_max = 0;

  // predict
  std::uint64_t predict_calls = 0;
  std::uint64_t predict_fail = 0;
  Errc last_predict_ec = Errc::kSuccess;

  // correct
  std::uint64_t correct_calls = 0;
  std::uint64_t correct_success = 0;
  std::uint64_t correct_reject = 0;
  std::uint64_t correct_fatal = 0;
  Errc last_correct_ec = Errc::kSuccess;
};

struct StalenessStatus {
  double imu_head_t;         // if no IMU, -inf
  double post_t;             // current published context time
  double processed_up_to_t;  // latest time for which meas are fully fused
  std::optional<double> late_meas_trigger_t;  // earliest late meas waiting
  bool rebuilding;                            // rebuild_.has_value()

  // Derived:
  double imu_lag;   // imu_head_t - post_t (how far post lags IMU horizon)
  double meas_lag;  // imu_head_t - processed_up_to_t (how far meas fusion lags
                    // horizon)
  double trigger_age;  // imu_head_t - late_meas_trigger_t (if any)
};

template <typename T>
concept HasEstimationQuantities = requires {
  typename T::Estimate;
  requires TimeStamped<typename T::Measurement>;
  requires TimeStamped<typename T::Input>;
};

template <typename T>
concept KalmanFilterAlgorithm =
    HasEstimationQuantities<T> &&
    requires(const T& self, T::Estimate& est, const T::Measurement& meas,
             const T::Input& u) {
      { self.predict(est, u, 0.01) } -> ErrorContext;
      { self.correct(est, meas) } -> ErrorContext;
    };

struct EstimationOutcome {
  EstimationOutcome() = default;

  explicit EstimationOutcome(double t, Errc status = Errc::kUnknown,
                             std::string_view message = {})
      : t(t), status(status), message(message) {}

  template <ErrorContext T>
  EstimationOutcome(double t, T ctx)
      : t(t), message(ctx.message()), status(ctx.errorCode()) {}

  bool success() const { return IsSuccess(status); }
  bool reject() const { return IsReject(status); }
  bool fatal() const { return IsFatal(status); }

  double t;
  Errc status = Errc::kUnknown;
  std::string_view message;
};

// POD option struct passable as NTTP since C++20; tested on g++-12
struct InertialOdometryOptions {
  // Flags if the predict/correct methods of the algorithm are atomic (i.e. they
  // update the state in-place on success, and do not modify the state on
  // failure).
  bool predict_is_atomic = false;
  bool correct_is_atomic = false;
};

template <KalmanFilterAlgorithm Algorithm, InertialOdometryOptions Opts = {}>
class InertialOdometryDriver {
 public:
  using Estimate = typename Algorithm::Estimate;
  using Measurement = typename Algorithm::Measurement;
  using Input = typename Algorithm::Input;
  using PredictEC = std::remove_cvref_t<
      std::invoke_result_t<decltype(&Algorithm::predict), const Algorithm&,
                           Estimate&, const Input&, double>>;
  using CorrectEC = std::remove_cvref_t<
      std::invoke_result_t<decltype(&Algorithm::correct), const Algorithm&,
                           Estimate&, const Measurement&>>;

  static constexpr bool kPredictIsAtomic = Opts.predict_is_atomic;
  static constexpr bool kCorrectIsAtomic = Opts.correct_is_atomic;

  struct Context {
    double t;
    typename Algorithm::Estimate est;
  };

  InertialOdometryDriver() = default;

  explicit InertialOdometryDriver(Algorithm alg) : alg_(std::move(alg)) {}

  ~InertialOdometryDriver() { stop(); }

  bool running() const {
    std::shared_lock lock(mtx_);
    return thread_.joinable();
  }

  void start() {
    std::scoped_lock lock(mtx_);
    if (halted_) {
      return;  // halted is terminal until reset()
    }
    if (thread_.joinable()) {
      return;
    }
    prio_ = post_;
    ckpts_.setSingle(post_);
    thread_ = std::jthread([this](std::stop_token stop) { process(stop); });
    cv_.notify_all();
  }

  void stop() {
    {
      std::scoped_lock lock(mtx_);
      if (thread_.joinable()) {
        thread_.request_stop();
        cv_.notify_all();
        // IMPORTANT: cannot join while holding mtx_ (deadlock risk)
      }
    }
    // Join outside lock
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  StalenessStatus status() const {
    std::shared_lock lock(mtx_);
    const double imu_head = imus_.empty()
                                ? -std::numeric_limits<double>::infinity()
                                : imus_.back().t;

    StalenessStatus s{
        .imu_head_t = imu_head,
        .post_t = post_.t,
        .processed_up_to_t = processed_up_to_t_,
        .late_meas_trigger_t = late_meas_trigger_t_,
        .rebuilding = rebuild_.has_value(),
    };

    s.imu_lag = s.imu_head_t - s.post_t;
    s.meas_lag = s.imu_head_t - s.processed_up_to_t;
    if (s.late_meas_trigger_t) {
      s.trigger_age = s.imu_head_t - *s.late_meas_trigger_t;
    } else {
      s.trigger_age = 0.0;
    }
    return s;
  }

  void reset(double t0 = 0.0, const Estimate& post0 = {}) {
    std::scoped_lock lock(mtx_);
    meas_hist_.clear();
    imus_.clear();
    post_ = {.t = t0, .est = post0};
    prio_ = {.t = t0, .est = post0};
    ckpts_.setSingle(post_);

    processed_up_to_t_ = -std::numeric_limits<double>::infinity();
    meas_next_idx_ = 0;
    rebuild_.reset();
    late_meas_trigger_t_.reset();

    halted_ = false;
    halted_reason_ = Errc::kSuccess;
    halted_t_ = -std::numeric_limits<double>::infinity();
    halted_msg_ = {};
  }

  void push_imu(Input imu) {
    std::scoped_lock lock(mtx_);

    imus_.insert(upper_bound(imus_, imu.t, {}, &Input::t), imu);
    cv_.notify_all();
  }

  void push_pose(Measurement meas) {
    std::scoped_lock lock(mtx_);
    const auto insert_pt = upper_bound(meas_hist_, meas.t, {}, &Measurement::t);
    const auto insert_idx = std::distance(meas_hist_.begin(), insert_pt);
    meas_hist_.insert(insert_pt, meas);

    // If the new measurement is inserted strictly *before* the next-to-process
    // index, we must advance meas_next_idx_ to keep it pointing at the same
    // logical "next unprocessed" element.
    //
    // IMPORTANT: use strict `<` (not `<=`).
    // - If insert_idx == meas_next_idx_, the newly inserted measurement is
    //   exactly the next one to process and must NOT be skipped.
    // - Using `<=` would increment meas_next_idx_ too eagerly and cause
    //   first-arrival or coincident measurements to be silently ignored.
    if (std::cmp_less(insert_idx, meas_next_idx_)) {
      ++meas_next_idx_;
    }

    if (meas.t <= processed_up_to_t_) {
      // If it's too old to replay, don't arm rebuild. (Still keep it in
      // history; it'll be skipped naturally.)
      if (!imus_.empty()) {
        const double keep_from = imus_.back().t - max_ckpt_age_;
        if (meas.t < keep_from) {
          cv_.notify_all();
          return;
        }
      }

      if (!late_meas_trigger_t_ || meas.t < *late_meas_trigger_t_) {
        late_meas_trigger_t_ = meas.t;
      }
    }

    cv_.notify_all();
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
      ckpts_.setSingle(post_);
    }

    if (late_meas_trigger_t_ && late_meas_trigger_t_.value() < keep_from) {
      late_meas_trigger_t_.reset();
    }
  }

  void startRebuild(double trigger_t) {
    // PRE: mtx is held, imus_ non-empty, ckpts_ non-empty

    RebuildPlan plan;
    plan.last_meas_encountered_t = processed_up_to_t_;

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

  struct EstimateWithStatus {
    Context ctx;
    EstimationOutcome outcome;
  };

  EstimateWithStatus getEstimate(double time) {
    Context ctx;
    std::vector<Input> imus_copy;  // Vector for better cache locality; Maybe
                                   // even inplace_vector
    {
      std::shared_lock lock(mtx_);  // Reader lock here; quickly unlocked since
      ctx = post_;
      if (imus_.empty()) {
        return {.ctx = ctx,
                .outcome = EstimationOutcome(ctx.t, Errc::kSuccess)};
      }

      if (time <= ctx.t) {
        return {.ctx = ctx,
                .outcome = EstimationOutcome(ctx.t, Errc::kSuccess)};
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
        return {.ctx = ctx,
                .outcome = EstimationOutcome(ctx.t, Errc::kSuccess)};
      }
      imus_copy.assign(it, sent);
      time = target_t;
    }

    EstimationOutcome outcome;
    // Successful if not overwritten in an error callback
    outcome.status = Errc::kSuccess;
    propagateImpl(
        imus_copy, ctx, time, [](auto&& /*swallow*/) {},
        [&outcome](const auto& ctx, auto ec) {
          outcome = EstimationOutcome(ctx.t, ec);
        });

    if (outcome.success()) {
      outcome.t = ctx.t;  // Sync final time in case of success
    }
    return {.ctx = ctx, .outcome = outcome};
  }

  // Returns false if rebuild encountered a predict failure (hard stop).
  bool stepRebuild() {
    // PRE: mtx held, rebuild_ engaged
    auto& plan = *rebuild_;

    for (int k = 0; k < max_events_; ++k) {
      if (plan.ctx.t >= plan.target_head_t) {
        break;
      }

      const bool have_next_imu = plan.imu_it_next < plan.imus.size();
      const double next_imu_t =
          have_next_imu ? plan.imus[plan.imu_it_next].t : plan.target_head_t;

      const bool have_next_meas = plan.meas_idx < plan.meas.size();
      const double next_meas_t =
          have_next_meas ? plan.meas[plan.meas_idx].t : plan.target_head_t;

      const double next_t =
          std::min({next_imu_t, next_meas_t, plan.target_head_t});
      const double dt = next_t - plan.ctx.t;

      if (dt > 0) {
        // Use the same atomicity policy as streaming
        auto ec = predictImpl(plan.ctx, *plan.u_zoh, dt);
        if (IsSuccess(ec.errorCode())) {
          plan.ctx.t += dt;
          plan.new_ckpts.tryPush(plan.ctx);
        } else {
          // Prediction failure implies no time advance; we stop propagation to
          // avoid stalling. Recovery (e.g. dt splitting / reinit) is handled
          // by higher-level policy (not implemented here).
          updateLastPredictOutcome(plan.ctx, ec);
          return false;
        }
      }

      // If we hit a measurement time, apply all measurements at this time

      while (plan.meas_idx < plan.meas.size() &&
             plan.meas[plan.meas_idx].t <= plan.ctx.t) {
        // assert(plan.meas[plan.meas_idx].t <= plan.ctx.t);
        const auto& meas = plan.meas[plan.meas_idx];

        plan.last_meas_encountered_t =
            std::max(plan.last_meas_encountered_t, meas.t);

        auto ec = correctImpl(plan.ctx, meas);
        updateLastCorrectOutcome(plan.ctx, ec);

        // We checkpoint after *encountering* a measurement timestamp even
        // if the update rejects/fails.
        //
        // Rationale:
        // - The driver treats measurement timestamps as "handled" once
        // encountered; we do not keep retrying the same measurement
        // indefinitely.
        // - Checkpoints are primarily replay seeds / time anchors, not a
        // guarantee that fusion succeeded.
        // - Acceptance/rejection is surfaced via last_correct_outcome_ (and
        // future counters), not via checkpoint presence.
        plan.ctx.t = meas.t;
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
      post_ = plan.ctx;
      ckpts_ = std::move(plan.new_ckpts);

      // Update processed frontier: we have incorporated all measurements <=
      // head_t
      processed_up_to_t_ =
          std::max(processed_up_to_t_, plan.last_meas_encountered_t);

      // Set meas_next_idx_ to first measurement with t > processed_up_to_t_
      auto it =
          upper_bound(meas_hist_, processed_up_to_t_, {}, &Measurement::t);
      meas_next_idx_ =
          static_cast<size_t>(std::distance(meas_hist_.begin(), it));

      pruneHistory(post_.t);
      rebuild_.reset();
    }

    return true;
  }

  enum class ProcessResult { kContinue, kExit };

  void process(std::stop_token stop) {
    while (!stop.stop_requested()) {
      std::unique_lock lock(mtx_);
      cv_.wait(lock, stop, [this, &stop] {
        return                                     // Wake conditions:
            stop.stop_requested()                  // 1. Exit if stopping
            || late_meas_trigger_t_.has_value()    // 2. Start rebuilding
            || rebuild_.has_value()                // 3. Rebuilding
            || meas_next_idx_ < meas_hist_.size()  // 4. Process measurements
            || (!imus_.empty() && imus_.back().t > post_.t);  // 5. Process IMU
      });

      if (halted_) {
        return;
      }

      switch (processOnceImpl()) {
        case ProcessResult::kContinue:
          break;  // Break from switch and continue loop
        case ProcessResult::kExit:
          return;
      }
    }
  }

  ProcessResult processOnce() {
    std::scoped_lock lock(mtx_);
    return processOnceImpl();
  }

  Algorithm& algorithm() { return alg_; }

  const Algorithm& algorithm() const { return alg_; }

  int max_events() const {
    std::shared_lock lock(mtx_);
    return max_events_;
  }

  bool setMaxEvents(int max_events) {
    std::scoped_lock lock(mtx_);
    if (max_events <= 0) {
      return false;
    }
    max_events_ = max_events;
    return true;
  }

  bool setMaxCkptAge(double max_ckpt_age) {
    std::scoped_lock lock(mtx_);
    if (max_ckpt_age <= 0) {
      return false;
    }
    max_ckpt_age_ = max_ckpt_age;
    return true;
  }

  double maxCkptAge() const {
    std::shared_lock lock(mtx_);
    return max_ckpt_age_;
  }

  EstimationOutcome last_predict_outcome() const {
    std::shared_lock lock(mtx_);
    return last_predict_outcome_;
  }

  EstimationOutcome last_correct_outcome() const {
    std::shared_lock lock(mtx_);
    return last_correct_outcome_;
  }

  bool halted() const {
    std::shared_lock lock(mtx_);
    return halted_;
  }

  EstimationOutcome getHaltedOutcome() const {
    std::shared_lock lock(mtx_);
    return EstimationOutcome(halted_t_, halted_reason_, halted_msg_);
  }

 private:
  PredictEC predictImpl(Context& ctx, const Input& u_zoh, double dt) const {
    PredictEC ec;
    if constexpr (kPredictIsAtomic) {
      ec = alg_.predict(ctx.est, u_zoh, dt);
    } else {
      auto est_cand = ctx.est;
      ec = alg_.predict(est_cand, u_zoh, dt);
      if (IsSuccess(ec.errorCode())) {
        ctx.est = std::move(est_cand);
      }
    }
    return ec;
  }

  CorrectEC correctImpl(Context& ctx, const Measurement& meas) const {
    CorrectEC ec;
    if constexpr (kCorrectIsAtomic) {
      ec = alg_.correct(ctx.est, meas);
    } else {
      auto cand_est = ctx.est;
      ec = alg_.correct(cand_est, meas);
      if (IsSuccess(ec.errorCode())) {
        ctx.est = std::move(cand_est);
      }
    }
    return ec;
  }

  template <std::ranges::bidirectional_range R, typename OnSuccess,
            typename OnFailure>
    requires std::same_as<std::ranges::range_value_t<R>, Input>
  bool propagateImpl(const R& imus, Context& ctx, double target_t,
                     OnSuccess&& on_success, OnFailure&& on_failure) const {
    if (target_t <= ctx.t) {
      return true;  // already there
    }
    if (imus.empty()) {
      return false;  // cannot advance
    }

    const auto imu_begin = std::ranges::begin(imus);
    const auto imu_end = std::ranges::end(imus);

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
      if (dt > 0) {
        auto ec = predictImpl(ctx, *u_zoh, dt);
        if (IsSuccess(ec.errorCode())) {
          ctx.t += dt;
          on_success(std::as_const(ctx));
        } else {
          // Prediction failure implies no time advance; we stop propagation to
          // avoid stalling. Recovery (e.g. dt splitting) is an outer policy.
          on_failure(std::as_const(ctx), ec);
          return false;  // did not reach target_t
        }
      }

      if (it == imu_end || ctx.t == target_t) {
        break;
      }
      u_zoh = &*it;
      ++it;
    }
    return (ctx.t >= target_t);
  }

  bool propagateTo(const std::deque<Input>& imus, Context& ctx,
                   double target_t) {
    return propagateImpl(
        imus, ctx, target_t, [this](const auto& c) { ckpts_.tryPush(c); },
        [this](const auto&... args) { updateLastPredictOutcome(args...); });
  }

  ProcessResult processOnceImpl() {
    if (halted_) {
      return ProcessResult::kExit;  // or kContinue if you prefer “no-op”
    }

    if (imus_.empty()) {
      return ProcessResult::kContinue;
    }
    const auto head_t = imus_.back().t;
    if (rebuild_) {
      if (!stepRebuild()) {
        haltAt(last_predict_outcome_.t, last_predict_outcome_.status,
               last_predict_outcome_.message);
        // Rebuild predict failure is fatal-for-progress
        return ProcessResult::kExit;
      }
      return ProcessResult::kContinue;
    }

    // If a late trigger exists but is outside the retained horizon, drop it.
    // (Prevents pointless rebuilds from pruned history.)
    if (late_meas_trigger_t_.has_value()) {
      const double keep_from = head_t - max_ckpt_age_;
      if (late_meas_trigger_t_.value() < keep_from) {
        late_meas_trigger_t_.reset();
      }
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
        // Break out of loop only since we may still need to propagate to
        // head_t below
        break;
      }

      if (meas.t < imus_.front().t) {
        // This measurement is before any available IMU data, so we can't
        // propagate and can return now
        ++meas_next_idx_;
        return ProcessResult::kContinue;
      }

      Context seed = ckpts_.get(meas.t);
      ckpts_.eraseAfter(meas.t);

      prio_ = seed;
      if (!propagateTo(imus_, prio_, meas.t)) {
        haltAt(last_predict_outcome_.t, last_predict_outcome_.status,
               last_predict_outcome_.message);
        return ProcessResult::kExit;  // Propagation error is fatal
      }
      CorrectEC ec = correctImpl(prio_, meas);
      prio_.t = meas.t;
      updateLastCorrectOutcome(prio_, ec);

      ckpts_.tryPush(prio_);  // checkpoint after measurement update
      if (!propagateTo(imus_, prio_, head_t)) {
        haltAt(last_predict_outcome_.t, last_predict_outcome_.status,
               last_predict_outcome_.message);
        return ProcessResult::kExit;  // Propagation error is fatal
      }

      post_ = prio_;
      processed_up_to_t_ = std::max(processed_up_to_t_, meas.t);
      ++meas_next_idx_;

      pruneHistory(post_.t);  // Retain bounded history
    }

    if (head_t > post_.t) {
      prio_ = post_;
      if (!propagateTo(imus_, prio_, imus_.back().t)) {
        haltAt(last_predict_outcome_.t, last_predict_outcome_.status,
               last_predict_outcome_.message);
        return ProcessResult::kExit;  // Propagation error is fatal
      }
      post_ = prio_;
      pruneHistory(post_.t);
    }
    return ProcessResult::kContinue;
  }

  void updateLastPredictOutcome(const Context& ctx,
                                const PredictEC& ec) noexcept {
    last_predict_outcome_ = EstimationOutcome(ctx.t, ec);
  }

  void updateLastCorrectOutcome(const Context& ctx,
                                const CorrectEC& ec) noexcept {
    last_correct_outcome_ = EstimationOutcome(ctx.t, ec);
  }

  void haltAt(double t, Errc reason, std::string_view msg = {}) {
    halted_ = true;
    halted_reason_ = reason;
    halted_t_ = t;
    halted_msg_ = msg;

    if (thread_.joinable()) {
      thread_.request_stop();
    }
    cv_.notify_all();
  }

  Algorithm alg_;

  // Time-ordered measurement history
  std::deque<Measurement> meas_hist_;

  // processed_up_to_t_ should usually mean "handled", not "successfully
  // applied".
  double processed_up_to_t_ = -std::numeric_limits<double>::infinity();

  // next measurement in meas_hist_ to be processed in normal streaming mode
  size_t meas_next_idx_ = 0;

  std::deque<Input> imus_;

  Context prio_;
  Context post_;

  // time-ordered checkpoint history covering (head_t - max_age) until head_t
  Checkpoint<Context> ckpts_;
  double max_ckpt_age_ = 10.0;

  mutable std::shared_mutex mtx_;
  std::condition_variable_any cv_;
  std::jthread thread_;

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

    double last_meas_encountered_t = -std::numeric_limits<double>::infinity();

    // Rebuilt checkpoints
    Checkpoint<Context> new_ckpts = {.period = 0.02};

    Context ctx;  // current in-progress context during replay

    double target_head_t = 0.0;
  };
  std::optional<RebuildPlan> rebuild_;
  std::optional<double> late_meas_trigger_t_;
  int max_events_ = 200;

  EstimationOutcome last_predict_outcome_;
  EstimationOutcome last_correct_outcome_;
  bool halted_ = false;
  Errc halted_reason_ = Errc::kSuccess;  // or kUnknown
  double halted_t_ = -std::numeric_limits<double>::infinity();
  std::string_view halted_msg_;
};

}  // namespace eskf

#endif  // ESKF_BASELINE_INERTIAL_ODOMETRY_DRIVER_HPP_
