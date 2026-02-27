#include "eskf_baseline/inertial_odometry_driver.hpp"
#include "fixtures.hpp"
#include "gtest/gtest.h"

class TestDriverThreaded : public ::testing::Test {
 public:
  struct TimeUpdateCall {
    std::int64_t seq;
    double dt;
  };
  struct MeasurementUpdateCall {
    std::int64_t seq;
  };

  std::mutex calls_mtx_;
  std::vector<TimeUpdateCall> time_updates_;
  std::vector<MeasurementUpdateCall> measurement_updates_;

  void SetUp() override {
    ON_CALL(driver_.algorithm(), timeUpdate)
        .WillByDefault([this](auto& ctx, const auto& u, double dt) {
          {
            std::lock_guard lk(calls_mtx_);
            time_updates_.push_back({u.seq, dt});
          }
          ctx.t += dt;
          return true;
        });

    ON_CALL(driver_.algorithm(), measurementUpdate)
        .WillByDefault([this](auto&, const auto& y) {
          std::lock_guard lk(calls_mtx_);
          measurement_updates_.push_back({y.seq});
        });
  }

  std::vector<MeasurementUpdateCall> snapshot_meas() {
    std::lock_guard lk(calls_mtx_);
    return measurement_updates_;
  }
  void clear_calls() {
    std::lock_guard lk(calls_mtx_);
    measurement_updates_.clear();
    time_updates_.clear();
  }

 protected:
  eskf::InertialOdometryDriver<NiceMockFilter> driver_;
};

static void PumpUntil(
    eskf::InertialOdometryDriver<NiceMockFilter>& d,
    const std::function<bool(const eskf::StalenessStatus&)>& pred,
    int max_iters = 20000) {
  for (int i = 0; i < max_iters; ++i) {
    if (pred(d.status())) {
      return;
    }
    // yield helps the worker run without sleeping for long
    std::this_thread::yield();
  }
  FAIL() << "PumpUntil hit max_iters: " << max_iters
         << "without reaching condition";
}

auto IsIdleAndCaughtUp(const eskf::StalenessStatus& s) {
  // meaning: no rebuild pending/active and post caught up to IMU head
  return !s.rebuilding && !s.late_meas_trigger_t.has_value() &&
         (s.imu_head_t <= s.post_t + 1e-12);
};

TEST_F(TestDriverThreaded, StartStopIdempotent_NoDeadlock) {
  driver_.reset({.t = 0.0});

  driver_.start();
  driver_.start();  // idempotent

  driver_.push_imu({.t = 1.0, .seq = 1});
  driver_.push_pose({.t = 1.0, .seq = 10});

  // Give worker a chance (but don't rely on time; just pump via processOnce
  // too)
  PumpUntil(driver_, IsIdleAndCaughtUp);

  driver_.stop();
  driver_.stop();  // idempotent

  SUCCEED();
}

TEST_F(TestDriverThreaded, ResetWhileRunning) {
  driver_.reset({.t = 0.0});
  driver_.start();

  // Phase 1: push and wait until caught up
  driver_.push_imu({.t = 1.0, .seq = 1});
  driver_.push_imu({.t = 2.0, .seq = 2});
  driver_.push_pose({.t = 2.0, .seq = 20});

  PumpUntil(driver_, [](const eskf::StalenessStatus& s) {
    return !s.rebuilding && !s.late_meas_trigger_t.has_value() &&
           (s.imu_head_t <= s.post_t + 1e-12) &&
           (s.processed_up_to_t >= 2.0 - 1e-12);
  });

  {
    auto meas = snapshot_meas();
    ASSERT_TRUE(std::any_of(meas.begin(), meas.end(),
                            [](auto m) { return m.seq == 20; }));
  }

  clear_calls();

  // Reset to new epoch while worker is running
  driver_.reset({.t = 100.0});

  // Phase 2: new data only
  driver_.push_imu({.t = 100.5, .seq = 3});
  driver_.push_pose({.t = 100.5, .seq = 50});

  PumpUntil(driver_, [](const eskf::StalenessStatus& s) {
    return !s.rebuilding && !s.late_meas_trigger_t.has_value() &&
           (s.imu_head_t <= s.post_t + 1e-12) &&
           (s.processed_up_to_t >= 100.5 - 1e-12);
  });

  {
    auto meas = snapshot_meas();
    EXPECT_FALSE(std::ranges::any_of(meas, [](auto m) { return m.seq == 20; }))
        << "Old measurement leaked after reset";
    EXPECT_TRUE(std::ranges::any_of(meas, [](auto m) { return m.seq == 50; }));
  }

  driver_.stop();
}

TEST_F(TestDriverThreaded, StopHaltsProcessing) {
  driver_.reset({.t = 0.0});
  driver_.start();

  driver_.push_imu({.t = 1.0, .seq = 1});
  PumpUntil(driver_, [](auto s) { return s.post_t >= 1.0 - 1e-12; });

  driver_.stop();

  const auto before = driver_.status().post_t;

  // Push more data; worker should not process
  driver_.push_imu({.t = 2.0, .seq = 2});
  // spin a bit
  for (int i = 0; i < 10000; ++i) {
    std::this_thread::yield();
  }

  const auto after = driver_.status().post_t;
  EXPECT_DOUBLE_EQ(before, after);
}

TEST_F(TestDriverThreaded, RestartAfterStopWorks) {
  driver_.reset({.t = 0.0});
  driver_.start();

  driver_.push_imu({.t = 1.0, .seq = 1});
  PumpUntil(driver_, [](auto s) { return s.post_t >= 1.0 - 1e-12; });

  driver_.stop();

  driver_.reset({.t = 100.0});
  clear_calls();

  driver_.start();
  driver_.push_imu({.t = 100.5, .seq = 2});
  driver_.push_pose({.t = 100.5, .seq = 50});

  PumpUntil(driver_, [](auto s) {
    return (s.post_t >= 100.5 - 1e-12) &&
           (s.processed_up_to_t >= 100.5 - 1e-12);
  });

  auto meas = snapshot_meas();
  EXPECT_TRUE(std::any_of(meas.begin(), meas.end(),
                          [](auto m) { return m.seq == 50; }));

  driver_.stop();
}

TEST_F(TestDriverThreaded, ResetWakesWorker_NoHang) {
  driver_.reset({.t = 0.0});
  driver_.start();

  // Let worker block (no data).
  for (int i = 0; i < 1000; ++i) {
    std::this_thread::yield();
  }

  // Reset should not hang, and should wake worker (even though it will go back
  // to waiting).
  driver_.reset({.t = 10.0});

  // Now push data and ensure it processes from new epoch.
  driver_.push_imu({.t = 10.1, .seq = 1});
  PumpUntil(driver_, [](auto s) { return s.post_t >= 10.1 - 1e-12; });

  driver_.stop();
}

// Checks no deadlocks and monotonous increaseing time
TEST_F(TestDriverThreaded, ConcurrentGetState) {
  driver_.reset({.t = 0.0});
  driver_.start();

  // Feed a bunch of IMUs.
  for (int i = 1; i <= 200; ++i) {
    driver_.push_imu({.t = i * 0.01, .seq = i});
  }

  std::atomic<bool> stop_readers{false};
  std::vector<std::jthread> readers;

  for (int k = 0; k < 4; ++k) {
    readers.emplace_back([&](const std::stop_token& st) {
      double last_t = -1e300;
      while (!st.stop_requested() && !stop_readers.load()) {
        auto s = driver_.status();
        auto ctx = driver_.getState(s.imu_head_t);  // query at head
        // Must never go backwards.
        EXPECT_GE(ctx.t + 1e-12, last_t);
        last_t = ctx.t;
        std::this_thread::yield();
      }
    });
  }

  // Wait until worker catches up.
  PumpUntil(driver_, IsIdleAndCaughtUp);

  stop_readers.store(true);
  std::ranges::for_each(readers, std::mem_fn(&std::jthread::request_stop));

  driver_.stop();
}

TEST_F(TestDriverThreaded, StopDuringRebuild_ExitsCleanly) {
  driver_.reset({.t = 0.0});
  driver_.start();

  // IMU coverage
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_imu({.t = 3.0, .seq = 12});
  driver_.push_imu({.t = 4.0, .seq = 13});

  // On-time poses
  driver_.push_pose({.t = 1.0, .seq = 1});
  driver_.push_pose({.t = 2.0, .seq = 2});
  driver_.push_pose({.t = 3.0, .seq = 3});

  PumpUntil(driver_, IsIdleAndCaughtUp);
  clear_calls();

  // Insert late measurement to trigger rebuild
  driver_.push_pose({.t = 2.2, .seq = 22});

  // Make rebuild slow
  driver_.setMaxEvents(1);

  // Wait until rebuild actually begins
  PumpUntil(driver_, [](auto s) {
    return s.rebuilding || s.late_meas_trigger_t.has_value();
  });

  // Now stop while rebuild is plausibly in progress
  driver_.stop();

  // After stop, status must remain stable even if we yield.
  const auto before = driver_.status();
  for (int i = 0; i < 10000; ++i) std::this_thread::yield();
  const auto after = driver_.status();

  EXPECT_DOUBLE_EQ(before.post_t, after.post_t);
  EXPECT_EQ(
      after.rebuilding,
      before.rebuilding);  // not required to be false; just stable after stop
}
