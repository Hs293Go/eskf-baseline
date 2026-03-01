#include "eskf_baseline/inertial_odometry_driver.hpp"
#include "fixtures.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::_;
using testing::AtLeast;
using testing::DoubleNear;
using testing::IsEmpty;
using testing::SizeIs;

class TestDriver : public ::testing::Test {
 public:
  struct TimeUpdateCall {
    std::int64_t seq;
    double dt;

    friend std::ostream& operator<<(std::ostream& os,
                                    const TimeUpdateCall& call) {
      return os << "TimeUpdateCall(seq=" << call.seq << ", dt=" << call.dt
                << ")";
    }
  };

  struct MeasurementUpdateCall {
    std::int64_t seq;

    friend std::ostream& operator<<(std::ostream& os,
                                    const MeasurementUpdateCall& call) {
      return os << "MeasurementUpdateCall(seq=" << call.seq << ")";
    }
  };

  std::vector<TimeUpdateCall> time_updates;

  std::vector<MeasurementUpdateCall> measurement_updates;

  void SetUp() override {
    ON_CALL(driver_.algorithm(), predict)
        .WillByDefault([this](auto& ctx, const auto& u, double dt) {
          time_updates.push_back({u.seq, dt});
          return eskf::BasicErrorContext{.ec = eskf::Errc::kSuccess};
          ;
        });

    ON_CALL(driver_.algorithm(), correct)
        .WillByDefault([this](auto& ctx, const auto& y) {
          measurement_updates.push_back({y.seq});
          return eskf::BasicErrorContext{.ec = eskf::Errc::kSuccess};
        });
  }

 protected:
  eskf::InertialOdometryDriver<NiceMockFilter> driver_;
};

static void DrainUntilSteady(eskf::InertialOdometryDriver<NiceMockFilter>& d,
                             int max_iters = 5000) {
  for (int i = 0; i < max_iters; ++i) {
    auto s = d.status();
    // "steady": no rebuild, no late trigger, post caught up to head (or no IMU)
    if (!s.rebuilding && !s.late_meas_trigger_t.has_value() &&
        (s.imu_head_t <= s.post_t + 1e-12)) {
      return;
    }
    d.processOnce();
  }
  FAIL() << "DrainUntilSteady hit max_iters without steady state";
}

static void PumpN(eskf::InertialOdometryDriver<NiceMockFilter>& d, int n) {
  for (int i = 0; i < n; ++i) {
    d.processOnce();
  }
}

static void PumpUntil(eskf::InertialOdometryDriver<NiceMockFilter>& d,
                      std::invocable auto pred, int max_iters = 200) {
  for (int i = 0; i < max_iters; ++i) {
    if (pred()) {
      return;
    }
    d.processOnce();
  }
  FAIL() << __func__ << " hit max_iters without reaching condition";
}

class TestDriverGetState : public TestDriver {
 protected:
  std::vector<double> times_ = {10.0, 10.1, 10.2, 10.3, 10.4};
};

TEST_F(TestDriver, PushDataWithoutProcessing) {
  driver_.reset(0.0);

  EXPECT_CALL(driver_.algorithm(), predict).Times(0);
  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  driver_.push_imu({.t = 0.1, .seq = 1});
  driver_.push_pose({.t = 0.2, .seq = 2});
}

TEST_F(TestDriverGetState, GetStateExactlyAtPost) {
  driver_.reset(times_.front());

  EXPECT_CALL(driver_.algorithm(), predict).Times(0);
  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  const auto ctx =
      driver_.getEstimate(times_.front());  // time <= post.t => early return
  EXPECT_DOUBLE_EQ(ctx.t, times_.front());
}

TEST_F(TestDriverGetState, GetStateExactlyAtImu) {
  driver_.reset(10.0);
  for (int i = 0; i < 5; ++i) {
    driver_.push_imu({.t = times_[i], .seq = i});
  }

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  const auto ctx = driver_.getEstimate(10.1);
  EXPECT_DOUBLE_EQ(ctx.t, 10.1);

  // Exactly one segment: [10.0 -> 10.1] held by seq0
  ASSERT_THAT(time_updates, SizeIs(1));
  EXPECT_EQ(time_updates[0].seq, 0);
  EXPECT_NEAR(time_updates[0].dt, 0.1, 1e-12);
}

TEST_F(TestDriverGetState, GetStateBetweenPostAndFirstImu) {
  driver_.reset(times_.front());

  EXPECT_CALL(driver_.algorithm(), predict).Times(0);
  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  driver_.push_imu(
      MockFilter::Input{.t = times_.back(), .seq = std::ssize(times_) - 1});

  const auto ctx = driver_.getEstimate(times_.front() + 0.2);

  ASSERT_THAT(time_updates, IsEmpty());
  EXPECT_DOUBLE_EQ(ctx.t, times_.front());
}

TEST_F(TestDriverGetState, GetStateAfterPostAndBetweenImus) {
  driver_.reset(times_.front() + 0.05);
  for (int i = 0; i < 2; ++i) {
    driver_.push_imu({.t = times_[i], .seq = i});
  }

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  const auto ctx = driver_.getEstimate(times_.front() + 0.08);
  EXPECT_DOUBLE_EQ(ctx.t, times_.front() + 0.08);

  ASSERT_THAT(time_updates, SizeIs(1));
  EXPECT_EQ(time_updates[0].seq, 0);
  EXPECT_NEAR(time_updates[0].dt, 0.03, 1e-12);
}

TEST_F(TestDriverGetState, GetStateSameTimeAsPostIgnoreImus) {
  driver_.reset(times_.front());

  EXPECT_CALL(driver_.algorithm(), predict).Times(0);
  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  for (int i = 0; i < 5; ++i) {
    driver_.push_imu(MockFilter::Input{.t = times_[i], .seq = i});
  }

  const auto ctx = driver_.getEstimate(times_.front());
  EXPECT_DOUBLE_EQ(ctx.t, times_.front());
}

TEST_F(TestDriverGetState, GetStateAfterPostAndPostBeforeAllImus) {
  driver_.reset(times_.front() - 0.03);
  for (int i = 0; i < 5; ++i) {
    driver_.push_imu({.t = times_[i], .seq = i});
  }

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  const auto ctx = driver_.getEstimate(times_.front() + 0.25);
  EXPECT_DOUBLE_EQ(ctx.t, times_.front() + 0.25);

  ASSERT_THAT(time_updates, SizeIs(4));
  EXPECT_EQ(time_updates[0].seq, 0);
  EXPECT_NEAR(time_updates[0].dt, 0.03, 1e-12);
  EXPECT_EQ(time_updates[1].seq, 0);
  EXPECT_NEAR(time_updates[1].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[2].seq, 1);
  EXPECT_NEAR(time_updates[2].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[3].seq, 2);
  EXPECT_NEAR(time_updates[3].dt, 0.05, 1e-12);
}

TEST_F(TestDriverGetState, GetStateAfterPostAndPostAtSomeImu) {
  driver_.reset(times_[1]);
  for (int i = 0; i < 5; ++i) {
    driver_.push_imu({.t = times_[i], .seq = i});
  }

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  const auto ctx = driver_.getEstimate(times_[1] + 0.25);
  EXPECT_DOUBLE_EQ(ctx.t, times_[1] + 0.25);

  ASSERT_THAT(time_updates, SizeIs(3));
  EXPECT_EQ(time_updates[0].seq, 1);
  EXPECT_NEAR(time_updates[0].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[1].seq, 2);
  EXPECT_NEAR(time_updates[1].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[2].seq, 3);
  EXPECT_NEAR(time_updates[2].dt, 0.05, 1e-12);
}

TEST_F(TestDriverGetState, GetStateAfterPostAndPostAfterSomeImus) {
  driver_.reset(times_[1]);
  for (int i = 0; i < 5; ++i) {
    driver_.push_imu({.t = times_[i], .seq = i});
  }

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  const auto ctx = driver_.getEstimate(times_.front() + 0.25);
  EXPECT_DOUBLE_EQ(ctx.t, times_.front() + 0.25);

  ASSERT_THAT(time_updates, SizeIs(2));
  EXPECT_EQ(time_updates[0].seq, 1);
  EXPECT_NEAR(time_updates[0].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[1].seq, 2);
  EXPECT_NEAR(time_updates[1].dt, 0.05, 1e-12);
}

TEST_F(TestDriverGetState, GetStateAfterPostAndPostAfterAllImus) {
  driver_.reset(times_.back());

  EXPECT_CALL(driver_.algorithm(), predict).Times(0);
  EXPECT_CALL(driver_.algorithm(), correct).Times(0);
  for (int i = 0; i < 5; ++i) {
    driver_.push_imu(MockFilter::Input{.t = times_[i], .seq = i});
  }
  const auto ctx = driver_.getEstimate(times_.back() + 0.1);
  EXPECT_DOUBLE_EQ(ctx.t, times_.back());
}

TEST_F(TestDriverGetState, GetStateAtLastImu) {
  driver_.reset(times_.front());
  for (int i = 0; i < 5; ++i) {
    driver_.push_imu({.t = times_[i], .seq = i});
  }

  const auto ctx = driver_.getEstimate(times_.back());
  ASSERT_THAT(time_updates, SizeIs(4));
  EXPECT_DOUBLE_EQ(ctx.t, times_.back());
  EXPECT_EQ(time_updates[0].seq, 0);
  EXPECT_NEAR(time_updates[0].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[1].seq, 1);
  EXPECT_NEAR(time_updates[1].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[2].seq, 2);
  EXPECT_NEAR(time_updates[2].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[3].seq, 3);
  EXPECT_NEAR(time_updates[3].dt, 0.1, 1e-12);
  // seq 4 is NOT used
}

TEST_F(TestDriverGetState, GetStateAtLastImuPostBetweenImus) {
  driver_.reset(10.05);
  for (int i = 0; i < 5; ++i) {
    driver_.push_imu({.t = times_[i], .seq = i});
  }

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  const auto ctx = driver_.getEstimate(10.4);
  EXPECT_DOUBLE_EQ(ctx.t, 10.4);

  // Segments:
  // 10.05 -> 10.10 held by seq0 (dt 0.05)
  // 10.10 -> 10.20 held by seq1 (dt 0.10)
  // 10.20 -> 10.30 held by seq2 (dt 0.10)
  // 10.30 -> 10.40 held by seq3 (dt 0.10)
  ASSERT_THAT(time_updates, SizeIs(4));
  EXPECT_EQ(time_updates[0].seq, 0);
  EXPECT_NEAR(time_updates[0].dt, 0.05, 1e-12);
  EXPECT_EQ(time_updates[1].seq, 1);
  EXPECT_NEAR(time_updates[1].dt, 0.10, 1e-12);
  EXPECT_EQ(time_updates[2].seq, 2);
  EXPECT_NEAR(time_updates[2].dt, 0.10, 1e-12);
  EXPECT_EQ(time_updates[3].seq, 3);
  EXPECT_NEAR(time_updates[3].dt, 0.10, 1e-12);
}

TEST_F(TestDriverGetState, GetStateAfterAllImusAndPostBeforeAllImus) {
  driver_.reset(times_.front());
  for (int i = 0; i < 5; ++i) {
    driver_.push_imu({.t = times_[i], .seq = i});
  }

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  const auto ctx = driver_.getEstimate(times_.back() + 0.1);
  EXPECT_DOUBLE_EQ(ctx.t, times_.back());

  ASSERT_THAT(time_updates, SizeIs(4));
  EXPECT_EQ(time_updates[0].seq, 0);
  EXPECT_NEAR(time_updates[0].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[1].seq, 1);
  EXPECT_NEAR(time_updates[1].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[2].seq, 2);
  EXPECT_NEAR(time_updates[2].dt, 0.1, 1e-12);
  EXPECT_EQ(time_updates[3].seq, 3);
  EXPECT_NEAR(time_updates[3].dt, 0.1, 1e-12);
}

TEST_F(TestDriverGetState, GetStateWithDuplicateImus) {
  driver_.reset(10.0);

  driver_.push_imu({.t = 10.0, .seq = 0});
  driver_.push_imu({.t = 10.1, .seq = 1});
  driver_.push_imu({.t = 10.1, .seq = 2});  // duplicate timestamp
  driver_.push_imu({.t = 10.2, .seq = 3});

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  const auto ctx = driver_.getEstimate(10.2);
  EXPECT_DOUBLE_EQ(ctx.t, 10.2);

  // We should have exactly two positive segments: 10.0->10.1, 10.1->10.2
  // (No dt=0 segment for the duplicate.)
  ASSERT_THAT(time_updates, SizeIs(2));
  EXPECT_GT(time_updates[0].dt, 0.0);
  EXPECT_GT(time_updates[1].dt, 0.0);

  EXPECT_NEAR(time_updates[0].dt, 0.1, 1e-12);
  EXPECT_NEAR(time_updates[1].dt, 0.1, 1e-12);

  // Hold choice at the duplicate boundary:
  // first segment should be held by seq0
  EXPECT_EQ(time_updates[0].seq, 0);

  // second segment's hold depends on how upper_bound treats duplicates:
  // it will typically advance u_zoh to the first 10.1 sample (seq1) when ctx
  // reaches 10.1.
  EXPECT_TRUE(time_updates[1].seq == 1 || time_updates[1].seq == 2);
}

class TestDriverProcess : public TestDriver {};

TEST_F(TestDriver, PushOnlyImuAndProcess) {
  // Arrange
  driver_.reset(0.0);
  MockFilter::Input imu{.t = 1.0, .seq = 42};

  // We expect exactly one predict step to reach head_t = 1.0.
  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(0);
  EXPECT_CALL(driver_.algorithm(), predict(_, _, _)).Times(1);

  driver_.push_imu(imu);
  driver_.processOnce();

  // Optional: if you have a way to inspect post.t, assert it reached 1.0
  auto post = driver_.getEstimate(1.0);
  EXPECT_THAT(post.t, DoubleNear(1.0, 1e-9));
}

TEST_F(TestDriver, PushOnlyPoseAndProcess) {
  driver_.reset(0.0);
  driver_.push_pose({.t = 1.0, .seq = 1});

  EXPECT_CALL(driver_.algorithm(), predict).Times(0);
  EXPECT_CALL(driver_.algorithm(), correct).Times(0);

  driver_.processOnce();
  // still should be at post0
  EXPECT_DOUBLE_EQ(driver_.getEstimate(0.0).t, 0.0);
}

TEST_F(TestDriverProcess, UpdateOnCoincidentPoseAndImu) {
  driver_.reset(0.0);
  driver_.push_pose({.t = 1.0, .seq = 1});
  driver_.push_imu({.t = 1.0, .seq = 42});

  testing::InSequence seq;
  EXPECT_CALL(driver_.algorithm(), predict(_, _, _)).Times(AtLeast(1));
  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(1);
  driver_.processOnce();
}

TEST_F(TestDriverProcess, PoseWaitsUntilHeadImu) {
  driver_.reset(0.0);

  // Pose arrives first
  driver_.push_pose({.t = 1.0, .seq = 7});

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);
  EXPECT_CALL(driver_.algorithm(), predict).Times(0);
  driver_.processOnce();  // no IMU => returns early

  ::testing::Mock::VerifyAndClearExpectations(&driver_.algorithm());

  // Now IMU arrives up to head=1.0
  driver_.push_imu({.t = 1.0, .seq = 42});

  // This is ok
  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(1);
  EXPECT_CALL(driver_.algorithm(), predict).Times(AtLeast(1));
  driver_.processOnce();
  ASSERT_THAT(measurement_updates, SizeIs(1));
  EXPECT_EQ(measurement_updates[0].seq, 7);
}

TEST_F(TestDriverProcess, MeasurementBeyondHeadDoesNotApply) {
  driver_.reset(0.0);

  driver_.push_imu({.t = 1.0, .seq = 1});
  driver_.push_pose({.t = 1.1, .seq = 9});

  EXPECT_CALL(driver_.algorithm(), correct).Times(0);
  driver_.processOnce();  // should only propagate to head_t, not fuse meas

  ::testing::Mock::VerifyAndClearExpectations(&driver_.algorithm());

  // Extend IMU horizon
  driver_.push_imu({.t = 1.2, .seq = 2});

  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(1);
  driver_.processOnce();
}

TEST_F(TestDriverProcess, StreamingAppliesMeasurementsInTimeOrder) {
  driver_.reset(0.0);

  // IMUs cover everything
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_imu({.t = 3.0, .seq = 12});
  driver_.push_imu({.t = 4.0, .seq = 13});

  driver_.push_pose({.t = 1.0, .seq = 1});
  driver_.push_pose({.t = 2.0, .seq = 2});
  driver_.push_pose({.t = 3.0, .seq = 3});

  driver_.processOnce();
  ASSERT_THAT(measurement_updates, SizeIs(3));
  EXPECT_EQ(measurement_updates[0].seq, 1);
  EXPECT_EQ(measurement_updates[1].seq, 2);
  EXPECT_EQ(measurement_updates[2].seq, 3);
}

TEST_F(TestDriver, LateMeasurementRebuildIsIncrementalAndCommits) {
  driver_.reset(0.0);

  // IMU coverage up to head_t=4.0
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_imu({.t = 3.0, .seq = 12});
  driver_.push_imu({.t = 4.0, .seq = 13});

  // On-time measurements
  driver_.push_pose({.t = 1.0, .seq = 1});
  driver_.push_pose({.t = 2.0, .seq = 2});
  driver_.push_pose({.t = 3.0, .seq = 4});

  // First streaming pass applies 1,2,4
  driver_.processOnce();
  ASSERT_THAT(measurement_updates, SizeIs(3));
  measurement_updates.clear();

  // Late measurement arrives
  driver_.push_pose({.t = 2.2, .seq = 3});

  // Make rebuild very incremental
  driver_.setMaxEvents(1);

  // Step 0: this call should ONLY arm rebuild (startRebuild) and return.
  driver_.processOnce();
  ASSERT_THAT(measurement_updates, IsEmpty());

  // Step 1: now rebuild_ is active, so stepRebuild runs (1 event max).
  // Depending on whether the first event is IMU propagation vs meas, there may
  // still be 0 measurement updates. So we don't assert >0 yet.
  driver_.processOnce();

  // Keep stepping until we observe all 3 replayed measurement updates.
  PumpUntil(driver_, [&] { return measurement_updates.size() >= 3; });

  ASSERT_THAT(measurement_updates, SizeIs(3));
  EXPECT_EQ(measurement_updates[0].seq, 2);
  EXPECT_EQ(measurement_updates[1].seq, 3);
  EXPECT_EQ(measurement_updates[2].seq, 4);
}

TEST_F(TestDriverProcess,
       MixedMeasurementsOnlyApplyUpToHeadImuThenContinueLater) {
  driver_.reset(0.0);

  // IMU head is 3.0 initially
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_imu({.t = 3.0, .seq = 12});

  // Measurements: two within head, one beyond
  driver_.push_pose({.t = 1.0, .seq = 1});
  driver_.push_pose({.t = 2.0, .seq = 2});
  driver_.push_pose({.t = 4.0, .seq = 4});

  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(2);
  EXPECT_CALL(driver_.algorithm(), predict(_, _, _)).Times(AtLeast(1));

  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });

  ASSERT_THAT(measurement_updates, SizeIs(2));
  EXPECT_EQ(measurement_updates[0].seq, 1);
  EXPECT_EQ(measurement_updates[1].seq, 2);

  ::testing::Mock::VerifyAndClearExpectations(&driver_.algorithm());

  // Now extend IMU horizon so the t=4.0 measurement becomes processable
  driver_.push_imu({.t = 4.0, .seq = 13});

  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(1);
  EXPECT_CALL(driver_.algorithm(), predict(_, _, _)).Times(AtLeast(1));

  driver_.processOnce();

  ASSERT_THAT(measurement_updates, SizeIs(3));
  EXPECT_EQ(measurement_updates[2].seq, 4);
}

TEST_F(TestDriverProcess, CoincidentPosesAppliedInStableOrder) {
  driver_.reset(0.0);

  // IMU coverage beyond t=2.0
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_imu({.t = 3.0, .seq = 12});

  // Two poses at the same timestamp, inserted in arrival order.
  driver_.push_pose({.t = 2.0, .seq = 21});
  driver_.push_pose({.t = 2.0, .seq = 22});

  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(2);
  EXPECT_CALL(driver_.algorithm(), predict(_, _, _)).Times(AtLeast(1));

  driver_.processOnce();

  ASSERT_THAT(measurement_updates, SizeIs(2));
  // With upper_bound insertion, duplicates should preserve arrival order.
  EXPECT_EQ(measurement_updates[0].seq, 21);
  EXPECT_EQ(measurement_updates[1].seq, 22);
}

TEST_F(TestDriverProcess,
       MultipleLatePosesAggregateToMinTriggerAndReplayCorrectWindow) {
  driver_.reset(0.0);

  // IMU coverage up to head_t = 4.0
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_imu({.t = 3.0, .seq = 12});
  driver_.push_imu({.t = 4.0, .seq = 13});

  // On-time poses applied in first streaming pass
  driver_.push_pose({.t = 1.1, .seq = 1});
  driver_.push_pose({.t = 2.1, .seq = 2});
  driver_.push_pose({.t = 3.1, .seq = 3});

  driver_.processOnce();
  ASSERT_THAT(measurement_updates, SizeIs(3));
  EXPECT_EQ(measurement_updates[0].seq, 1);
  EXPECT_EQ(measurement_updates[1].seq, 2);
  EXPECT_EQ(measurement_updates[2].seq, 3);

  measurement_updates.clear();
  time_updates.clear();

  // Now insert multiple late poses (all < processed_up_to_t_).
  // The trigger should become the minimum: 2.1.
  driver_.push_pose({.t = 2.9, .seq = 29});
  driver_.push_pose({.t = 2.2, .seq = 21});  // min trigger
  driver_.push_pose({.t = 2.5, .seq = 25});

  // Drive rebuild to completion.
  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return !s.rebuilding && !s.late_meas_trigger_t.has_value() &&
           (s.imu_head_t <= s.post_t + 1e-12);
  });
  // Expected rebuild window seed:
  // trigger_t = 2.1 -> seed ckpt should be at 2.0 (from streaming meas
  // boundary) so replay window is [2.0, head=4.0]
  //
  // Measurements replayed in-window, time order:
  //   2.0 (seq 2) replayed
  //   2.1 (seq 21) late
  //   2.5 (seq 25) late
  //   2.9 (seq 29) late
  //   3.0 (seq 3) replayed
  //
  // Measurement at 1.0 (seq 1) must NOT replay (before seed).
  ASSERT_THAT(measurement_updates, SizeIs(5));
  EXPECT_EQ(measurement_updates[0].seq, 2);
  EXPECT_EQ(measurement_updates[1].seq, 21);
  EXPECT_EQ(measurement_updates[2].seq, 25);
  EXPECT_EQ(measurement_updates[3].seq, 29);
  EXPECT_EQ(measurement_updates[4].seq, 3);
}

TEST_F(TestDriverProcess, PoseAtImuFrontTimeIsProcessable) {
  driver_.reset(0.0);

  driver_.push_imu({.t = 1.0, .seq = 10});  // front=head=1.0
  driver_.push_pose({.t = 1.0, .seq = 7});

  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(1);
  EXPECT_CALL(driver_.algorithm(), predict(_, _, _)).Times(AtLeast(1));

  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });

  ASSERT_THAT(measurement_updates, SizeIs(1));
  EXPECT_EQ(measurement_updates[0].seq, 7);
}

TEST_F(TestDriverProcess, PoseOlderThanImuFrontIsSkippedAndNotRetried) {
  driver_.reset(0.0);

  // Oldest available IMU is 2.0
  driver_.push_imu({.t = 2.0, .seq = 20});
  driver_.push_imu({.t = 3.0, .seq = 21});

  // Measurement before IMU front => must be skipped
  driver_.push_pose({.t = 1.0, .seq = 99});

  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(0);
  EXPECT_CALL(driver_.algorithm(), predict(_, _, _)).Times(AtLeast(1));

  // First call should skip it and then still propagate to head (3.0)
  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });

  // Now, even if we call processOnce again, the skipped measurement must not
  // reappear.
  ::testing::Mock::VerifyAndClearExpectations(&driver_.algorithm());
  EXPECT_CALL(driver_.algorithm(), correct(_, _)).Times(0);
  driver_.processOnce();
}

TEST_F(TestDriverProcess, LateCoincidentPoseRebuildReplaysBothInStableOrder) {
  driver_.reset(0.0);

  // IMU coverage
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_imu({.t = 3.0, .seq = 12});

  // On-time measurement at t=2.0
  driver_.push_pose({.t = 2.0, .seq = 20});

  // First streaming pass: apply seq=20 once
  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });

  ASSERT_THAT(measurement_updates, SizeIs(1));
  EXPECT_EQ(measurement_updates[0].seq, 20);
  measurement_updates.clear();

  // Late duplicate at same timestamp
  driver_.push_pose({.t = 2.0, .seq = 21});

  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return !s.rebuilding && !s.late_meas_trigger_t.has_value() &&
           (s.imu_head_t <= s.post_t + 1e-12);
  });

  // Rebuild should replay both measurements at t=2.0, in stable order:
  // original (20) then late (21).
  ASSERT_THAT(measurement_updates, SizeIs(2));
  EXPECT_EQ(measurement_updates[0].seq, 20);
  EXPECT_EQ(measurement_updates[1].seq, 21);
}

TEST_F(TestDriverProcess, PruningClearsLateMeasTriggerWhenOutOfWindow) {
  driver_.reset(0.0);

  // Step A: create a processed frontier so "late" is meaningful
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_pose({.t = 2.0, .seq = 20});
  driver_.processOnce();  // apply seq=20, processed_up_to_t_ >= 2.0

  measurement_updates.clear();

  // Step B: insert a late measurement at t=1.5 => sets late_meas_trigger_t_
  driver_.push_pose({.t = 1.5, .seq = 15});
  {
    auto s = driver_.status();
    ASSERT_TRUE(s.late_meas_trigger_t.has_value());
    EXPECT_NEAR(*s.late_meas_trigger_t, 1.5, 1e-12);
  }

  // Step C: advance IMU head far enough that keep_from > 1.5, causing prune to
  // clear it. keep_from = head - max_ckpt_age_. If max_ckpt_age_ defaults 10,
  // choose head=100.
  driver_.push_imu({.t = 100.0, .seq = 99});

  // Pump until caught up; pruneHistory runs inside processing.
  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });

  // late_meas_trigger_t_ should now be cleared as out-of-window.
  {
    auto s = driver_.status();
    EXPECT_FALSE(s.late_meas_trigger_t.has_value());
  }
}

TEST_F(TestDriverProcess, PoseAtImuFrontBoundaryIsApplied) {
  driver_.reset(0.0);

  // IMUs start at t=1.0
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});

  // Pose exactly at imus_.front().t
  driver_.push_pose({.t = 1.0, .seq = 100});

  // Pump until post reaches head (2.0). The pose at 1.0 should be applied.
  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });

  ASSERT_THAT(measurement_updates, SizeIs(1));
  EXPECT_EQ(measurement_updates[0].seq, 100);
}

TEST_F(TestDriverProcess, PoseOlderThanImuFrontIsSkippedAndNeverRetried) {
  driver_.reset(0.0);

  // IMUs start at t=1.0
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});

  // Pose before any available IMU data => should be skipped
  driver_.push_pose({.t = 0.5, .seq = 50});

  // Run processing to head; skip should happen, no measurement updates.
  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });

  ASSERT_THAT(measurement_updates, IsEmpty());

  // Now extend IMU horizon further; skipped pose must NOT later "come back".
  driver_.push_imu({.t = 3.0, .seq = 12});
  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });

  ASSERT_THAT(measurement_updates, IsEmpty());
}

TEST_F(TestDriverProcess, PruningClearsLateMeasTriggerWhenOutsideRetention) {
  driver_.reset(0.0);

  // Retain only the last 0.25s of history.
  driver_.setMaxCkptAge(0.25);

  // IMU coverage and one on-time pose to advance processed_up_to_t_.
  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_imu({.t = 3.0, .seq = 12});
  driver_.push_pose({.t = 2.9, .seq = 29});

  // Process to head (3.0); this should apply seq=29.
  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });
  ASSERT_THAT(measurement_updates, SizeIs(1));
  measurement_updates.clear();

  // Now insert a "late" measurement far in the past.
  // Since processed_up_to_t_ >= 2.9, t=1.0 is late, but it's also outside
  // retention because keep_from = head(3.0) - 0.25 = 2.75.
  driver_.push_pose({.t = 1.0, .seq = 100});

  // Pump: we expect the trigger to get cleared (no rebuild), and post stays at
  // head.
  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return !s.rebuilding && !s.late_meas_trigger_t.has_value() &&
           (s.imu_head_t <= s.post_t + 1e-12);
  });

  // Must NOT have applied the too-old measurement.
  ASSERT_THAT(measurement_updates, IsEmpty());
}

TEST_F(TestDriverProcess, LateMeasurementOlderThanKeepFromIsIgnored) {
  driver_.reset(0.0);
  driver_.setMaxCkptAge(0.25);

  driver_.push_imu({.t = 1.0, .seq = 10});
  driver_.push_imu({.t = 2.0, .seq = 11});
  driver_.push_imu({.t = 3.0, .seq = 12});
  driver_.push_pose({.t = 2.9, .seq = 29});

  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return s.imu_head_t <= s.post_t + 1e-12;
  });
  measurement_updates.clear();

  // Insert an ancient late pose multiple times.
  for (int i = 0; i < 3; ++i) {
    driver_.push_pose({.t = 1.0, .seq = 100 + i});
  }

  PumpUntil(driver_, [&] {
    auto s = driver_.status();
    return !s.rebuilding && !s.late_meas_trigger_t.has_value() &&
           (s.imu_head_t <= s.post_t + 1e-12);
  });

  ASSERT_THAT(measurement_updates, IsEmpty());
}
