#include <cstdint>

#include "eskf_baseline/inertial_odometry_driver.hpp"
#include "gmock/gmock.h"

class MockFilter {
 public:
  struct Estimate {};

  struct Input {
    double t;
    std::int64_t seq;
  };

  struct Measurement {
    double t;
    std::int64_t seq;
  };

  MOCK_METHOD(eskf::BasicErrorContext, predict,
              (Estimate & ctx, const Input& u, double dt), (const));

  MOCK_METHOD(eskf::BasicErrorContext, correct,
              (Estimate & ctx, const Measurement& meas), (const));
};

using NiceMockFilter = ::testing::NiceMock<MockFilter>;
