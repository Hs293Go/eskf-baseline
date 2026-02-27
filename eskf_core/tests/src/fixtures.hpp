#include <cstdint>

#include "gmock/gmock.h"

class MockFilter {
 public:
  struct Context {
    double t;
  };

  struct Input {
    double t;
    std::int64_t seq;
  };

  struct Measurement {
    double t;
    std::int64_t seq;
  };

  MOCK_METHOD(bool, timeUpdate, (Context & ctx, const Input& u, double dt),
              (const));

  MOCK_METHOD(void, measurementUpdate, (Context & ctx, const Measurement& meas),
              (const));
};

using NiceMockFilter = ::testing::NiceMock<MockFilter>;
