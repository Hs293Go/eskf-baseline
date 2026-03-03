#ifndef ESKF_BASELINE_ESKF_HPP_
#define ESKF_BASELINE_ESKF_HPP_

#include "Eigen/Dense"
#include "eskf_baseline/definitions.hpp"
#include "eskf_baseline/eskf_baseline.hpp"
#include "eskf_baseline/outlier_classifier.hpp"

namespace eskf {
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime,
              Derived::RowsAtCompileTime>
EnsureSymmetric(const Eigen::MatrixBase<Derived>& m) {
  return (m + m.transpose()) / 2;
}

static constexpr int kTangentDim = eskf::NominalState<double>::kTangentDim;
using Covariance = Eigen::Matrix<double, kTangentDim, kTangentDim>;

class Eskf {
 public:
  struct Estimate {
    eskf::NominalState<double> x;
    Covariance P;
  };

  struct Input {
    double t;
    eskf::ImuInput<double> data;
  };

  struct Measurement {
    double t;
    eskf::Pose<double> data;
    Eigen::Matrix<double, 6, 6> R;
  };

  eskf::BasicErrorContext predict(Estimate& ctx, const Input& u,
                                  double dt) const;

  eskf::BasicErrorContext correct(Estimate& ctx, const Measurement& meas) const;

  bool setConfig(const eskf::Config<double>& cfg);

 private:
  OutlierClassifier outlier_classifier_ = {6, 0.95, 0.99};
  eskf::Config<double> cfg_{.accel_noise_density = 1,
                            .gyro_noise_density = 0.01};
};
}  // namespace eskf

#endif  // ESKF_BASELINE_ESKF_HPP_
