#include "Eigen/Dense"
#include "eskf_baseline/eskf_baseline.hpp"
#include "eskf_baseline/inertial_odometry_driver.hpp"

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime,
              Derived::RowsAtCompileTime>
EnsureSymmetric(const Eigen::MatrixBase<Derived>& m) {
  return (m + m.transpose()) / 2;
}

using Covariance = Eigen::Matrix<double, 18, 18>;

// TODO

class KalmanFilter {
 public:
  struct Context {
    double t;
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

  bool timeUpdate(Context& ctx, const Input& u, double dt) const {
    if (dt <= 0.0) {
      return false;
    }
    const auto [F, Q] = eskf::MotionJacobians(ctx.x, u.data, dt, {});
    ctx.P = EnsureSymmetric(F * ctx.P * F.transpose() + Q);
    ctx.x = eskf::Motion(ctx.x, u.data, dt);
    ctx.t += dt;
    return true;
  }

  void measurementUpdate(Context& ctx, const Measurement& meas) const {
    const Eigen::Vector<double, 6> y =
        meas.data.boxminus(eskf::PoseObservation(ctx.x));
    const Eigen::Matrix<double, 6, 18> H = eskf::PoseObservationJacobian(ctx.x);
    const Eigen::Matrix<double, 6, 6> S = H * ctx.P * H.transpose() + meas.R;

    const Eigen::Matrix<double, 18, 6> PHt = ctx.P * H.transpose();
    const Eigen::Matrix<double, 18, 6> K =
        S.llt().solve(PHt.transpose()).transpose();

    Covariance i_m_km = -K * H;
    i_m_km.diagonal().array() += 1.0;
    ctx.P = EnsureSymmetric(i_m_km * ctx.P * i_m_km.transpose() +
                            K * meas.R * K.transpose());
    ctx.x = ctx.x.boxplus(K * y);
    ctx.t = meas.t;
  }
};

int main() {
  eskf::InertialOdometryDriver<KalmanFilter> driver;
  driver.start();
  driver.processOnce();
}
