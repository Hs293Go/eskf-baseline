#ifndef ESKF_MATH_ESKF_MATH_HPP_
#define ESKF_MATH_ESKF_MATH_HPP_

#include "Eigen/Dense"

namespace eskf {

namespace rotation {

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> hat(
    const Eigen::MatrixBase<Derived>& v) {
  using Scalar = typename Derived::Scalar;
  Eigen::Matrix3<Scalar> v_hat;
  v_hat << Scalar(0), -v.z(), v.y(),  //
      v.z(), Scalar(0), -v.x(),       //
      -v.y(), v.x(), Scalar(0);
  return v_hat;
}

template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> AngleAxisToQuaternion(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  using Scalar = typename Derived::Scalar;
  using std::abs;
  using std::cos;
  using std::sin;
  using std::sqrt;

  const Scalar theta_sq = angle_axis.squaredNorm();

  Scalar imag_factor;
  Scalar real_factor;

  if (theta_sq < Scalar(1e-6)) {
    const Scalar theta_po4 = theta_sq * theta_sq;
    imag_factor = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq +
                  Scalar(1.0 / 3840.0) * theta_po4;
    real_factor = Scalar(1) - Scalar(1.0 / 8.0) * theta_sq +
                  Scalar(1.0 / 384.0) * theta_po4;
  } else {
    const Scalar theta = sqrt(theta_sq);
    const Scalar half_theta = Scalar(0.5) * theta;
    const Scalar sin_half_theta = sin(half_theta);
    imag_factor = sin_half_theta / theta;
    real_factor = cos(half_theta);
  }

  Eigen::Quaternion<Scalar> quaternion;
  quaternion.w() = real_factor;
  quaternion.vec() = imag_factor * angle_axis;
  return quaternion;
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> AngleAxisToRotationMatrix(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  using Scalar = typename Derived::Scalar;

  const Scalar theta_sq = angle_axis.squaredNorm();
  Eigen::Matrix3<Scalar> rotation_matrix = Eigen::Matrix3<Scalar>::Identity();
  const Eigen::Matrix3<Scalar> hat_phi = hat(angle_axis);
  const Eigen::Matrix3<Scalar> hat_phi_sq = hat_phi * hat_phi;

  if (theta_sq < Scalar(1e-6)) {
    rotation_matrix += hat_phi + hat_phi_sq / Scalar(2);
  } else {
    const Scalar theta = sqrt(theta_sq);
    const Scalar cos_theta = cos(theta);
    const Scalar sin_theta = sin(theta);

    rotation_matrix += (Scalar(1) - cos_theta) / theta_sq * hat_phi_sq +
                       sin_theta / theta * hat_phi;
  }

  return rotation_matrix;
}
}  // namespace rotation

template <typename T>
struct NominalState {
  Eigen::Vector3<T> p;
  Eigen::Quaternion<T> q;
  Eigen::Vector3<T> v;
  Eigen::Vector3<T> accel_bias;
  Eigen::Vector3<T> gyro_bias;
};

template <typename T>
struct Input {
  Eigen::Vector3<T> accel;
  Eigen::Vector3<T> gyro;
};

template <typename T>
struct Config {
  T accel_noise_density{0.005};
  T gyro_noise_density{5e-5};
  T accel_bias_random_walk{0.001};
  T gyro_bias_random_walk{0.0001};
  Eigen::Vector3<T> grav_vector{T(0), T(0), T(-9.81)};
};

template <typename Scalar>
NominalState<Scalar> Motion(const NominalState<Scalar>& state,
                            const Input<Scalar>& input, Scalar dt,
                            const Config<Scalar>& cfg = {}) {
  const auto& [p, q, v, accel_bias, gyro_bias] = state;
  const auto& [accel, gyro] = input;
  const Eigen::Vector3<Scalar> acc_unbiased = accel - accel_bias;
  const Eigen::Vector3<Scalar> accel_world = q * acc_unbiased + cfg.grav_vector;
  const Eigen::Vector3<Scalar> delta_velocity = accel_world * dt;
  const Eigen::Vector3<Scalar> gyro_unbiased = gyro - gyro_bias;
  const Eigen::Vector3<Scalar> delta_angle = gyro_unbiased * dt;

  // f(x, u) =
  // [p + v*dt;
  //  q * Exp(ω*dt);
  //  v + a*dt];

  return {
      .p = p + v * dt,
      .q = q * rotation::AngleAxisToQuaternion(delta_angle),
      .v = v + delta_velocity,
      .accel_bias = accel_bias,
      .gyro_bias = gyro_bias,
  };
}

template <typename T>
struct Jacobians {
  Eigen::Matrix<T, 15, 15> fjac;
  Eigen::Matrix<T, 15, 15> qcov;
};

template <typename Scalar>
Jacobians<Scalar> MotionJacobians(const NominalState<Scalar>& state,
                                  const Input<Scalar>& input, Scalar dt,
                                  const Config<Scalar>& cfg = {}) {
  const auto& [p, q, v, accel_bias, gyro_bias] = state;
  const auto& [accel, gyro] = input;
  const Eigen::Vector3<Scalar> acc_unbiased = accel - accel_bias;
  const Eigen::Vector3<Scalar> gyro_unbiased = gyro - gyro_bias;
  const Eigen::Vector3<Scalar> delta_angle = gyro_unbiased * dt;

  Eigen::Matrix<Scalar, 15, 15> fjac = Eigen::Matrix<Scalar, 15, 15>::Zero();

  const auto seq3 = [](Eigen::Index start) {
    return Eigen::seqN(start, Eigen::fix<3>());
  };

  // F =
  //   [I, O,            dt * I, O,     O    ;
  //    O, R(-dt*ω),     O,      O,     -dt*I;
  //    O, -dt*R*hat(a), I,      -dt*R, O    ;
  //    O, O,            O,      I,     O    ;
  //    O, O,            O,      O,     I    ];
  fjac(seq3(0), seq3(0)).setIdentity();
  fjac(seq3(0), seq3(6)) = dt * Eigen::Matrix3<Scalar>::Identity();

  // Rotation
  fjac(seq3(3), seq3(3)) = rotation::AngleAxisToRotationMatrix(-delta_angle);
  fjac(seq3(3), seq3(12)) = -dt * Eigen::Matrix3<Scalar>::Identity();

  const Eigen::Matrix3<Scalar> rmat = q.toRotationMatrix();
  // Velocity
  fjac(seq3(6), seq3(3)) = -dt * rmat * rotation::hat(acc_unbiased);
  fjac(seq3(6), seq3(6)).setIdentity();
  fjac(seq3(6), seq3(9)) = -dt * rmat;

  // Accel bias
  fjac(seq3(9), seq3(9)).setIdentity();

  // Gyro bias
  fjac(seq3(12), seq3(12)).setIdentity();

  // Process Noise Q (Discrete approx)
  Eigen::Matrix<Scalar, 15, 15> qcov = Eigen::Matrix<Scalar, 15, 15>::Zero();

  // G = blkdiag(O, ...
  //             σ_gn * dt.^2 * I, ...
  //             σ_an * dt.^2 * I, ...
  //             σ_ab * dt * I,    ...
  //             σ_gb * dt * I);

  const Scalar dt_sq = dt * dt;
  const auto [accel_noise_density, gyro_noise_density, accel_bias_random_walk,
              gyro_bias_random_walk, _] = cfg;
  qcov(seq3(3), seq3(3)).diagonal().setConstant(dt_sq * gyro_noise_density);
  qcov(seq3(6), seq3(6)).diagonal().setConstant(dt_sq * accel_noise_density);
  qcov(seq3(9), seq3(9)).diagonal().setConstant(dt * accel_bias_random_walk);
  qcov(seq3(12), seq3(12)).diagonal().setConstant(dt * gyro_bias_random_walk);

  return {.fjac = fjac, .qcov = qcov};
}

template <typename T>
struct Pose {
  Eigen::Vector3<T> p;
  Eigen::Quaternion<T> q;
};

template <typename Scalar>
Pose<Scalar> PoseObservation(const NominalState<Scalar>& state) {
  return {.p = state.p, .q = state.q};
}

template <typename Scalar>
Eigen::Matrix<Scalar, 6, 15> PoseObservationJacobian(
    const NominalState<Scalar>& state) {
  Eigen::Matrix<Scalar, 6, 15> jacobian = Eigen::Matrix<Scalar, 6, 15>::Zero();
  jacobian.template leftCols<6>().setIdentity();
  return jacobian;
}

template <typename T>
struct CompassVector {
  Eigen::Vector3<T> b;
};

template <typename Scalar>
CompassVector<Scalar> CompassObservation(
    const NominalState<Scalar>& state,
    const Eigen::Vector3<Scalar>& b_inertial) {
  return {.b = state.q.inverse() * b_inertial};
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 15> CompassObservationJacobian(
    const NominalState<Scalar>& state,
    const Eigen::Vector3<Scalar>& mag_inertial) {
  Eigen::Matrix<Scalar, 3, 15> jacobian = Eigen::Matrix<Scalar, 3, 15>::Zero();
  jacobian(Eigen::all, Eigen::seqN(3, Eigen::fix<3>)) =
      rotation::hat(state.q.inverse() * mag_inertial);
  return jacobian;
}
}  // namespace eskf

#endif  // ESKF_MATH_ESKF_MATH_HPP_
