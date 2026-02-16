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

  if (IsClose(theta_sq, Scalar(0))) {
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

  if (IsClose(theta_sq, Scalar(0))) {
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
  Eigen::Vector3<T> omega;
};

template <typename PDerived, typename QDerived, typename VDerived,
          typename ADerived, typename GDerived, typename GravDerived,
          typename Scalar = typename PDerived::Scalar>
NominalState<typename PDerived::Scalar> kinematics(
    const Eigen::MatrixBase<PDerived>& p,
    const Eigen::QuaternionBase<QDerived>& q,
    const Eigen::MatrixBase<VDerived>& v,
    const Eigen::MatrixBase<ADerived>& acc_unbiased,
    const Eigen::MatrixBase<GDerived>& gyr_unbiased, Scalar dt,
    const Eigen::MatrixBase<GravDerived>& grav_vector =
        Eigen::Vector3<Scalar>(0, 0, -9.81)) {
  const Eigen::Vector3d acc_world = q * acc_unbiased + grav_vector;

  const Eigen::Vector3d delta_velocity = acc_world * dt;
  const Eigen::Vector3d delta_angle = gyr_unbiased * dt;

  // f(x, u) =
  // [p + v*dt + a * dt^2 / 2;
  //  q * Exp(ω*dt);
  //  v + a*dt];

  return {
      .p = p + v * dt + 0.5 * delta_velocity * dt,
      .q = q * rotation::AngleAxisToQuaternion(delta_angle),
      .v = v + delta_velocity,
      .omega = gyr_unbiased,  // Pass through gyro reading for bias estimation
  };
}

template <typename T>
struct Jacobians {
  Eigen::Matrix<T, 15, 15> fjac;
  Eigen::Matrix<T, 15, 15> qcov;
};

template <typename T>
struct NoiseConfig {
  T accel_noise_density;
  T gyro_noise_density;
  T accel_bias_random_walk;
  T gyro_bias_random_walk;
};

template <typename PDerived, typename QDerived, typename VDerived,
          typename ADerived, typename GDerived, typename Cfg,
          typename GravDerived, typename Scalar = typename PDerived::Scalar>
Jacobians<typename PDerived::Scalar> ComputeJacobians(
    const Eigen::MatrixBase<PDerived>& /*p*/,
    const Eigen::QuaternionBase<QDerived>& q,
    const Eigen::MatrixBase<VDerived>& /*v*/,
    const Eigen::MatrixBase<ADerived>& acc_unbiased,
    const Eigen::MatrixBase<GDerived>& gyr_unbiased, Scalar dt,
    const Cfg& config,
    const Eigen::MatrixBase<GravDerived>& grav_vector =
        Eigen::Vector3<Scalar>(0, 0, -9.81)) {
  using Eigen::fix;
  using Eigen::seqN;
  const Eigen::Vector3d acc_world = q * acc_unbiased + grav_vector;

  const Eigen::Vector3d delta_velocity = acc_world * dt;
  const Eigen::Vector3d delta_angle = gyr_unbiased * dt;

  Eigen::Matrix<Scalar, 15, 15> fjac = Eigen::Matrix<Scalar, 15, 15>::Zero();

  const auto seq3 = [](Eigen::Index start) { return seqN(start, fix<3>()); };

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
  fjac(seq3(6), seq3(3)) = -dt * rmat * hat(acc_unbiased);
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

  qcov(seq3(6), seq3(6))
      .diagonal()
      .setConstant(dt * dt * config.accel_noise_density);
  qcov(seq3(3), seq3(3))
      .diagonal()
      .setConstant(dt * dt * config.gyro_noise_density);
  qcov(seq3(9), seq3(9))
      .diagonal()
      .setConstant(dt * config.accel_bias_random_walk);
  qcov(seq3(12), seq3(12))
      .diagonal()
      .setConstant(dt * config.gyro_bias_random_walk);

  return {.fjac = fjac, .qcov = qcov};
}

}  // namespace eskf

#endif  // ESKF_MATH_ESKF_MATH_HPP_
