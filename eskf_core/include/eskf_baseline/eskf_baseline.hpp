#ifndef ESKF_MATH_ESKF_MATH_HPP_
#define ESKF_MATH_ESKF_MATH_HPP_

#include "Eigen/Dense"
#include "eskf_baseline/math.hpp"

namespace eskf {

template <typename T>
struct NominalState {
  static constexpr Eigen::Index kTangentDim = 18;
  Eigen::Vector3<T> p;
  Eigen::Quaternion<T> q;
  Eigen::Vector3<T> v;
  Eigen::Vector3<T> accel_bias = Eigen::Vector3<T>::Zero();
  Eigen::Vector3<T> gyro_bias = Eigen::Vector3<T>::Zero();
  Eigen::Vector3<T> grav_vector{T(0), T(0), T(-9.81)};

  NominalState<T> boxplus(const Eigen::Vector<T, kTangentDim>& delta) const {
    return {
        .p = p + delta.template head<3>(),
        .q = q * rotation::AngleAxisToQuaternion(delta.template segment<3>(3)),
        .v = v + delta.template segment<3>(6),
        .accel_bias = accel_bias + delta.template segment<3>(9),
        .gyro_bias = gyro_bias + delta.template segment<3>(12),
        .grav_vector = grav_vector + delta.template segment<3>(15),
    };
  }
};

template <typename T>
struct ImuInput {
  Eigen::Vector3<T> accel;
  Eigen::Vector3<T> gyro;
};

template <typename T>
struct Config {
  T accel_noise_density{0.005};
  T gyro_noise_density{5e-5};
  T accel_bias_random_walk{0.001};
  T gyro_bias_random_walk{0.0001};
};

template <typename Scalar>
NominalState<Scalar> Motion(const NominalState<Scalar>& state,
                            const ImuInput<Scalar>& input, Scalar dt,
                            const Config<Scalar>& cfg = {}) {
  const auto& [p, q, v, accel_bias, gyro_bias, grav_vector] = state;
  const auto& [accel, gyro] = input;
  const Eigen::Vector3<Scalar> acc_unbiased = accel - accel_bias;
  const Eigen::Vector3<Scalar> accel_world = q * acc_unbiased + grav_vector;
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
  Eigen::Matrix<T, NominalState<T>::kTangentDim, NominalState<T>::kTangentDim>
      fjac;
  Eigen::Matrix<T, NominalState<T>::kTangentDim, NominalState<T>::kTangentDim>
      qcov;
};

auto seq3(auto start) { return Eigen::seqN(start, Eigen::fix<3>()); };

template <typename Scalar>
Jacobians<Scalar> MotionJacobians(const NominalState<Scalar>& state,
                                  const ImuInput<Scalar>& input, Scalar dt,
                                  const Config<Scalar>& cfg = {}) {
  static constexpr auto kTangentDim = NominalState<Scalar>::kTangentDim;

  const auto& [p, q, v, accel_bias, gyro_bias, _] = state;
  const auto& [accel, gyro] = input;
  const Eigen::Vector3<Scalar> acc_unbiased = accel - accel_bias;
  const Eigen::Vector3<Scalar> gyro_unbiased = gyro - gyro_bias;
  const Eigen::Vector3<Scalar> delta_angle = gyro_unbiased * dt;

  Eigen::Matrix<Scalar, kTangentDim, kTangentDim> fjac =
      Eigen::Matrix<Scalar, kTangentDim, kTangentDim>::Zero();

  // F =
  //   [I, O,            dt * I, O,     O    , O   ;
  //    O, R(-dt*ω),     O,      O,     -dt*I, O   ;
  //    O, -dt*R*hat(a), I,      -dt*R, O    , dt*I;
  //    O, O,            O,      I,     O    , O   ;
  //    O, O,            O,      O,     I    , O   ;
  //    O, O,            O,      O,     O    , I   ];
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
  fjac(seq3(6), seq3(15)) = dt * Eigen::Matrix3<Scalar>::Identity();

  // Accel bias
  fjac(seq3(9), seq3(9)).setIdentity();

  // Gyro bias
  fjac(seq3(12), seq3(12)).setIdentity();
  fjac(seq3(15), seq3(15)).setIdentity();

  // Process Noise Q (Discrete approx)
  Eigen::Matrix<Scalar, kTangentDim, kTangentDim> qcov =
      Eigen::Matrix<Scalar, kTangentDim, kTangentDim>::Zero();

  // G = blkdiag(O, ...
  //             σ_gn * dt.^2 * I, ...
  //             σ_an * dt.^2 * I, ...
  //             σ_ab * dt * I,    ...
  //             σ_gb * dt * I,    ...
  //             O);

  const Scalar dt_sq = dt * dt;
  const auto [accel_noise_density, gyro_noise_density, accel_bias_random_walk,
              gyro_bias_random_walk] = cfg;
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

  Eigen::Vector<T, 6> boxminus(const Pose<T>& other) const {
    Eigen::Vector<T, 6> res;
    res(seq3(0)) = p - other.p,
    res(seq3(3)) = rotation::QuaternionToAngleAxis(other.q.inverse() * q);
    return res;
  }
};

template <typename Scalar>
Pose<Scalar> PoseObservation(const NominalState<Scalar>& state) {
  return {.p = state.p, .q = state.q};
}

template <typename Scalar>
Eigen::Matrix<Scalar, 6, NominalState<Scalar>::kTangentDim>
PoseObservationJacobian(const NominalState<Scalar>& state) {
  static constexpr auto kTangentDim = NominalState<Scalar>::kTangentDim;

  Eigen::Matrix<Scalar, 6, kTangentDim> jacobian =
      Eigen::Matrix<Scalar, 6, kTangentDim>::Zero();
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
Eigen::Matrix<Scalar, 3, NominalState<Scalar>::kTangentDim>
CompassObservationJacobian(const NominalState<Scalar>& state,
                           const Eigen::Vector3<Scalar>& mag_inertial) {
  static constexpr auto kTangentDim = NominalState<Scalar>::kTangentDim;

  Eigen::Matrix<Scalar, 3, kTangentDim> jacobian =
      Eigen::Matrix<Scalar, 3, kTangentDim>::Zero();
  jacobian(Eigen::all, Eigen::seqN(3, Eigen::fix<3>)) =
      rotation::hat(state.q.inverse() * mag_inertial);
  return jacobian;
}
}  // namespace eskf

#endif  // ESKF_MATH_ESKF_MATH_HPP_
