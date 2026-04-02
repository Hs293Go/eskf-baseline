#ifndef MANIFOLD_BASELINE_MANIFOLD_BASELINE_HPP_
#define MANIFOLD_BASELINE_MANIFOLD_BASELINE_HPP_

#include "Eigen/Dense"

namespace ix {
using Eigen::all;

auto seq3(auto start) { return Eigen::seqN(start, Eigen::fix<3>); }
}  // namespace ix

namespace manifold {
using ix::seq3;

/// SO(3) hat operator, mapping a 3D vector to a 3×3 skew-symmetric matrix.
/// Said matrix, when multiplied by another vector, computes the cross product
/// of the original with the other vector.
///
/// ``` c++
/// hat(a) * b == a.cross(b)
/// ```
template <typename Derived>
  requires(bool(Derived::IsVectorAtCompileTime) &&
           Derived::SizeAtCompileTime == 3)
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
Eigen::Vector3<typename Derived::Scalar> vee(
    const Eigen::MatrixBase<Derived>& v_hat) {
  return {v_hat(2, 1), v_hat(0, 2), v_hat(1, 0)};
}

/// Convert an angle-axis representation `angle_axis` to a quaternion
///
/// The input `angle_axis` is a 3D vector whose direction is the rotation axis
/// `a` and whose magnitude is the rotation angle `theta` in radians. This
/// operation is equivalent to:
/// - Constructing a unit quaternion from a rotation angle and axis by its
///   definition [sin(angle/2) * axs; cos(angle/2)]
/// - Applying the quaternionic exponential map to a pure imaginary quaternion
/// [angle * axis / 2; 0]
///
/// # Argument
/// - `angle_axis`: a 3-vector representing a rotation vector/scaled rotation
/// axis
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

template <typename Derived>
Eigen::Vector<typename Derived::Scalar, 3> QuaternionToAngleAxis(
    const Eigen::QuaternionBase<Derived>& quaternion) {
  using T = typename Derived::Scalar;

  using std::atan2;
  using std::sqrt;
  const T squared_n = quaternion.vec().squaredNorm();
  const T& w = quaternion.w();

  T two_atan_nbyw_by_n;

  if (squared_n < T(1e-6)) {
    const T squared_w = w * w;
    two_atan_nbyw_by_n =
        T(2.0) / w - T(2.0 / 3.0) * (squared_n) / (w * squared_w);
  } else {
    const T n = sqrt(squared_n);
    const T atan_nbyw = (w < T(0)) ? atan2(-n, -w) : atan2(n, w);
    two_atan_nbyw_by_n = T(2) * atan_nbyw / n;
  }

  return two_atan_nbyw_by_n * quaternion.vec();
}

template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> RotationMatrixToAngleAxis(
    const Eigen::MatrixBase<Derived>& rotation_matrix) {
  return QuaternionToAngleAxis(
      Eigen::Quaternion<typename Derived::Scalar>(rotation_matrix));
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> SO3LeftJacobian(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  using Scalar = typename Derived::Scalar;

  const Scalar theta_sq = angle_axis.squaredNorm();
  const Eigen::Matrix3<Scalar> hat_phi = hat(angle_axis);

  if (theta_sq < Scalar(1e-6)) {
    return Eigen::Matrix3<Scalar>::Identity() + Scalar(0.5) * hat_phi +
           Scalar(1.0 / 6.0) * hat_phi * hat_phi;
  }
  const Scalar theta = sqrt(theta_sq);
  const Scalar cos_theta = cos(theta);
  const Scalar sin_theta = sin(theta);

  return Eigen::Matrix3<Scalar>::Identity() +
         (Scalar(1) - cos_theta) / theta_sq * hat_phi +
         (theta - sin_theta) / (theta_sq * theta) * hat_phi * hat_phi;
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> SO3LeftJacobianInverse(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  using Scalar = typename Derived::Scalar;

  const Scalar theta_sq = angle_axis.squaredNorm();
  const Eigen::Matrix3<Scalar> hat_phi = hat(angle_axis);
  const Eigen::Matrix3<Scalar> hat_phi_sq = hat_phi * hat_phi;

  if (theta_sq < Scalar(1e-6)) {
    return Eigen::Matrix3<Scalar>::Identity() - Scalar(0.5) * hat_phi +
           Scalar(1.0 / 12.0) * hat_phi_sq;
  }
  const Scalar theta = sqrt(theta_sq);
  const Scalar half_theta = Scalar(0.5) * theta;
  const Scalar cot_half_theta = cos(half_theta) / sin(half_theta);

  return Eigen::Matrix3<Scalar>::Identity() - Scalar(0.5) * hat_phi +
         (Scalar(1) - theta * cot_half_theta / Scalar(2)) / theta_sq *
             hat_phi_sq;
}

/// A rigid body transform compactly represented by a translation as a 3-vector
/// and an orientation as a quaternion.
///
/// This is plainly a data container and does not implement any manifold
/// operations as members.
template <typename T>
struct Transform {
  Eigen::Quaternion<T> rotation;
  Eigen::Matrix<T, 3, 1> translation;
};

using TransformF32 = Transform<float>;
using TransformF64 = Transform<double>;

/// Composes two `Transform` objects, representing performing one transformation
/// after the other.
template <typename T>
Transform<T> TransformProduct(const Transform<T>& a, const Transform<T>& b) {
  return {.rotation = a.rotation * b.rotation,
          .translation = a.rotation * b.translation + a.translation};
}

/// Computes a `Transform` object representing the inverse of the transformation
/// represented in the original.
template <typename T>
Transform<T> TransformInverse(const Transform<T>& t) {
  const Eigen::Quaternion<T> inv_rot = t.rotation.conjugate();
  return {.rotation = inv_rot, .translation = -(inv_rot * t.translation)};
}

/// Transforms a point through a rotatation, followed by a translation,
/// specified by the `Transform` object in the argument.
template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> TransformPoint(
    const Transform<typename Derived::Scalar>& t,
    const Eigen::MatrixBase<Derived>& p) {
  return t.rotation * p + t.translation;
}

/// Converts a `Transform` object into a 4x4 transformation matrix
template <typename Derived>
Eigen::Matrix4<typename Derived::Scalar> TransformToMatrix(
    const Transform<typename Derived::Scalar>& pose) {
  using Scalar = typename Derived::Scalar;
  Eigen::Matrix4<Scalar> tform = Eigen::Matrix4<Scalar>::Zero();
  tform(seq3(0), seq3(0)) = pose.rotation.toRotationMatrix();
  tform(seq3(0), 3) = pose.translation;
  tform(3, 3) = Scalar(1);
  return tform;
}

/// Converts a 4x4 transformation matrix into a `Transform` object
template <typename Derived>
Transform<typename Derived::Scalar> MatrixToTransform(
    const Eigen::MatrixBase<Derived>& tform) {
  using Scalar = typename Derived::Scalar;
  return {.rotation = Eigen::Quaternion<Scalar>(tform(seq3(0), seq3(0))),
          .translation = tform(seq3(0), 3)};
}

/// Converts a 6x1 screw vector with a leading translational part (n.b., warped
/// by SO(3) left Jacobian) and trailing rotational part into a transformation.
///
/// # Recipe
///
/// Integrating a pose by an Euler Step is accomplished by:
///
/// ``` c++
/// auto new_pose = TransformProduct(pose, ScrewToTransform(dt * twist));
/// ```
///
/// # Arguments
/// - `screw`: A 6-vector consisting of `[upsilon; omega]`, where `upsilon` is
///   the translation vector warped by the SO(3) left Jacobian corresponding to
///   `omega`
///
/// # Returns
/// - A `Transform` containing the quaternion corresponding to `omega` and the
///   unwarped translation vector
template <typename Derived>
Transform<typename Derived::Scalar> ScrewToTransform(
    const Eigen::MatrixBase<Derived>& screw) {
  using Scalar = typename Derived::Scalar;
  Eigen::Ref<const Eigen::Vector3<Scalar>> upsilon(screw(seq3(0)));
  Eigen::Ref<const Eigen::Vector3<Scalar>> omega(screw(seq3(3)));
  return {.rotation = AngleAxisToQuaternion(omega),
          .translation = SO3LeftJacobian(omega) * upsilon};
}

/// Converts a transformation into a 6x1 screw vector with a leading
/// translational part (n.b., warped by SO(3) left Jacobian) and trailing
/// rotational part
///
/// # Arguments
/// - `tform`: A `Transform` object
///
/// # Returns
/// - A 6-vector consisting of `[upsilon; omega]`, where `upsilon` is the
///   translation vector warped by the SO(3) left Jacobian corresponding to
///   `omega`

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 6, 1> TransformToScrew(
    const Transform<typename Derived::Scalar>& tform) {
  using Scalar = typename Derived::Scalar;
  const auto& [trans, rot] = tform;
  const Eigen::Vector3<Scalar> omega = QuaternionToAngleAxis(rot);
  Eigen::Vector<Scalar, 6> screw;
  screw << SO3LeftJacobianInverse(omega) * trans, omega;
  return screw;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 6, 6> PoseToAdjoint(const Transform<Scalar>& tform) {
  Eigen::Matrix<Scalar, 6, 6> adj = Eigen::Matrix<Scalar, 6, 6>::Zero();
  const auto& [trans, rot] = tform;
  auto rotmat = rot.toRotationMatrix();
  adj(seq3(0), seq3(0)) = rotmat;
  adj(seq3(0), seq3(3)) = hat(trans) * rotmat;
  adj(seq3(3), seq3(3)) = rotmat;
  return adj;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 6, 6> ScrewToAdjoint(
    const Eigen::MatrixBase<Derived>& screw) {
  using Scalar = typename Derived::Scalar;
  Eigen::Ref<const Eigen::Vector3<Scalar>> upsilon(screw(seq3(0)));
  Eigen::Ref<const Eigen::Vector3<Scalar>> omega(screw(seq3(3)));

  Eigen::Matrix<Scalar, 6, 6> adj = Eigen::Matrix<Scalar, 6, 6>::Zero();
  const Eigen::Matrix3<Scalar> hat_omega = hat(omega);
  adj(seq3(0), seq3(0)) = hat_omega;
  adj(seq3(3), seq3(0)) = hat(upsilon);
  adj(seq3(3), seq3(3)) = hat_omega;
  return adj;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 6> JacTransformedPointByTransform(
    const Transform<typename Derived::Scalar>& tform,
    const Eigen::MatrixBase<Derived>& p) {
  using Scalar = typename Derived::Scalar;
  Eigen::Matrix<Scalar, 3, 6> jacobian = Eigen::Matrix<Scalar, 3, 6>::Zero();
  jacobian(Eigen::all, seq3(0)).setIdentity();
  jacobian(Eigen::all, seq3(3)) = -tform.rotation.toRotationMatrix() * hat(p);
  return jacobian;
}

}  // namespace manifold

#endif  // MANIFOLD_BASELINE_MANIFOLD_BASELINE_HPP_
