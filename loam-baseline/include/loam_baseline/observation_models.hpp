#ifndef LOAM_BASELINE_OBSERVATION_MODELS_HPP_
#define LOAM_BASELINE_OBSERVATION_MODELS_HPP_

#include "manifold_baseline/manifold_baseline.hpp"

namespace loam_baseline {

template <typename T, Eigen::Index D, Eigen::Index M>
struct ResidualAndJacobian {
  using Residual = std::conditional_t<(D == 1), T, Eigen::Vector<T, D>>;
  Residual residual;
  Eigen::Matrix<T, D, M> jacobian;
};

template <typename T>
struct PointToEdgeFactor {
  Eigen::Vector3<T> pa_map;
  Eigen::Vector3<T> pb_map;

  template <typename Derived>
  ResidualAndJacobian<T, 3, 6> operator()(
      const manifold::Transform<T>& local_to_map,
      const Eigen::MatrixBase<Derived>& p_local) const {
    const auto p_map = manifold::TransformPoint(local_to_map, p_local);
    // Cross-product distance: nu = (lp - pa) × (lp - pb), de = pa - pb
    const auto nu = (p_map - pa_map).cross(p_map - pb_map);  // WARN: use once
    const Eigen::Vector3<T> dp_a2b = pb_map - pa_map;

    ResidualAndJacobian<T, 3, 6> res;
    auto jac = manifold::JacTransformedPointByTransform(local_to_map, p_local);
    const T dp_nrm = dp_a2b.norm();
    res.residual = nu / dp_nrm;
    res.jacobian = manifold::hat(dp_a2b) * jac / dp_nrm;
    return res;
  }
};

template <typename T>
struct PointToPlaneFactor {
  Eigen::Vector3<T> n_map;
  T d_map;

  template <typename Derived>
  ResidualAndJacobian<T, 1, 6> operator()(
      manifold::Transform<T> local_to_map,
      const Eigen::MatrixBase<Derived>& p_local) const {
    const auto p_map = manifold::TransformPoint(local_to_map, p_local);

    ResidualAndJacobian<T, 1, 6> res;
    auto jac = manifold::JacTransformedPointByTransform(local_to_map, p_local);
    res.residual = n_map.dot(p_map) + d_map;
    res.jacobian = n_map.transpose() * jac;
    return res;
  }
};

template <typename T>
T TukeyWeight(T r, T c) {
  using std::abs;
  if (abs(r) >= c) {
    return T(0);
  }
  const T x = r / c;
  const T tmp = T(1) - x * x;
  return tmp * tmp;
}

}  // namespace loam_baseline

#endif  // LOAM_BASELINE_OBSERVATION_MODELS_HPP_
