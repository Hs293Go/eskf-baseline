#ifndef LOAM_BASELINE_IRWGN_HPP_
#define LOAM_BASELINE_IRWGN_HPP_

#include <vector>

#include "loam_baseline/observation_models.hpp"
#include "manifold_baseline/manifold_baseline.hpp"

namespace loam_baseline {

template <typename T>
struct EdgeContext {
  /// Point in the local scan frame
  Eigen::Vector3<T> p_local;

  /// Point A defining the edge line in the map frame
  Eigen::Vector3<T> pa_map;

  /// Point B defining the edge line in the map frame
  Eigen::Vector3<T> pb_map;

  /// Optional scaling factor for the residual, e.g. to implement a robust loss.
  T residual_coeff = 1.0;
};

template <typename T>
struct PlaneContext {
  /// Point in the local scan frame
  Eigen::Vector3<T> p_local;

  /// Unit normal of the plane in the map frame
  Eigen::Vector3<T> n_map;

  /// Scalar offset of the plane: −(n_map · A) where A is any point on the plane
  T d_map;

  /// Optional scaling factor for the residual, e.g. to implement a robust loss.
  T residual_coeff = 1.0;
};

/// IRWGN: Iteratively Reweighted Gauss-Newton for lidar scan-to-map matching
/// with point-to-edge and point-to-plane residuals.
template <typename T>
struct IRWGN {
  std::vector<EdgeContext<T>> edges;
  std::vector<PlaneContext<T>> planes;
  T c_edge;
  T c_plane;

  struct IterateResult {
    manifold::Transform<T> new_pose;
    Eigen::Matrix<T, 6, 6> hessian;
    std::size_t edge_num;
    std::size_t plane_num;
    bool converged;
  };

  IterateResult iterateOnce(const manifold::Transform<T>& pose) const {
    using std::abs;
    Eigen::Matrix<T, 6, 6> hess = Eigen::Matrix<T, 6, 6>::Zero();
    Eigen::Vector<T, 6> grad = Eigen::Vector<T, 6>::Zero();

    // TODO: Think about whether the two loops can run concurrently to produce
    // two (hess, grad) pairs to be added together at the end.

    // TODO: parallelize this loop with TBB or OpenMP
    std::size_t edge_num = 0;
    for (const auto& edge : edges) {
      const PointToEdgeFactor<T> fac{.pa_map = edge.pa_map,
                                     .pb_map = edge.pb_map};
      const auto& [resid, hjac] = fac(pose, edge.p_local);
      const auto w = edge.residual_coeff * TukeyWeight(resid.norm(), c_edge);
      if (w > 0.0) {
        hess += w * hjac.transpose() * hjac;
        grad += w * hjac.transpose() * resid;
        ++edge_num;
      }
    }

    // TODO: parallelize this loop with TBB or OpenMP
    std::size_t surf_num = 0;
    for (const auto& plane : planes) {
      const PointToPlaneFactor<T> fac{.n_map = plane.n_map,
                                      .d_map = plane.d_map};
      const auto& [resid, hjac] = fac(pose, plane.p_local);
      const auto w = plane.residual_coeff * TukeyWeight(abs(resid), c_plane);
      if (w > 0.0) {
        hess += w * hjac.transpose() * hjac;
        grad += w * hjac.transpose() * resid;
        ++surf_num;
      }
    }

    constexpr T kLambda = 1e-5;
    Eigen::Vector<T, 6> dx =
        -(hess + kLambda * Eigen::Matrix<T, 6, 6>::Identity())
             .ldlt()
             .solve(grad);

    auto new_pose =
        manifold::TransformProduct(manifold::ScrewToTransform(dx), pose);

    return {.new_pose = new_pose,
            .hessian = hess,
            .edge_num = edge_num,
            .plane_num = surf_num,
            .converged = dx.norm() < 1e-6};
  }
};

}  // namespace loam_baseline

#endif  // LOAM_BASELINE_IRWGN_HPP_
