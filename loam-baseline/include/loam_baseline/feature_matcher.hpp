#ifndef LOAM_BASELINE_FEATURE_MATCHER_HPP_
#define LOAM_BASELINE_FEATURE_MATCHER_HPP_

#include <cmath>
#include <span>

#include "Eigen/Dense"
#include "loam_baseline/irwgn.hpp"

namespace loam_baseline {

template <typename T, Eigen::Index C>
struct PCAResult {
  using MatrixType = Eigen::Matrix<double, C, C>;
  Eigen::SelfAdjointEigenSolver<MatrixType> eig;
  Eigen::Vector<double, C> mean;
};

template <std::ranges::input_range R,
          typename Derived = std::ranges::range_value_t<R>,
          typename T = typename Derived::Scalar,
          Eigen::Index C = Derived::RowsAtCompileTime>
  requires(std::derived_from<Derived, Eigen::MatrixBase<Derived>>)
PCAResult<T, C> ComputePCA(R&& data) {
  Eigen::Vector<double, C> mean = Eigen::Vector<double, C>::Zero();
  for (const auto& vec : data) {
    mean += vec;
  }
  mean /= static_cast<double>(data.size());

  Eigen::Matrix<double, C, C> var_cov = Eigen::Matrix<double, C, C>::Zero();
  for (const auto& vec : data) {
    var_cov.template selfadjointView<Eigen::Lower>().rankUpdate(vec - mean);
  }
  return {Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, C, C>>(
              var_cov.template selfadjointView<Eigen::Lower>()),
          mean};
}

// Returns count filled in [0, capacity]; 0 = search miss or insufficient
// points. Functor does NOT allocate; it fills up to capacity entries inplace.
template <typename F, typename T>
concept EdgeSearchFn =
    requires(F f, const Eigen::Vector3<T>& q, std::span<Eigen::Vector3<T>> pts,
             std::span<T> dists, T max_dist) {
      { f(q, pts, dists, max_dist) } -> std::convertible_to<int>;
    };

template <typename F, typename T>
concept PlaneSearchFn =
    requires(F f, const Eigen::Vector3<T>& q, std::span<Eigen::Vector3<T>> pts,
             std::span<T> dists) {
      { f(q, pts, dists) } -> std::convertible_to<int>;
    };

enum class MatchResult : uint8_t {
  kSuccess = 0,
  kNotEnoughNeighbors = 1,
  kNeighborsTooFar = 2,
  kBadPcaStructure = 3,
  kInvalidNumerical = 4,
  kMseTooLarge = 5,
  kUnknown = 6,
  kNRejectionCauses = 7
};

template <typename T>
struct FeatureMatcher {
  struct MatchedEdges {
    MatchResult result;
    EdgeContext<T> feature;
  };

  /// Matches a local point to an edge in the map.
  ///
  /// # Reference
  /// This is ported from `LidarSLAM::ComputeLineDistanceParameters`
  template <typename Derived, EdgeSearchFn<typename Derived::Scalar> EdgeSearch>
  MatchedEdges matchEdge(const Eigen::MatrixBase<Derived>& p_local,
                         const manifold::Transform<T>& lidar_to_world,
                         EdgeSearch&& search) const {
    const Eigen::Vector3<T>& p_init = p_local;
    const Eigen::Vector3<T> p_final =
        manifold::TransformPoint(lidar_to_world, p_init);

    constexpr std::size_t kEdgeK = 10;
    alignas(16) std::array<Eigen::Vector3<T>, kEdgeK> nearest_pts;
    std::array<T, kEdgeK> nearest_dist;

    const T square_max_dist = 3.0 * line_res;

    std::size_t found = search(p_final, std::span(nearest_pts),
                               std::span(nearest_dist), max_dist_inlier);

    if (found < min_line_neighbor_rejection) {
      return {.result = MatchResult::kNotEnoughNeighbors};
    }
    std::span found_nearest(nearest_pts.begin(), found);

    if (nearest_dist[found - 1] > square_max_dist) {
      return {.result = MatchResult::kNeighborsTooFar};
    }

    auto [eig, mean] = ComputePCA(found_nearest);

    const Eigen::Vector3<T> d = eig.eigenvalues();
    if (d[2] < static_cast<T>(found_nearest.size()) * d[1]) {
      return {.result = MatchResult::kBadPcaStructure};
    }

    const Eigen::Vector3<T> n = eig.eigenvectors()(Eigen::all, 2);
    const Eigen::Matrix3<T> amat =
        Eigen::Matrix3<T>::Identity() - n * n.transpose();

    if (!std::isfinite(amat(0, 0))) {
      return {.result = MatchResult::kInvalidNumerical};
    }

    T mean_sq_dst(0);

    for (const auto& pt : found_nearest) {
      const Eigen::Vector3<T> pt2mean = pt - mean;
      const T sq = pt2mean.transpose() * amat * pt2mean;
      if (sq > square_max_dist) {
        return {.result = MatchResult::kMseTooLarge};
      }
      mean_sq_dst += sq;
    }
    mean_sq_dst /= static_cast<T>(found_nearest.size());

    const T fit_quality = T(1) - std::sqrt(mean_sq_dst / square_max_dist);
    return {.result = MatchResult::kSuccess,
            .feature = {.p_local = p_local,
                        .pa_map = T(0.1) * n + mean,
                        .pb_map = T(-0.1) * n + mean,
                        .residual_coeff = fit_quality}};
  }

  struct MatchedPlanes {
    MatchResult result;
    PlaneContext<T> feature;
  };

  /// Matches a local point to a plane in the map.
  ///
  /// # Reference
  /// This is ported from `LidarSLAM::ComputePlaneDistanceParameters`
  template <typename Derived,
            PlaneSearchFn<typename Derived::Scalar> PlaneSearch>
  MatchedPlanes matchPlane(const Eigen::MatrixBase<Derived>& p_local,
                           const manifold::Transform<T>& lidar_to_world,
                           PlaneSearch&& search) const {
    const Eigen::Vector3<T>& p_init = p_local;
    const Eigen::Vector3<T> p_final =
        manifold::TransformPoint(lidar_to_world, p_init);

    const T square_max_dist = 3.0 * plane_res;
    constexpr std::size_t kPlaneK = 5;
    alignas(16) std::array<Eigen::Vector3<T>, kPlaneK> nearest_pts;
    std::array<T, kPlaneK> nearest_dist;

    std::size_t found =
        search(p_final, std::span(nearest_pts), std::span(nearest_dist));
    if (found < kPlaneK) {
      return {.result = MatchResult::kNotEnoughNeighbors};
    }

    if (nearest_dist[kPlaneK - 1] > square_max_dist) {
      return {.result = MatchResult::kNeighborsTooFar};
    }

    std::span found_nearest(nearest_pts);

    Eigen::Matrix<T, kPlaneK, 3> mat_a0;
    Eigen::Vector<T, kPlaneK> mat_b0 = -Eigen::Vector<T, kPlaneK>::Ones();

    for (std::size_t j = 0; j < kPlaneK; ++j) {
      mat_a0(j, Eigen::all) = found_nearest[j].transpose();
    }

    Eigen::Vector3<T> norm = mat_a0.colPivHouseholderQr().solve(mat_b0);
    const T negative_oa_dot_norm = T(1) / norm.norm();
    norm *= negative_oa_dot_norm;

    T mean_sq_dst(0);
    for (std::size_t j = 0; j < kPlaneK; ++j) {
      const T dis = std::abs(norm.dot(found_nearest[j]) + negative_oa_dot_norm);
      if (dis > plane_res / T(2)) {
        return {.result = MatchResult::kMseTooLarge};
      }
      mean_sq_dst += dis;
    }
    mean_sq_dst /= static_cast<T>(kPlaneK);
    const T fit_quality = T(1) - std::sqrt(mean_sq_dst / square_max_dist);

    return {.result = MatchResult::kSuccess,
            .feature = PlaneContext<T>{.p_local = p_local,
                                       .n_map = norm,
                                       .d_map = negative_oa_dot_norm,
                                       .residual_coeff = fit_quality}};
  }

  T line_res;
  std::size_t min_line_neighbor_rejection;
  T max_dist_inlier;

  T plane_res;
};

}  // namespace loam_baseline

#endif  // LOAM_BASELINE_FEATURE_MATCHER_HPP_
