#ifndef LOAM_BASELINE_LIDAR_SLAM_HPP_
#define LOAM_BASELINE_LIDAR_SLAM_HPP_

#include <span>

#include "loam_baseline/LocalMap.h"
#include "loam_baseline/feature_matcher.hpp"
#include "loam_baseline/irwgn.hpp"
#include "manifold_baseline/manifold_baseline.hpp"

namespace loam_baseline {

template <typename T>
constexpr T rad2deg(T rad) noexcept {
  return rad * static_cast<T>(180) / std::numbers::pi_v<T>;
}

// What we do not implement:
// - `LidarSLAM::MannualYawCorrection`: We have a downstream estimator
// - `LidarSLAM::FeatureObservabilityAnalysis`: Feature matching quality is
//   better encapsulated in the Hessian
// - `LidarSLAM::DegeneracyDetection`: Not used in the original code;
//   potentially very interesting to Hs293Go

struct RegistrationError {
  Eigen::Vector3<double> position_error = Eigen::Vector3<double>::Zero();
  double position_inv_cond = 1.0;
  Eigen::Vector3<double> orientation_error = Eigen::Vector3<double>::Zero();
  double orientation_inv_cond = 1.0;
  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
};

struct LocalizationCfg {
  double velocity_failure_threshold = 5.0;
  int max_surface_features = 2000;
};

struct IterationStats {
  std::size_t num_surf_from_scan = 0;
  std::size_t num_corner_from_scan = 0;
};

struct OptimizationStats {
  // These are populated from LocalMap::get5x5LocalMapFeatureSize(), which
  // returns int
  int laser_cloud_corner_from_map_num = 0;
  int laser_cloud_surf_from_map_num = 0;
  // These are populated from size(), use std::size_t
  std::size_t laser_cloud_corner_stack_num = 0;
  std::size_t laser_cloud_surf_stack_num = 0;

  // TODO: These seem to be dependent on histogram
  // int plane_match_success = 0;
  // int plane_no_enough_neighbor = 0;
  // int plane_neighbor_too_far = 0;
  // int plane_badpca_structure = 0;
  // int plane_invalid_numerical = 0;
  // int plane_mse_too_large = 0;
  // int plane_unknown = 0;

  // TODO: Figure out how to formalize time profiling
  double time_elapsed = 0.0;

  double total_translation = 0.0;
  double total_rotation = 0.0;
  double translation_from_last = 0.0;
  double rotation_from_last = 0.0;

  std::vector<IterationStats> iterations;
};

struct Localization {
  struct SolveResult {
    manifold::TransformF64 pose;
    RegistrationError reg_err;
    OptimizationStats stats;
    bool accept_result;
  };

  Localization(const LocalizationCfg& cfg);

  void initialize(const manifold::TransformF64& start_pose,
                  std::span<const Eigen::Vector3d> edge_points_local,
                  std::span<const Eigen::Vector3d> plane_points_local);

  SolveResult solve(const manifold::TransformF64& initial_guess,
                    std::span<const Eigen::Vector3d> edge_points,
                    std::span<const Eigen::Vector3d> plane_points,
                    int max_iterations, double dt,
                    const manifold::TransformF64& prev_solution);

  /// Transforms scan points into the world frame and inserts them into the
  /// local map. Call once per frame, after solve(), using the final pose.
  /// Points are only added when accept_result is true; shiftMap is always
  /// called (already done inside solve()).
  void updateMap(bool accept_result, const manifold::TransformF64& final_pose,
                 std::span<const Eigen::Vector3d> edge_points_local,
                 std::span<const Eigen::Vector3d> plane_points_local);

  LocalMap local_map;
  FeatureMatcher<double> matcher;
  IRWGN<double> solver;
  LocalizationCfg cfg;

 private:
  // Adapter: LocalMap edge search → EdgeSearchFn contract.
  // Converts pcl::PointXYZI ↔ Eigen::Vector3d at the boundary.
  // Internal LocalMap vector allocations are a known transitional cost.
  int searchEdge(const Eigen::Vector3d& q, std::span<Eigen::Vector3d> pts_out,
                 std::span<double> dists_out, double max_dist) const {
    pcl::PointXYZI query;
    query.x = static_cast<float>(q.x());
    query.y = static_cast<float>(q.y());
    query.z = static_cast<float>(q.z());

    std::vector<pcl::PointXYZI> tmp_pts;
    std::vector<float> tmp_dists;
    if (!local_map.nearestKSearchSpecificEdgePoint(
            query, tmp_pts, tmp_dists, static_cast<int>(pts_out.size()),
            static_cast<float>(max_dist))) {
      return 0;
    }

    const int n = static_cast<int>(std::min(tmp_pts.size(), pts_out.size()));
    for (int i = 0; i < n; ++i) {
      pts_out[i] = tmp_pts[i].getVector3fMap().template cast<double>();
      dists_out[i] = static_cast<double>(tmp_dists[i]);
    }
    return n;
  }

  // Adapter: LocalMap surf search → PlaneSearchFn contract.
  int searchPlane(const Eigen::Vector3d& q, std::span<Eigen::Vector3d> pts_out,
                  std::span<double> dists_out) const {
    pcl::PointXYZI query;
    query.x = static_cast<float>(q.x());
    query.y = static_cast<float>(q.y());
    query.z = static_cast<float>(q.z());

    std::vector<pcl::PointXYZI> tmp_pts;
    std::vector<float> tmp_dists;
    if (!local_map.nearestKSearchSurf(query, tmp_pts, tmp_dists,
                                      static_cast<int>(pts_out.size()))) {
      return 0;
    }

    const int n = static_cast<int>(std::min(tmp_pts.size(), pts_out.size()));
    for (int i = 0; i < n; ++i) {
      pts_out[i] = tmp_pts[i].getVector3fMap().template cast<double>();
      dists_out[i] = static_cast<double>(tmp_dists[i]);
    }
    return n;
  }
};
}  // namespace loam_baseline

#endif  // LOAM_BASELINE_LIDAR_SLAM_HPP_
