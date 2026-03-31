
#include "loam_baseline/lidar_slam.hpp"
namespace loam_baseline {

/// Estimates the registration error from the Hessian of the last optimization
/// iteration.
///
/// # Reference
/// This is ported from `LidarSLAM::EstimateRegistrationError`
RegistrationError EstimateRegistrationError(
    const Eigen::Matrix<double, 6, 6>& hess) {
  using std::sqrt;
  RegistrationError err;

  // Covariance ≈ H⁻¹ via full SVD (matches upstream ceres::DENSE_SVD path).
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(
      hess, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Vector<double, 6> inv_s = svd.singularValues();
  inv_s = (inv_s.array() > 1e-10).select(inv_s.array().inverse(), 0.0);

  err.cov = svd.matrixV() * inv_s.asDiagonal() * svd.matrixU().transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<double>> eig;
  eig.compute(err.cov(ix::seq3(0), ix::seq3(0)));
  err.position_error =
      sqrt(eig.eigenvalues()[2]) * eig.eigenvectors()(ix::all, 2);
  err.position_inv_cond =
      sqrt(eig.eigenvalues()[0]) / sqrt(eig.eigenvalues()[2]);

  eig.compute(err.cov(ix::seq3(3), ix::seq3(3)));
  err.orientation_error =
      rad2deg(sqrt(eig.eigenvalues()(2))) * eig.eigenvectors()(ix::all, 2);
  err.orientation_inv_cond =
      sqrt(eig.eigenvalues()[0]) / sqrt(eig.eigenvalues()[2]);
  return err;
}

Localization::Localization(const LocalizationCfg& cfg) : cfg(cfg) {}

void Localization::initialize(
    const manifold::TransformF64& start_pose,
    std::span<const Eigen::Vector3d> edge_points_local,
    std::span<const Eigen::Vector3d> plane_points_local) {
  local_map.setOrigin(start_pose.translation.template cast<double>());
  // Seed the map — same body as updateMap but unconditional
  updateMap(/*accept_result=*/true, start_pose, edge_points_local,
            plane_points_local);
}

Localization::SolveResult Localization::solve(
    const manifold::TransformF64& initial_guess,
    std::span<const Eigen::Vector3d> edge_points,
    std::span<const Eigen::Vector3d> plane_points, int max_iterations,
    double dt, const manifold::TransformF64& prev_solution) {
  using std::fmod;
  // Derive Tukey thresholds from the map voxel resolution.
  solver.c_edge = std::sqrt(3.0 * matcher.line_res);
  solver.c_plane = std::sqrt(3.0 * matcher.plane_res);

  // T_wl: tranformation of world w.r.t. lidar/lidar to world
  manifold::TransformF64 lidar_to_world = initial_guess;

  // Shift the local map to the current estimate; record map feature counts
  // for the small-motion gate check.
  OptimizationStats stats;
  const Eigen::Vector3i pos_in_map =
      local_map.shiftMap(initial_guess.translation);
  auto [corner_map_num, surf_map_num] =
      local_map.get5x5LocalMapFeatureSize(pos_in_map);
  stats.laser_cloud_corner_from_map_num = corner_map_num;
  stats.laser_cloud_surf_from_map_num = surf_map_num;
  stats.laser_cloud_corner_stack_num = edge_points.size();
  stats.laser_cloud_surf_stack_num = plane_points.size();

  Eigen::Matrix<double, 6, 6> last_hess =
      Eigen::Matrix<double, 6, 6>::Identity();
  RegistrationError reg_err;

  constexpr auto kPlanarPointFromMapGate = 50;
  if (surf_map_num > kPlanarPointFromMapGate) {
    for (int icp_iter = 0; icp_iter < max_iterations; ++icp_iter) {
      solver.edges.clear();
      solver.planes.clear();

      // Edge feature matching
      for (const auto& pt : edge_points) {
        auto [result, feature] =
            matcher.matchEdge(pt, lidar_to_world,
                              std::bind_front(&Localization::searchEdge, this));
        if (result == MatchResult::kSuccess) {
          solver.edges.push_back(std::move(feature));
        }
      }

      // Plane feature matching with optional subsampling
      const auto n_surf = std::ssize(plane_points);
      double sampling_rate = -1.0;
      if (n_surf > cfg.max_surface_features) {
        sampling_rate = static_cast<double>(cfg.max_surface_features) /
                        static_cast<double>(n_surf);
      }
      for (std::size_t i = 0; i < static_cast<std::size_t>(n_surf); ++i) {
        if (sampling_rate > 0.0) {
          double rem = fmod(static_cast<double>(i) * sampling_rate, double(1));
          if (rem + 0.001 > sampling_rate) {
            continue;
          }
        }
        const auto& pt = plane_points[i];
        auto [result, feature] = matcher.matchPlane(
            pt, lidar_to_world,
            std::bind_front(&Localization::searchPlane, this));
        if (result == MatchResult::kSuccess) {
          solver.planes.emplace_back(std::move(feature));
        }
      }

      constexpr int kMaxInnerIterations = 4;
      bool inner_converged = false;
      for (int i = 0; i < kMaxInnerIterations; ++i) {
        auto iter_result = solver.iterateOnce(lidar_to_world);
        lidar_to_world = iter_result.new_pose;
        last_hess = iter_result.hessian;
        if (iter_result.converged) {
          inner_converged = true;
          stats.iterations.emplace_back(IterationStats{
              .num_surf_from_scan = iter_result.plane_num,
              .num_corner_from_scan = iter_result.edge_num,
          });
          break;
        }
      }
      if (inner_converged || icp_iter == max_iterations - 1) {
        reg_err = EstimateRegistrationError(last_hess);
        break;
      }
    }
  } else {
    // TODO: Log warning about insufficient map features for optimization
  }

  // NOTE: LOAM source offers an option to override IMU roll/pitch here. We
  // drop it because we fuse lidar pose with IMU downstream

  // T_w{l*} \ T_wl = T_{l*}l: total correction applied by this solve call.
  const manifold::TransformF64 total_tform = manifold::TransformProduct(
      manifold::TransformInverse(initial_guess), lidar_to_world);
  stats.total_translation = total_tform.translation.norm();
  stats.total_rotation =
      total_tform.rotation.angularDistance(Eigen::Quaterniond::Identity());

  // Delta from the previously accepted solution to the new estimate.
  const manifold::TransformF64 delta_tform = manifold::TransformProduct(
      manifold::TransformInverse(prev_solution), lidar_to_world);
  stats.translation_from_last = delta_tform.translation.norm();
  stats.rotation_from_last =
      delta_tform.rotation.angularDistance(Eigen::Quaterniond::Identity());

  bool accept_result = true;

  // Reject result if the estimated velocity (translation / dt) is implausible
  if (stats.translation_from_last > cfg.velocity_failure_threshold * dt) {
    // TODO: Log warning
    accept_result = false;
  }

  constexpr double kSmallTranslationTol(0.02);
  constexpr double kTinyTranslationTol(0.005);
  constexpr double kSmallRotationTol(0.005);
  constexpr int kSmallTranslationCornerGate = 10;
  constexpr int kSmallTranslationSurfGate = 50;

  // Reject result if the solution is close to the previous one even though
  // enough map features are observed
  if ((stats.translation_from_last < kSmallTranslationTol &&
       stats.rotation_from_last < kSmallRotationTol) &&
      stats.laser_cloud_corner_from_map_num > kSmallTranslationCornerGate &&
      stats.laser_cloud_surf_from_map_num > kSmallTranslationSurfGate) {
    accept_result = false;
    if (stats.translation_from_last < kTinyTranslationTol &&
        stats.rotation_from_last < kSmallRotationTol) {
      lidar_to_world = prev_solution;
    }
    // TODO: Log warning
  }

  local_map.shiftMap(lidar_to_world.translation);

  return SolveResult{
      .pose = lidar_to_world,
      .reg_err = reg_err,
      .stats = stats,
      .accept_result = accept_result,
  };
}

void Localization::updateMap(
    bool accept_result, const manifold::TransformF64& final_pose,
    std::span<const Eigen::Vector3d> edge_points_local,
    std::span<const Eigen::Vector3d> plane_points_local) {
  using std::ranges::transform;
  if (!accept_result) {
    return;
  }

  auto tf = [&final_pose](const Eigen::Vector3d& p_local) {
    auto p = manifold::TransformPoint(final_pose, p_local).cast<float>();
    return pcl::PointXYZI(p.x(), p.y(), p.z());
  };

  pcl::PointCloud<pcl::PointXYZI> world_edges;
  transform(edge_points_local, std::back_inserter(world_edges.points), tf);
  local_map.addEdgePointCloud(world_edges);

  pcl::PointCloud<pcl::PointXYZI> world_planes;
  transform(plane_points_local, std::back_inserter(world_planes.points), tf);
  local_map.addSurfPointCloud(world_planes);
}
}  // namespace loam_baseline
