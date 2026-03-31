// Ported from arise_slam_mid360/src/LidarProcess/LidarSlam.cpp
// Changes vs. upstream:
//   - Ceres problem/solver replaced by iteratively re-weighted Gauss-Newton
//     (IRWGN) with Tukey loss via edgeResidualAndJacobian /
//     planeResidualAndJacobian from lidarResiduals.hpp.
//   - EstimateRegistrationError takes the final GN Hessian directly (H^-1
//     via JacobiSVD) instead of ceres::Covariance.
//   - initROSInterface / RCLCPP_* / tf2 removed; glog used for warnings.
//   - SE3AbsolutatePoseFactor (visual-odometry only) removed.
//   - ComputePointInitAndFinalPose inlined; only NONE undistortion supported.
//   - MannualYawCorrection uses Eigen RPY instead of tf2.
//   - TicToc replaced with std::chrono.

#include "loam_baseline/LidarSlam.h"
#include "loam_baseline/utils/Utilities.h"
#include "loam_baseline/lidarResiduals.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace loam_baseline {

namespace {

void rotationToRPY(const Eigen::Matrix3d& R,
                   double& roll, double& pitch, double& yaw) {
    roll  = std::atan2(R(2, 1), R(2, 2));
    pitch = std::atan2(-R(2, 0), std::hypot(R(2, 1), R(2, 2)));
    yaw   = std::atan2(R(1, 0), R(0, 0));
}

}  // namespace

// ---------------------------------------------------------------------------

LidarSLAM::LidarSLAM() {
    EdgesPoints.reset(new PointCloud());
    PlanarsPoints.reset(new PointCloud());
    WorldEdgesPoints.reset(new PointCloud());
    WorldPlanarsPoints.reset(new PointCloud());
}

// ---------------------------------------------------------------------------

void LidarSLAM::Localization(bool initialization,
                              PredictionSource /*predictodom*/,
                              Transformd position,
                              pcl::PointCloud<Point>::Ptr edge_point,
                              pcl::PointCloud<Point>::Ptr planner_point,
                              double timeLaserOdometry) {
    bInitilization = initialization;
    T_w_lidar = position;
    const Transformd T_w_initial_guess(T_w_lidar);

    this->EdgesPoints->clear();
    this->PlanarsPoints->clear();
    this->EdgesPoints->reserve(edge_point->size());
    this->PlanarsPoints->reserve(planner_point->size());
    for (const Point& p : *edge_point)    this->EdgesPoints->push_back(p);
    for (const Point& p : *planner_point) this->PlanarsPoints->push_back(p);

    if (not bInitilization) {
        // First frame: seed the local map.
        localMap.setOrigin(T_w_lidar.pos);

        this->WorldEdgesPoints->clear();
        this->WorldEdgesPoints->points.reserve(this->EdgesPoints->size());
        this->WorldEdgesPoints->header = this->EdgesPoints->header;
        for (const Point& p : *this->EdgesPoints)
            WorldEdgesPoints->push_back(TransformPointd(p, this->T_w_lidar));
        localMap.addEdgePointCloud(*this->WorldEdgesPoints);

        this->WorldPlanarsPoints->clear();
        this->WorldPlanarsPoints->points.reserve(this->PlanarsPoints->size());
        this->WorldPlanarsPoints->header = this->PlanarsPoints->header;
        for (const Point& p : *this->PlanarsPoints)
            WorldPlanarsPoints->push_back(TransformPointd(p, this->T_w_lidar));
        localMap.addSurfPointCloud(*this->WorldPlanarsPoints);

        lasttimeLaserOdometry = timeLaserOdometry;
        return;
    }

    // -----------------------------------------------------------------------
    // Localization frame
    // -----------------------------------------------------------------------
    this->pos_in_localmap = localMap.shiftMap(T_w_lidar.pos);

    auto [edgePointsFromMapNum, planarPointsFromMapNum] =
        localMap.get5x5LocalMapFeatureSize(this->pos_in_localmap);

    stats.laser_cloud_corner_from_map_num = edgePointsFromMapNum;
    stats.laser_cloud_surf_from_map_num   = planarPointsFromMapNum;
    stats.laser_cloud_corner_stack_num    = EdgesPoints->size();
    stats.laser_cloud_surf_stack_num      = PlanarsPoints->size();
    stats.iterations.clear();

    const auto t_opt_start = std::chrono::steady_clock::now();

    OptimizationData.clear();

    // Tukey loss thresholds (matching upstream: sqrt(3 * res))
    const double c_edge  = std::sqrt(3.0 * localMap.lineRes_);
    const double c_plane = std::sqrt(3.0 * localMap.planeRes_);

    // Working pose — kept separate so T_w_lidar is always consistent for
    // matching queries at the start of each outer ICP iteration.
    Eigen::Quaterniond q_curr = T_w_lidar.rot;
    Eigen::Vector3d    t_curr = T_w_lidar.pos;

    // Final GN Hessian (6×6) used for covariance estimation.
    Eigen::Matrix<double, 6, 6> last_H =
        Eigen::Matrix<double, 6, 6>::Identity();

    int Good_Planner_Feature_Num = 0;

    if (planarPointsFromMapNum > 50) {
        for (size_t icpIter = 0; icpIter < this->LocalizationICPMaxIter;
             ++icpIter) {
            IterationStats iter_stats;
            ResetDistanceParameters();
            Good_Planner_Feature_Num = 0;

            // Sync working pose into T_w_lidar so matching uses current estimate.
            T_w_lidar.rot = q_curr;
            T_w_lidar.pos = t_curr;

            // --- Edge feature matching ---
            if (not this->EdgesPoints->empty()) {
                for (const Point& pt : this->EdgesPoints->points) {
                    auto opt = ComputeLineDistanceParameters(this->localMap, pt);
                    this->MatchRejectionHistogramLine[opt.match_result]++;
                    if (opt.match_result == MatchingResult::SUCCESS)
                        OptimizationData.push_back(std::move(opt));
                }
            }

            // --- Plane feature matching (with optional subsampling) ---
            const int nSurf = static_cast<int>(this->PlanarsPoints->points.size());
            double sampling_rate = -1.0;
            if (nSurf > OptSet.max_surface_features)
                sampling_rate = 1.0 * OptSet.max_surface_features / nSurf;

            if (not this->PlanarsPoints->empty()) {
                for (size_t i = 0; i < this->PlanarsPoints->points.size(); ++i) {
                    if (sampling_rate > 0.0) {
                        double rem = std::fmod(i * sampling_rate, 1.0);
                        if (rem + 0.001 > sampling_rate) continue;
                    }
                    auto opt = ComputePlaneDistanceParameters(
                        this->localMap, this->PlanarsPoints->points[i]);
                    this->MatchRejectionHistogramPlane[opt.match_result]++;
                    if (opt.match_result == MatchingResult::SUCCESS) {
                        const auto& obs = opt.feature.observability;
                        this->PlaneFeatureHistogramObs[obs.at(0)]++;
                        this->PlaneFeatureHistogramObs[obs.at(1)]++;
                        this->PlaneFeatureHistogramObs[obs.at(2)]++;
                        ++Good_Planner_Feature_Num;
                        OptimizationData.push_back(std::move(opt));
                    }
                }
            }

            if (icpIter == 0) {
                stats.plane_match_success      = MatchRejectionHistogramPlane.at(0);
                stats.plane_no_enough_neighbor = MatchRejectionHistogramPlane.at(1);
                stats.plane_neighbor_too_far   = MatchRejectionHistogramPlane.at(2);
                stats.plane_badpca_structure   = MatchRejectionHistogramPlane.at(3);
                stats.plane_invalid_numerical  = MatchRejectionHistogramPlane.at(4);
                stats.plane_mse_too_large      = MatchRejectionHistogramPlane.at(5);
                stats.plane_unknown            = MatchRejectionHistogramPlane.at(6);
            }

            // --- IRWGN inner loop (replaces ceres::Solve with max_num_iterations=4) ---
            int edge_num = 0, surf_num = 0;
            bool inner_converged = false;
            constexpr int kMaxInnerIter = 4;

            for (int inner = 0; inner < kMaxInnerIter; ++inner) {
                Eigen::Matrix<double, 6, 6> H =
                    Eigen::Matrix<double, 6, 6>::Zero();
                Eigen::Matrix<double, 6, 1> b =
                    Eigen::Matrix<double, 6, 1>::Zero();
                edge_num = 0;
                surf_num = 0;

                for (const auto& opt : OptimizationData) {
                    if (opt.feature_type == FeatureType::EdgeFeature) {
                        Eigen::Vector3d r_e;
                        Eigen::Matrix<double, 3, 6> J_e;
                        edgeResidualAndJacobian(opt.Xvalue,
                                               opt.corres.first,
                                               opt.corres.second,
                                               q_curr, t_curr, r_e, J_e);
                        const double w = opt.residualCoefficient *
                                         tukeyWeight(r_e.norm(), c_edge);
                        if (w > 0.0) {
                            H += w * J_e.transpose() * J_e;
                            b += w * J_e.transpose() * r_e;
                        }
                        ++edge_num;
                    } else {  // PlaneFeature
                        double r_p;
                        Eigen::Matrix<double, 1, 6> h_p;
                        planeResidualAndJacobian(opt.Xvalue,
                                                opt.NormDir,
                                                opt.negative_OA_dot_norm,
                                                q_curr, t_curr, r_p, h_p);
                        const double w = opt.residualCoefficient *
                                         tukeyWeight(std::abs(r_p), c_plane);
                        if (w > 0.0) {
                            H += w * h_p.transpose() * h_p;
                            b += w * h_p.transpose() * r_p;
                        }
                        ++surf_num;
                    }
                }

                last_H = H;

                // Solve (H + λI)·dx = −b with a small Tikhonov ridge
                const Eigen::Matrix<double, 6, 1> dx =
                    -(H + 1e-8 * Eigen::Matrix<double, 6, 6>::Identity())
                          .ldlt()
                          .solve(b);

                retractPose(q_curr, t_curr, dx);

                if (dx.norm() < 1e-6) {
                    inner_converged = true;
                    break;
                }
            }  // inner IRWGN

            iter_stats.num_surf_from_scan   = surf_num;
            iter_stats.num_corner_from_scan = edge_num;

            const Transformd prev_T = T_w_lidar;
            T_w_lidar.rot = q_curr;
            T_w_lidar.pos = t_curr;

            const Transformd incremental_T = prev_T.inverse() * T_w_lidar;
            iter_stats.translation_norm = incremental_T.pos.norm();
            iter_stats.rotation_norm    =
                2.0 * std::atan2(incremental_T.rot.vec().norm(),
                                 incremental_T.rot.w());
            stats.iterations.push_back(iter_stats);

            // Optional: override roll/pitch from IMU (keep optimised yaw).
            if (OptSet.use_imu_roll_pitch) {
                const Eigen::Matrix3d R = T_w_lidar.rot.toRotationMatrix();
                double roll, pitch, yaw;
                rotationToRPY(R, roll, pitch, yaw);

                const Eigen::Quaterniond yaw_q(
                    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
                q_curr = (yaw_q * OptSet.imu_roll_pitch).normalized();
                T_w_lidar.rot = q_curr;
            }

            // Break when inner loop converged (gradient already small) or
            // at the last ICP iteration — mirrors upstream ceres check:
            //   (summary.num_successful_steps == 1) || (last iter)
            if (inner_converged || icpIter == this->LocalizationICPMaxIter - 1) {
                this->LocalizationUncertainty =
                    EstimateRegistrationError(last_H, 100.0);
                break;
            }
        }  // outer ICP
    } else {
        // LOG(WARNING) << "Not enough features for optimization";
    }

    MannualYawCorrection();

    const auto t_opt_end = std::chrono::steady_clock::now();
    stats.time_elapsed =
        std::chrono::duration<double>(t_opt_end - t_opt_start).count() * 1e3;

    const Transformd total_T = T_w_initial_guess.inverse() * T_w_lidar;
    stats.total_translation =
        total_T.pos.norm();
    stats.total_rotation =
        2.0 * std::atan2(total_T.rot.vec().norm(), total_T.rot.w());

    const Transformd diff_T = last_T_w_lidar.inverse() * T_w_lidar;
    stats.translation_from_last =
        diff_T.pos.norm();
    stats.rotation_from_last =
        2.0 * std::atan2(diff_T.rot.vec().norm(), diff_T.rot.w());
    last_T_w_lidar = T_w_lidar;
    stats.prediction_source = 0;

    bool acceptResult = true;
    const double delta_t = timeLaserOdometry - lasttimeLaserOdometry;

    if (stats.translation_from_last / delta_t > OptSet.velocity_failure_threshold) {
        T_w_lidar = last_T_w_lidar;
        startupCount = 5;
        acceptResult = false;
        // DLOG(WARNING) << "Large motion detected, ignoring predictor for a while. "
        //                  "translation_from_last=" << stats.translation_from_last;
    }

    if ((stats.translation_from_last < 0.02 &&
         stats.rotation_from_last    < 0.005) &&
        stats.laser_cloud_corner_from_map_num > 10 &&
        stats.laser_cloud_surf_from_map_num   > 50) {
        acceptResult = false;
        if (stats.translation_from_last < 0.005 &&
            stats.rotation_from_last    < 0.005)
            T_w_lidar = last_T_w_lidar;
        // DLOG(WARNING) << "Very small motion, not accumulating. "
        //                  "translation_from_last=" << stats.translation_from_last;
    }

    lasttimeLaserOdometry = timeLaserOdometry;
    localMap.shiftMap(this->T_w_lidar.pos);

    // Add features to map
    const auto t_add = std::chrono::steady_clock::now();

    this->WorldEdgesPoints->clear();
    this->WorldEdgesPoints->points.reserve(this->EdgesPoints->size());
    this->WorldEdgesPoints->header = this->EdgesPoints->header;
    for (const Point& p : *this->EdgesPoints)
        WorldEdgesPoints->push_back(TransformPointd(p, this->T_w_lidar));
    if (acceptResult) localMap.addEdgePointCloud(*this->WorldEdgesPoints);

    this->WorldPlanarsPoints->clear();
    this->WorldPlanarsPoints->points.reserve(this->PlanarsPoints->size());
    this->WorldPlanarsPoints->header = this->PlanarsPoints->header;
    for (const Point& p : *this->PlanarsPoints)
        WorldPlanarsPoints->push_back(TransformPointd(p, this->T_w_lidar));
    if (acceptResult) localMap.addSurfPointCloud(*this->WorldPlanarsPoints);

    kdtree_time_analysis.kd_tree_building_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_add).count() * 1e3;
}

// ---------------------------------------------------------------------------

LidarSLAM::OptimizationParameter LidarSLAM::ComputeLineDistanceParameters(
    LocalMap& local_map, const LidarSLAM::Point& p) {

    // Transform query point to world frame (NONE undistortion only).
    const Eigen::Vector3d pInit  = p.getVector3fMap().cast<double>();
    const Eigen::Vector3d pFinal = T_w_lidar * pInit;

    const size_t min_neighbors   = this->LocalizationMinmumLineNeighborRejection;
    size_t requiredNearest       = this->LocalizationLineDistanceNbrNeighbors;
    const double square_max_dist = 3.0 * local_map.lineRes_;

    Point pFinal_query;
    pFinal_query.x = static_cast<float>(pFinal.x());
    pFinal_query.y = static_cast<float>(pFinal.y());
    pFinal_query.z = static_cast<float>(pFinal.z());

    std::vector<Point> nearest_pts;
    std::vector<float> nearest_dist;

    OptimizationParameter result;

    const bool bFind = local_map.nearestKSearchSpecificEdgePoint(
        pFinal_query, nearest_pts, nearest_dist, requiredNearest,
        static_cast<float>(this->LocalizationLineMaxDistInlier));

    if (not bFind) {
        result.match_result = MatchingResult::NOT_ENOUGH_NEIGHBORS;
        return result;
    }
    if (nearest_pts.size() < min_neighbors) {
        result.match_result = MatchingResult::NOT_ENOUGH_NEIGHBORS;
        return result;
    }

    // Update to actual number returned.
    requiredNearest = nearest_pts.size();

    if (nearest_dist.back() > square_max_dist) {
        result.match_result = MatchingResult::NEIGHBORS_TOO_FAR;
        return result;
    }

    Eigen::MatrixXd data(requiredNearest, 3);
    for (size_t k = 0; k < requiredNearest; ++k)
        data.row(k) << nearest_pts[k].x, nearest_pts[k].y, nearest_pts[k].z;

    Eigen::Vector3d mean;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig = ComputePCA(data, mean);
    const Eigen::Vector3d D = eig.eigenvalues();

    if (D(2) < static_cast<double>(requiredNearest) * D(1)) {
        result.match_result = BAD_PCA_STRUCTURE;
        return result;
    }

    const Eigen::Vector3d n = eig.eigenvectors().col(2);
    const Eigen::Matrix3d A = Eigen::Matrix3d::Identity() - n * n.transpose();

    if (!std::isfinite(A(0, 0))) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return result;
    }

    double meanSquareDist = 0.0;
    for (const auto& pt : nearest_pts) {
        const Eigen::Vector3d Xtemp{pt.x, pt.y, pt.z};
        const double sq = (Xtemp - mean).transpose() * A * (Xtemp - mean);
        if (sq > square_max_dist) {
            result.match_result = MatchingResult::MSE_TOO_LARGE;
            return result;
        }
        meanSquareDist += sq;
    }
    meanSquareDist /= static_cast<double>(requiredNearest);

    const double fitQuality = 1.0 - std::sqrt(meanSquareDist / square_max_dist);
    const Eigen::Vector3d point_a =  0.1 * n + mean;
    const Eigen::Vector3d point_b = -0.1 * n + mean;

    result.feature_type        = FeatureType::EdgeFeature;
    result.match_result        = MatchingResult::SUCCESS;
    result.Avalue              = A;
    result.Pvalue              = mean;
    result.Xvalue              = pInit;
    result.corres              = {point_a, point_b};
    result.TimeValue           = 1.0;
    result.residualCoefficient = fitQuality;
    return result;
}

// ---------------------------------------------------------------------------

LidarSLAM::OptimizationParameter LidarSLAM::ComputePlaneDistanceParameters(
    LocalMap& local_map, const LidarSLAM::Point& p) {

    const Eigen::Vector3d pInit  = p.getVector3fMap().cast<double>();
    const Eigen::Vector3d pFinal = T_w_lidar * pInit;

    const size_t requiredNearest = this->LocalizationPlaneDistanceNbrNeighbors;
    const double squredMaxDist   = 3.0 * local_map.planeRes_;

    Point pFinal_query;
    pFinal_query.x = static_cast<float>(pFinal.x());
    pFinal_query.y = static_cast<float>(pFinal.y());
    pFinal_query.z = static_cast<float>(pFinal.z());

    std::vector<Point> nearest_pts;
    std::vector<float> nearest_dist;

    const auto t_kd = std::chrono::steady_clock::now();
    const bool bFind = local_map.nearestKSearchSurf(
        pFinal_query, nearest_pts, nearest_dist, requiredNearest);
    kdtree_time_analysis.kd_tree_query_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_kd).count() * 1e3;

    OptimizationParameter result;

    if (not bFind || nearest_dist.size() < 5 || nearest_pts.size() < 5) {
        result.match_result = MatchingResult::NOT_ENOUGH_NEIGHBORS;
        return result;
    }
    if (nearest_dist.back() > squredMaxDist) {
        result.match_result = MatchingResult::NEIGHBORS_TOO_FAR;
        return result;
    }

    // PCA for observability analysis
    Eigen::MatrixXd data(requiredNearest, 3);
    for (size_t k = 0; k < requiredNearest; ++k)
        data.row(k) << nearest_pts[k].x, nearest_pts[k].y, nearest_pts[k].z;

    Eigen::Vector3d mean;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig = ComputePCA(data, mean);
    const Eigen::Vector3d D = eig.eigenvalues();

    // Fit plane: matA0 * norm = matB0 (least-squares)
    Eigen::Matrix<double, 5, 3> matA0;
    Eigen::Matrix<double, 5, 1> matB0 =
        -Eigen::Matrix<double, 5, 1>::Ones();

    if (nearest_dist.at(4) < squredMaxDist) {
        for (int j = 0; j < 5; ++j) {
            matA0(j, 0) = nearest_pts.at(j).x;
            matA0(j, 1) = nearest_pts.at(j).y;
            matA0(j, 2) = nearest_pts.at(j).z;
        }
    }

    Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
    const double negative_OA_dot_norm = 1.0 / norm.norm();
    norm.normalize();

    double meanSquaredDist = 0.0;
    for (int j = 0; j < 5; ++j) {
        const double dis = std::abs(norm(0) * nearest_pts.at(j).x +
                                    norm(1) * nearest_pts.at(j).y +
                                    norm(2) * nearest_pts.at(j).z +
                                    negative_OA_dot_norm);
        if (dis > local_map.planeRes_ / 2.0f) {
            result.match_result = MatchingResult::MSE_TOO_LARGE;
            return result;
        }
        meanSquaredDist += dis;
    }
    meanSquaredDist /= 5.0;

    const double fitQuality = 1.0 - std::sqrt(meanSquaredDist / squredMaxDist);

    // Normal direction: orient towards the sensor (dot product check).
    Eigen::Vector3d correct_normal = eig.eigenvectors().col(0);
    if (pFinal.dot(correct_normal) < 0.0) correct_normal = -correct_normal;

    pcaFeature feature;
    FeatureObservabilityAnalysis(feature, pFinal, D, correct_normal,
                                 eig.eigenvectors().col(2));

    result.feature_type         = FeatureType::PlaneFeature;
    result.feature              = feature;
    result.match_result         = MatchingResult::SUCCESS;
    result.Pvalue               = mean;
    result.Xvalue               = pInit;
    result.NormDir              = norm;
    result.negative_OA_dot_norm = negative_OA_dot_norm;
    result.TimeValue            = 1.0;
    result.residualCoefficient  = fitQuality;
    return result;
}

// ---------------------------------------------------------------------------

void LidarSLAM::FeatureObservabilityAnalysis(
    pcaFeature& feature, Eigen::Vector3d p_query, Eigen::Vector3d lamada,
    Eigen::Vector3d normal_direction, Eigen::Vector3d principal_direction) {

    feature.pt.x = p_query.x();
    feature.pt.y = p_query.y();
    feature.pt.z = p_query.z();

    normal_direction.normalize();
    principal_direction.normalize();

    feature.vectors.principalDirection = principal_direction.cast<float>();
    feature.vectors.normalDirection    = normal_direction.cast<float>();
    feature.values.lamada1 = std::sqrt(lamada(2));
    feature.values.lamada2 = std::sqrt(lamada(1));
    feature.values.lamada3 = std::sqrt(lamada(0));

    const double sum_lam = feature.values.lamada1 +
                           feature.values.lamada2 +
                           feature.values.lamada3;
    feature.curvature  = (sum_lam > 0.0) ? feature.values.lamada3 / sum_lam : 0.0;
    feature.linear_2   = (feature.values.lamada1 - feature.values.lamada2) /
                          feature.values.lamada1;
    feature.planar_2   = (feature.values.lamada2 - feature.values.lamada3) /
                          feature.values.lamada1;
    feature.spherical_2 = feature.values.lamada3 / feature.values.lamada1;

    const Eigen::Vector3f point = p_query.cast<float>();
    Eigen::Quaternionf rot(T_w_lidar.rot.w(), T_w_lidar.rot.x(),
                           T_w_lidar.rot.y(), T_w_lidar.rot.z());
    rot.normalize();

    const Eigen::Vector3f rot_x = rot * Eigen::Vector3f::UnitX();
    const Eigen::Vector3f rot_y = rot * Eigen::Vector3f::UnitY();
    const Eigen::Vector3f rot_z = rot * Eigen::Vector3f::UnitZ();

    const Eigen::Vector3f cross = point.cross(feature.vectors.normalDirection);

    feature.rx_cross     =  cross.dot(rot_x);
    feature.neg_rx_cross = -cross.dot(rot_x);
    feature.ry_cross     =  cross.dot(rot_y);
    feature.neg_ry_cross = -cross.dot(rot_y);
    feature.rz_cross     =  cross.dot(rot_z);
    feature.neg_rz_cross = -cross.dot(rot_z);

    const float p2 = static_cast<float>(feature.planar_2 * feature.planar_2);
    feature.tx_dot = p2 * std::abs(feature.vectors.normalDirection.dot(rot_x));
    feature.ty_dot = p2 * std::abs(feature.vectors.normalDirection.dot(rot_y));
    feature.tz_dot = p2 * std::abs(feature.vectors.normalDirection.dot(rot_z));

    using Pair = std::pair<float, Feature_observability>;
    std::vector<Pair> rotation_quality = {
        {feature.rx_cross,     Feature_observability::rx_cross},
        {feature.neg_rx_cross, Feature_observability::neg_rx_cross},
        {feature.ry_cross,     Feature_observability::ry_cross},
        {feature.neg_ry_cross, Feature_observability::neg_ry_cross},
        {feature.rz_cross,     Feature_observability::rz_cross},
        {feature.neg_rz_cross, Feature_observability::neg_rz_cross},
    };
    std::vector<Pair> trans_quality = {
        {feature.tx_dot, Feature_observability::tx_dot},
        {feature.ty_dot, Feature_observability::ty_dot},
        {feature.tz_dot, Feature_observability::tz_dot},
    };

    auto cmp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
    std::sort(rotation_quality.begin(), rotation_quality.end(), cmp);
    std::sort(trans_quality.begin(),    trans_quality.end(),    cmp);

    feature.observability.at(0) = rotation_quality.at(0).second;
    feature.observability.at(1) = rotation_quality.at(1).second;
    feature.observability.at(2) = trans_quality.at(0).second;
    feature.observability.at(3) = trans_quality.at(1).second;
}

// ---------------------------------------------------------------------------

inline void LidarSLAM::ResetDistanceParameters() {
    this->OptimizationData.clear();
    for (auto& e : MatchRejectionHistogramLine)  e = 0;
    for (auto& e : MatchRejectionHistogramPlane) e = 0;
    for (auto& e : PlaneFeatureHistogramObs)     e = 0;
}

// ---------------------------------------------------------------------------

LidarSLAM::RegistrationError LidarSLAM::EstimateRegistrationError(
    const Eigen::Matrix<double, 6, 6>& H, double /*eigen_thresh*/) {

    RegistrationError err;

    // Covariance ≈ H⁻¹ via full SVD (matches upstream ceres::DENSE_SVD path).
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(
        H, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix<double, 6, 1> inv_s = svd.singularValues();
    for (int i = 0; i < 6; ++i)
        inv_s(i) = (inv_s(i) > 1e-10) ? 1.0 / inv_s(i) : 0.0;

    err.Covariance =
        svd.matrixV() * inv_s.asDiagonal() * svd.matrixU().transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigPos(
        err.Covariance.topLeftCorner<3, 3>());
    err.PositionError          = std::sqrt(eigPos.eigenvalues()(2));
    err.PositionErrorDirection = eigPos.eigenvectors().col(2);
    err.PosInverseConditionNum = std::sqrt(eigPos.eigenvalues()(0)) /
                                 std::sqrt(eigPos.eigenvalues()(2));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigOri(
        err.Covariance.bottomRightCorner<3, 3>());
    err.OrientationError          = Rad2Deg(std::sqrt(eigOri.eigenvalues()(2)));
    err.OrientationErrorDirection = eigOri.eigenvectors().col(2);
    err.OriInverseConditionNum    = std::sqrt(eigOri.eigenvalues()(0)) /
                                    std::sqrt(eigOri.eigenvalues()(2));
    return err;
}

// ---------------------------------------------------------------------------

void LidarSLAM::MannualYawCorrection() {
    const Transformd last_current_T = last_T_w_lidar.inverse() * T_w_lidar;
    const float translation_norm = last_current_T.pos.norm();

    const Eigen::Matrix3d R = T_w_lidar.rot.toRotationMatrix();
    double roll, pitch, yaw;
    rotationToRPY(R, roll, pitch, yaw);

    const double correct_yaw = yaw + translation_norm * OptSet.yaw_ratio * M_PI / 180.0;

    T_w_lidar.rot = Eigen::Quaterniond(
        Eigen::AngleAxisd(correct_yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch,       Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll,        Eigen::Vector3d::UnitX()))
        .normalized();
}

// ---------------------------------------------------------------------------

bool LidarSLAM::DegeneracyDetection(
    double eigThreshold,
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& matAtA,
    Eigen::Matrix<double, 6, 1>& matX) {

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> esolver(matAtA);
    const Eigen::Matrix<double, 6, 1> matE = esolver.eigenvalues().real();
    const Eigen::Matrix<double, 6, 6> matV = esolver.eigenvectors().real();
    Eigen::Matrix<double, 6, 6> matV2 = matV;

    bool deg = false;
    for (int i = 0; i < 6; ++i) {
        if (matE(i) < eigThreshold) {
            matV2.row(i).setZero();
            deg = true;
        } else {
            break;
        }
    }

    if (deg) {
        const Eigen::Matrix<double, 6, 1> matX2(matX);
        matX = matV.inverse() * matV2 * matX2;
    }

    this->isDegenerate = deg;
    return deg;
}

}  // namespace loam_baseline
