#pragma once
// Ported from arise_slam_mid360/LidarProcess/LidarSlam.h
// Changes vs. upstream:
//   - Ceres and all factor headers removed; replaced by lidarResiduals.hpp
//   - rclcpp / tf2 / arise_slam_mid360_msgs removed; logging via glog
//   - arise_slam_mid360_msgs::OptimizationStats replaced by local POD structs
//   - SE3AbsolutatePoseFactor (visual-odometry only) removed
//   - WithinFrameMotion / MotionModel deferred (undistortion modes unused)

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <tbb/concurrent_vector.h>

#include <Eigen/Dense>
#include <array>
#include <atomic>
#include <string>
#include <vector>

#include "loam_baseline/LocalMap.h"
#include "loam_baseline/lidarResiduals.hpp"
#include "loam_baseline/sensor_data/LidarPoint.h"
#include "loam_baseline/utils/EigenTypes.h"
#include "loam_baseline/utils/Twist.h"

namespace loam_baseline {

// ---------------------------------------------------------------------------
// Lightweight stats structs replacing arise_slam_mid360_msgs ROS messages
// ---------------------------------------------------------------------------

struct IterationStats {
  int num_surf_from_scan = 0;
  int num_corner_from_scan = 0;
  double translation_norm = 0.0;
  double rotation_norm = 0.0;
};

struct OptimizationStats {
  int laser_cloud_corner_from_map_num = 0;
  int laser_cloud_surf_from_map_num = 0;
  int laser_cloud_corner_stack_num = 0;
  int laser_cloud_surf_stack_num = 0;

  int plane_match_success = 0;
  int plane_no_enough_neighbor = 0;
  int plane_neighbor_too_far = 0;
  int plane_badpca_structure = 0;
  int plane_invalid_numerical = 0;
  int plane_mse_too_large = 0;
  int plane_unknown = 0;

  double time_elapsed = 0.0;
  double total_translation = 0.0;
  double total_rotation = 0.0;
  double translation_from_last = 0.0;
  double rotation_from_last = 0.0;

  int prediction_source = 0;  // 0 = lidar only

  std::vector<IterationStats> iterations;
};

// ---------------------------------------------------------------------------

class LidarSLAM {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Point = pcl::PointXYZI;
  using PointCloud = pcl::PointCloud<Point>;

  enum class PredictionSource { IMU_ORIENTATION, IMU_ODOM };

  enum class MatchingMode { EGO_MOTION = 0, LOCALIZATION = 1 };

  // Scan undistortion mode — only NONE is implemented; others deferred.
  enum UndistortionMode { NONE = 0, APPROXIMATED = 1, OPTIMIZED = 2 };

  enum MatchingResult : uint8_t {
    SUCCESS = 0,
    NOT_ENOUGH_NEIGHBORS = 1,
    NEIGHBORS_TOO_FAR = 2,
    BAD_PCA_STRUCTURE = 3,
    INVAVLID_NUMERICAL = 4,
    MSE_TOO_LARGE = 5,
    UNKNON = 6,
    nRejectionCauses = 7
  };

  enum Feature_observability : uint8_t {
    rx_cross = 0,
    neg_rx_cross = 1,
    ry_cross = 2,
    neg_ry_cross = 3,
    rz_cross = 4,
    neg_rz_cross = 5,
    tx_dot = 6,
    ty_dot = 7,
    tz_dot = 8,
    nFeatureObs = 9
  };

  enum FeatureType : uint8_t { EdgeFeature = 0, PlaneFeature = 1 };

  struct LaserOptSet {
    Eigen::Quaterniond imu_roll_pitch = Eigen::Quaterniond::Identity();
    bool debug_view_enabled = false;
    bool use_imu_roll_pitch = false;
    float velocity_failure_threshold = 5.0f;
    float yaw_ratio = 1.0f;
    int max_surface_features = 2000;
  };

  struct RegistrationError {
    double PositionError = 0.0;
    double PositionUncertainty = 0.0;
    double MaxPositionError = 0.1;
    double PosInverseConditionNum = 1.0;
    Eigen::VectorXf LidarUncertainty;
    Eigen::Vector3d PositionErrorDirection = Eigen::Vector3d::Zero();
    double OrientationError = 0.0;
    double OrientationUncertainty = 0.0;
    double MaxOrientationError = 10.0;
    double OriInverseConditionNum = 1.0;
    Eigen::Vector3d OrientationErrorDirection = Eigen::Vector3d::Zero();
    // 6×6 pose covariance [X Y Z rX rY rZ]
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Covariance =
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
  };

  struct eigenValue {
    double lamada1, lamada2, lamada3;
  };
  struct eigenVector {
    Eigen::Vector3f principalDirection, middleDirection, normalDirection;
  };

  struct pcaFeature {
    eigenValue values;
    eigenVector vectors;
    double curvature = 0, linear = 0, planar = 0, spherical = 0;
    double linear_2 = 0, planar_2 = 0, spherical_2 = 0;
    pcl::PointNormal pt;
    size_t ptId = 0, ptNum = 0;
    std::vector<int> neighbor_indices;
    std::array<int, 4> observability{};
    double rx_cross = 0, neg_rx_cross = 0;
    double ry_cross = 0, neg_ry_cross = 0;
    double rz_cross = 0, neg_rz_cross = 0;
    double tx_dot = 0, ty_dot = 0, tz_dot = 0;
  };

  struct LidarOdomUncertainty {
    double uncertainty_x = 0, uncertainty_y = 0, uncertainty_z = 0;
    double uncertainty_roll = 0, uncertainty_pitch = 0, uncertainty_yaw = 0;
  };

  struct OptimizationParameter {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MatchingResult match_result;
    FeatureType feature_type;
    pcaFeature feature;
    Eigen::Matrix3d Avalue;
    Eigen::Vector3d Pvalue, Xvalue, NormDir;
    double negative_OA_dot_norm = 0.0;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> corres;
    double residualCoefficient = 1.0;
    double TimeValue = 0.0;
  };

  struct kdtree_time {
    double timestamp = 0.0;
    int frameID = 0;
    double kd_tree_building_time = 0.0;
    double kd_tree_query_time = 0.0;
  };

 public:
  LocalMap localMap;
  OptimizationStats stats;
  RegistrationError LocalizationUncertainty;
  LidarOdomUncertainty lidarOdomUncer;
  kdtree_time kdtree_time_analysis;
  LaserOptSet OptSet;

  Transformd T_w_lidar;
  Transformd last_T_w_lidar;
  Eigen::Matrix<double, 6, 1> Tworld = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Vector3i pos_in_localmap;

  int frame_count = 0;
  int startupCount = 0;

  float Pos_degeneracy_threshold = 100.0f;
  float Ori_degeneracy_threshold = 100.0f;
  float Visual_confidence_factor = 0.0f;  // 0 = no visual constraint

  UndistortionMode Undistortion = UndistortionMode::NONE;

  std::array<std::atomic_int, Feature_observability::nFeatureObs>
      PlaneFeatureHistogramObs{};
  std::array<std::atomic_int, MatchingResult::nRejectionCauses>
      MatchRejectionHistogramLine{};
  std::array<std::atomic_int, MatchingResult::nRejectionCauses>
      MatchRejectionHistogramPlane{};

  tbb::concurrent_vector<OptimizationParameter> OptimizationData;

  // Optimization parameters
  size_t LocalizationICPMaxIter = 4;
  size_t LocalizationLineDistanceNbrNeighbors = 10;
  size_t LocalizationMinmumLineNeighborRejection = 4;
  size_t LocalizationPlaneDistanceNbrNeighbors = 5;
  double LocalizationLineDistancefactor = 5.0;
  double LocalizationPlaneDistancefactor1 = 16.0;
  double LocalizationPlaneDistancefactor2 = 8.0;
  double LocalizationMaxPlaneDistance = 1.0;
  double LocalizationMaxLineDistance = 0.2;
  double LocalizationLineMaxDistInlier = 0.2;
  double MinNbrMatchedKeypoints = 20.0;
  double MaxDistanceForICPMatching = 20.0;
  double SaturationDistance = 1.0;
  double FrameDuration = 0.0;
  double lasttimeLaserOdometry = 0.0;

  PointCloud::Ptr EdgesPoints, PlanarsPoints;
  PointCloud::Ptr WorldEdgesPoints, WorldPlanarsPoints;

  bool bInitilization = false;
  bool isDegenerate = false;

 public:
  LidarSLAM();

  void Localization(bool initialization, PredictionSource predictodom,
                    Transformd T_w_lidar,
                    pcl::PointCloud<Point>::Ptr edge_point,
                    pcl::PointCloud<Point>::Ptr planner_point,
                    double timeLaserOdometry);

  OptimizationParameter ComputeLineDistanceParameters(LocalMap& local_map,
                                                      const Point& p);
  OptimizationParameter ComputePlaneDistanceParameters(LocalMap& local_map,
                                                       const Point& p);

  // Returns the 6×6 pose covariance from the final GN Hessian.
  // Replaces the upstream ceres::Covariance path.
  RegistrationError EstimateRegistrationError(
      const Eigen::Matrix<double, 6, 6>& H, double eigen_thresh);

  bool DegeneracyDetection(double eigThreshold,
                           Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& matAtA,
                           Eigen::Matrix<double, 6, 1>& matX);

  void FeatureObservabilityAnalysis(pcaFeature& feature,
                                    Eigen::Vector3d p_query,
                                    Eigen::Vector3d lamada,
                                    Eigen::Vector3d normal_direction,
                                    Eigen::Vector3d principal_direction);

  inline void ResetDistanceParameters();
  void MannualYawCorrection();

 private:
  std::function<void(const char* msg, bool is_debug)> warning_logger_;
};

}  // namespace loam_baseline
