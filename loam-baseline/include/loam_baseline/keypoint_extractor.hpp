#ifndef LOAM_BASELINE_KEYPOINT_EXTRACTOR_HPP_
#define LOAM_BASELINE_KEYPOINT_EXTRACTOR_HPP_

#include <pcl/point_cloud.h>

#include <bitset>
#include <numbers>
#include <vector>

#include "loam_classic/sensor_data/LidarPoint.h"

namespace loam_baseline {

/// Bitfield of keypoint type labels for a single point.
/// Bit positions match the Keypoint enum values.
using KeypointFlags = std::bitset<3>;

/// Keypoint type labels. Used as bitset indices so kept as plain enum.
enum Keypoint : std::size_t {
  kEdge = 0,   ///< sharp local structure (line-like)
  kPlane = 1,  ///< flat local structure (surface-like)
  kBlob = 2,   ///< spherical local structure
};

/// Configuration for KeypointExtractor. All thresholds are in SI units.
struct KeypointExtractorCfg {
  /// Number of TBB threads (1 = serial).
  int nb_threads = 1;

  /// Half-width of the neighbourhood used for curvature operators.
  int neighbor_width = 4;

  /// Points closer than this to the sensor are rejected [m].
  double min_distance_to_sensor = 0.5;

  /// Points farther than this from the sensor are rejected [m].
  double max_distance_to_sensor = 100.0;

  /// Expected azimuthal angular step of the lidar [rad].
  /// Default: 0.4° (VLP-16 + 20% margin).
  double angle_resolution = 0.4 * std::numbers::pi_v<double> / 180.0;

  /// Sine-of-angle threshold for plane selection.
  /// A point is planar when sin(angle) < threshold.
  double plane_sin_angle_threshold = 0.6;  // ≈ sin(37°)

  /// Sine-of-angle threshold for edge selection.
  /// A point is an edge when sin(angle) > threshold.
  double edge_sin_angle_threshold = 0.86;  // ≈ sin(60°)

  /// Max distance from a point to its fitted neighbourhood line [m].
  double dist_to_line_threshold = 0.20;

  /// Depth-gap magnitude required to label a point as an edge [m].
  double edge_depth_gap_threshold = 0.15;

  /// Saliency (out-of-plane distance) required to label a point as an edge [m].
  double edge_saliency_threshold = 1.5;

  /// Intensity discontinuity required to label a point as an edge.
  double edge_intensity_gap_threshold = 50.0;
};

/// Output of a single KeypointExtractor::computeKeyPoints() call.
struct KeypointExtractorResult {
  pcl::PointCloud<PointXYZTIId> edges;
  pcl::PointCloud<PointXYZTIId> planes;
};

/// Curvature-based lidar keypoint extractor.
///
/// Ported from arise_slam_mid360::LidarKeypointExtractor.
/// All arise_slam / ROS dependencies removed; algorithm is unchanged.
class KeypointExtractor {
 public:
  using Point = PointXYZTIId;

  explicit KeypointExtractor(KeypointExtractorCfg cfg = {});

  /// Extract edge and plane keypoints from a pre-sorted point cloud.
  ///
  /// @param cloud       Input cloud; points must carry a valid `laserId` field.
  /// @param n_scans     Number of lidar scan lines (rings) in the cloud.
  /// @param dynamic_mask When true, invalidates points in a forward-facing
  ///                     horizontal band to suppress dynamic obstacles.
  KeypointExtractorResult computeKeyPoints(const pcl::PointCloud<Point>& cloud,
                                           int n_scans, bool dynamic_mask);

  KeypointExtractorCfg cfg;

 private:
  void prepareForNextFrame();
  void convertAndSortScanLines();
  void computeCurvature();
  void invalidPointWithBadCriteria(bool dynamic_mask);
  void setKeyPointsLabels(KeypointExtractorResult& out);

  bool isScanLineAlmostEmpty(int n_pts) const {
    return n_pts < 2 * cfg.neighbor_width + 1;
  }

  // Per-frame state. Reset at the start of each computeKeyPoints() call.
  int n_lasers_ = 0;
  pcl::PointCloud<Point>::Ptr current_frame_;
  std::vector<pcl::PointCloud<Point>::Ptr> frame_by_scan_;

  std::vector<std::vector<double>> angles_;
  std::vector<std::vector<double>> depth_gap_;
  std::vector<std::vector<double>> saliency_;
  std::vector<std::vector<double>> intensity_gap_;
  std::vector<std::vector<KeypointFlags>> is_point_valid_;
  std::vector<std::vector<KeypointFlags>> label_;
};

}  // namespace loam_baseline

#endif  // LOAM_BASELINE_KEYPOINT_EXTRACTOR_HPP_
