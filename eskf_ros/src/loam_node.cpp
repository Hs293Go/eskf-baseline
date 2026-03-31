// ESKF + LOAM tightly coupled node.
//
// Feature pipeline (arise_slam feature extractor → loam_baseline → ESKF):
//   feature_info  →  Localization::solve()  →  driver_.pushPose()
//   imu           →  driver_.pushImu()
//
// The ESKF's propagated estimate at each scan time is used as the LOAM
// initial guess.  The 6×6 pose covariance from the GN Hessian (H⁻¹) is
// used directly as the ESKF measurement noise matrix R — completing the
// H⁻¹→R covariance propagation loop.

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <memory>
#include <numbers>
#include <vector>

#include "arise_slam_mid360_msgs/msg/laser_feature.hpp"
#include "diagnostic_updater/diagnostic_updater.hpp"
#include "eskf_baseline/eskf.hpp"
#include "eskf_baseline/eskf_baseline.hpp"
#include "eskf_baseline/inertial_odometry_driver.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "loam_baseline/lidar_slam.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2_eigen/tf2_eigen.hpp"

using eskf::Eskf;

namespace {
template <typename T>
T sq(T x) noexcept {
  return x * x;
}
template <typename T>
T deg2rad(T deg) noexcept {
  return deg * std::numbers::pi_v<T> / T(180);
}

std::vector<Eigen::Vector3d> cloudToEigen(
    const pcl::PointCloud<pcl::PointXYZI>& cloud) {
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(cloud.size());
  for (const auto& p : cloud.points)
    pts.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y),
                     static_cast<double>(p.z));
  return pts;
}
}  // namespace

class LoamNode : public rclcpp::Node {
 public:
  LoamNode() : Node("loam_node"), diag_(this) {
    // -----------------------------------------------------------------------
    // ESKF configuration
    // -----------------------------------------------------------------------
    const bool accel_unit_g =
        declare_parameter("accelerometer_unit_is_g", false);

    eskf::Config<double> cfg;
    cfg.accel_noise_density = declare_parameter("accel_noise_density", 1.0);
    cfg.gyro_noise_density = declare_parameter("gyro_noise_density", 0.01);
    if (!driver_.algorithm().setConfig(cfg))
      throw std::runtime_error("Invalid ESKF config parameters");

    RCLCPP_INFO(get_logger(),
                "ESKF: accel_noise_density=%.3f, gyro_noise_density=%.3f",
                cfg.accel_noise_density, cfg.gyro_noise_density);

    // -----------------------------------------------------------------------
    // Localization configuration
    // -----------------------------------------------------------------------
    loam_baseline::LocalizationCfg loc_cfg;
    loc_cfg.max_surface_features =
        declare_parameter("max_surface_features", 2000);
    loc_cfg.velocity_failure_threshold =
        declare_parameter("velocity_failure_threshold", 5.0);
    icp_max_iter_ = declare_parameter("icp_max_iter", 4);

    slam_ = std::make_unique<loam_baseline::Localization>(loc_cfg);
    slam_->matcher.line_res = declare_parameter("line_res", 0.2);
    slam_->matcher.plane_res = declare_parameter("plane_res", 0.4);
    slam_->matcher.min_line_neighbor_rejection = 4;  // from upstream default
    slam_->matcher.max_dist_inlier = 0.2;            // from upstream default

    // Minimum per-element variance floor for the measurement noise matrix R.
    // Prevents LOAM from claiming perfect certainty when the GN Hessian
    // is poorly conditioned (e.g., degenerate corridors).
    const double pos_min_std = declare_parameter("pos_min_stddev", 0.05);
    const double ori_min_std_deg = declare_parameter("ori_min_stddev_deg", 2.0);
    r_min_diag_.head<3>().setConstant(sq(pos_min_std));
    r_min_diag_.tail<3>().setConstant(sq(deg2rad(ori_min_std_deg)));

    RCLCPP_INFO(get_logger(),
                "Localization: icp_max_iter=%d, line_res=%.3f, plane_res=%.3f",
                icp_max_iter_, slam_->matcher.line_res,
                slam_->matcher.plane_res);

    // -----------------------------------------------------------------------
    // Subscriptions
    // -----------------------------------------------------------------------
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "eskf/imu", 10,
        [this, accel_unit_g](const sensor_msgs::msg::Imu::SharedPtr msg) {
          Eskf::Input imu;
          imu.t = rclcpp::Time(msg->header.stamp).seconds();
          tf2::fromMsg(msg->linear_acceleration, imu.data.accel);
          tf2::fromMsg(msg->angular_velocity, imu.data.gyro);
          angular_velocity_ = imu.data.gyro;
          if (accel_unit_g) imu.data.accel *= 9.81;
          driver_.pushImu(imu);
        });

    feature_sub_ =
        create_subscription<arise_slam_mid360_msgs::msg::LaserFeature>(
            "loam/feature_info", 2,
            [this](const arise_slam_mid360_msgs::msg::LaserFeature::SharedPtr
                       msg) { onFeature(msg); });

    RCLCPP_INFO(get_logger(), "IMU topic:     %s", imu_sub_->get_topic_name());
    RCLCPP_INFO(get_logger(), "Feature topic: %s",
                feature_sub_->get_topic_name());

    // -----------------------------------------------------------------------
    // Publishers
    // -----------------------------------------------------------------------
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("eskf/estimate", 1);
    grav_pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>(
        "eskf/estimate_gravity", 1);

    timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() {
      if (!driver_.running()) return;
      const auto now = get_clock()->now();
      const auto& [ctx, _] = driver_.getEstimate(now.seconds());

      nav_msgs::msg::Odometry odom;
      odom.header.stamp = now;
      odom.header.frame_id = frame_id_;
      odom.pose.pose.position = tf2::toMsg(ctx.est.x.p);
      odom.pose.pose.orientation = tf2::toMsg(ctx.est.x.q);
      tf2::toMsg(ctx.est.x.v, odom.twist.twist.linear);
      tf2::toMsg(angular_velocity_, odom.twist.twist.angular);
      odom_pub_->publish(odom);

      geometry_msgs::msg::Vector3Stamped grav;
      grav.header.stamp = now;
      grav.header.frame_id = frame_id_;
      tf2::toMsg(ctx.est.x.grav_vector, grav.vector);
      grav_pub_->publish(grav);
    });

    // -----------------------------------------------------------------------
    // Diagnostics (mirrors eskf_node)
    // -----------------------------------------------------------------------
    diag_.setHardwareID("loam_node");
    diag_.add("eskf/driver",
              [this](diagnostic_updater::DiagnosticStatusWrapper& stat) {
                const auto ws = driver_.getWindowStats();
                if (driver_.halted()) {
                  const auto h = driver_.getHaltedOutcome();
                  stat.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                               "HALTED: predict failure (or fatal)");
                  stat.add("halted_t", h.t);
                  stat.add("halted_reason", static_cast<int>(h.status));
                  stat.add("halted_msg", std::string(h.message));
                } else {
                  int level = diagnostic_msgs::msg::DiagnosticStatus::OK;
                  std::string msg = "OK";
                  if (ws.sample_age_s > 1.0) {
                    level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
                    msg = "Stats sample stale";
                  }
                  if (ws.predict_fail_hz > 0.0) {
                    level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
                    msg = "Predict failures observed";
                  }
                  stat.summary(level, msg);
                }
                stat.add("process_hz", ws.process_hz);
                stat.add("predict_ok_hz", ws.predict_ok_hz);
                stat.add("correct_ok_hz", ws.correct_ok_hz);
                stat.add("correct_reject_hz", ws.correct_reject_hz);
                stat.add("cpu_predict", ws.predict_cpu);
                stat.add("cpu_correct", ws.correct_cpu);
                stat.add("mean_predict_us", ws.mean_predict_us);
                stat.add("mean_correct_us", ws.mean_correct_us);
              });

    const double diag_hz = declare_parameter("diag_hz", 1.0);
    diag_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / std::max(0.1, diag_hz)),
        [this]() { diag_.force_update(); });
  }

 private:
  // -------------------------------------------------------------------------

  void onFeature(
      const arise_slam_mid360_msgs::msg::LaserFeature::SharedPtr& msg) {
    const double t = rclcpp::Time(msg->header.stamp).seconds();

    // Decode PCL clouds from the LaserFeature bundle.
    pcl::PointCloud<pcl::PointXYZI> corner, surface;
    pcl::fromROSMsg(msg->cloud_corner, corner);
    pcl::fromROSMsg(msg->cloud_surface, surface);
    const auto edge_pts = cloudToEigen(corner);
    const auto plane_pts = cloudToEigen(surface);

    frame_id_ = msg->header.frame_id.empty() ? "map" : msg->header.frame_id;

    // Build initial pose guess.
    //   - Before first LOAM result: use IMU quaternion from the feature msg
    //     (position unknown → zero).
    //   - After first result: propagate ESKF to scan time.
    manifold::TransformF64 T_init{
        .rotation = Eigen::Quaterniond::Identity(),
        .translation = Eigen::Vector3d::Zero(),
    };
    if (driver_.running()) {
      const auto& [ctx, _] = driver_.getEstimate(t);
      T_init.translation = ctx.est.x.p;
      T_init.rotation = ctx.est.x.q;
    } else {
      T_init.rotation = Eigen::Quaterniond(msg->initial_quaternion_w,
                                           msg->initial_quaternion_x,
                                           msg->initial_quaternion_y,
                                           msg->initial_quaternion_z)
                            .normalized();
    }

    // First frame: seed the local map, start the ESKF, return.
    if (!map_initialized_) {
      slam_->initialize(T_init, edge_pts, plane_pts);
      prev_pose_ = T_init;
      prev_t_ = t;
      map_initialized_ = true;

      if (!driver_.running()) {
        Eskf::Estimate post0;
        post0.x.p = T_init.translation;
        post0.x.q = T_init.rotation;
        post0.x.v.setZero();
        post0.P.setIdentity();
        driver_.reset(t, post0);
        driver_.start();
        RCLCPP_INFO(get_logger(),
                    "Map seeded and ESKF started at t=%.3f, "
                    "p=[%.2f, %.2f, %.2f]",
                    t, T_init.translation.x(), T_init.translation.y(),
                    T_init.translation.z());
      }
      return;
    }

    // Subsequent frames: ICP localization.
    const double dt = t - prev_t_;
    prev_t_ = t;

    auto result = slam_->solve(T_init, edge_pts, plane_pts, icp_max_iter_, dt,
                               prev_pose_);
    slam_->updateMap(result.accept_result, result.pose, edge_pts, plane_pts);
    prev_pose_ = result.pose;

    // Build ESKF measurement from LOAM result.
    Eskf::Measurement meas;
    meas.t = t;
    meas.data.p = result.pose.translation;
    meas.data.q = result.pose.rotation;

    // Use GN Hessian inverse as R (H⁻¹→R propagation).
    // Apply a per-element variance floor to guard against degenerate scans.
    meas.R = result.reg_err.cov;
    for (int i = 0; i < 6; ++i)
      meas.R(i, i) = std::max(meas.R(i, i), r_min_diag_(i));

    driver_.pushPose(meas);
  }

  // -------------------------------------------------------------------------

  bool map_initialized_ = false;
  std::string frame_id_ = "map";
  Eigen::Vector3d angular_velocity_ = Eigen::Vector3d::Zero();
  int icp_max_iter_ = 4;
  double prev_t_ = 0.0;
  manifold::TransformF64 prev_pose_{
      .rotation = Eigen::Quaterniond::Identity(),
      .translation = Eigen::Vector3d::Zero(),
  };

  // Minimum variance floor for each of the 6 pose DOF [x y z rx ry rz].
  Eigen::Matrix<double, 6, 1> r_min_diag_ =
      Eigen::Matrix<double, 6, 1>::Ones() * 1e-4;

  rclcpp::SubscriptionBase::SharedPtr imu_sub_;
  rclcpp::SubscriptionBase::SharedPtr feature_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr grav_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr diag_timer_;
  diagnostic_updater::Updater diag_;

  eskf::InertialOdometryDriver<Eskf> driver_;
  std::unique_ptr<loam_baseline::Localization> slam_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LoamNode>());
  rclcpp::shutdown();
  return 0;
}
