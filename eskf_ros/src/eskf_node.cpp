#include "Eigen/Dense"
#include "diagnostic_updater/diagnostic_updater.hpp"
#include "eskf_baseline/eskf.hpp"
#include "eskf_baseline/eskf_baseline.hpp"
#include "eskf_baseline/inertial_odometry_driver.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2_eigen/tf2_eigen.hpp"

using eskf::Eskf;

template <typename T>
T deg2rad(T deg) noexcept {
  return deg * std::numbers::pi_v<T> / T(180);
}

template <typename T>
T sq(T x) noexcept {
  return x * x;
}

class EskfNode : public rclcpp::Node {
 public:
  EskfNode() : Node("eskf_node"), diag_(this) {
    auto accelerometer_unit_is_g =
        declare_parameter("accelerometer_unit_is_g", false);

    RCLCPP_INFO(get_logger(), "Accelerometer unit is %s",
                accelerometer_unit_is_g ? "g" : "m/s^2");

    eskf::Config<double> cfg;
    cfg.accel_noise_density = declare_parameter("accel_noise_density", 1.0);
    cfg.gyro_noise_density = declare_parameter("gyro_noise_density", 0.01);
    if (!driver_.algorithm().setConfig(cfg)) {
      throw std::runtime_error("Invalid ESKF config parameters");
    }
    RCLCPP_INFO(get_logger(),
                "Using Eskf with accel_noise_density=%.3f and "
                "gyro_noise_density=%.3f",
                cfg.accel_noise_density, cfg.gyro_noise_density);

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "eskf/imu", 10,
        [this,
         accelerometer_unit_is_g](const sensor_msgs::msg::Imu::SharedPtr msg) {
          Eskf::Input imu = {.t = rclcpp::Time(msg->header.stamp).seconds()};

          tf2::fromMsg(msg->linear_acceleration, imu.data.accel);
          tf2::fromMsg(msg->angular_velocity, imu.data.gyro);
          angular_velocity_ = imu.data.gyro;

          if (accelerometer_unit_is_g) {
            imu.data.accel *= 9.81;
          }

          driver_.pushImu(imu);
        });
    RCLCPP_INFO(this->get_logger(), "Subscribing to IMU on: %s",
                imu_sub_->get_topic_name());

    const bool pose_format_is_odometry =
        declare_parameter("pose_format_is_odometry", false);

    double horz_position_meas_stddev =
        declare_parameter("horz_position_meas_stddev", 0.1);
    double vert_position_meas_stddev =
        declare_parameter("vert_position_meas_stddev", 0.1);
    double orientation_meas_stddev_deg =
        declare_parameter("orientation_meas_stddev_deg", 5.0);

    // Remember to capture this matrix by copying in the callback lambda!!!
    rcov_.diagonal() << sq(horz_position_meas_stddev),
        sq(horz_position_meas_stddev), sq(vert_position_meas_stddev),
        sq(deg2rad(orientation_meas_stddev_deg)),
        sq(deg2rad(orientation_meas_stddev_deg)),
        sq(deg2rad(orientation_meas_stddev_deg));

    RCLCPP_INFO(
        get_logger(),
        "Measurement noise stddev: horz_pos=%.3f m, vert_pos=%.3f m, "
        "orientation=%.3f deg; R matrix diagonal: [%f, %f, %f, %f, %f, %f]",
        horz_position_meas_stddev, vert_position_meas_stddev,
        orientation_meas_stddev_deg, rcov_(0, 0), rcov_(1, 1), rcov_(2, 2),
        rcov_(3, 3), rcov_(4, 4), rcov_(5, 5));

    if (pose_format_is_odometry) {
      odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
          "eskf/meas", 10,
          [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
            auto time = rclcpp::Time(msg->header.stamp).seconds();
            Eskf::Measurement meas = {.t = time};
            tf2::fromMsg(msg->pose.pose.position, meas.data.p);
            tf2::fromMsg(msg->pose.pose.orientation, meas.data.q);
            frame_id_ = msg->header.frame_id;
            meas.R = rcov_;

            tryInitDriver(time, meas);
            driver_.pushPose(meas);
          });
    } else {
      odom_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
          "eskf/meas", 10,
          [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            auto time = rclcpp::Time(msg->header.stamp).seconds();
            Eskf::Measurement meas = {.t = time};
            tf2::fromMsg(msg->pose.position, meas.data.p);
            tf2::fromMsg(msg->pose.orientation, meas.data.q);
            frame_id_ = msg->header.frame_id;
            meas.R = rcov_;

            tryInitDriver(time, meas);
            driver_.pushPose(meas);
          });
    }
    RCLCPP_INFO(this->get_logger(),
                "Subscribing to measurement in %s format on: %s",
                pose_format_is_odometry ? "nav_msgs/Odometry"
                                        : "geometry_msgs/PoseStamped",
                odom_sub_->get_topic_name());

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("eskf/estimate", 1);
    grav_pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>(
        "eskf/estimate_gravity", 1);

    timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() {
      if (driver_.running()) {
        auto now = get_clock()->now();
        auto t = now.seconds();
        const auto& [ctx, outcome] = driver_.getEstimate(t);
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
      }
    });

    diag_.setHardwareID("eskf_baseline");

    bool verbose_diagnostics = declare_parameter("verbose_diagnostics", false);
    double verbose_diagnostics_hz =
        declare_parameter("verbose_diagnostics_hz", 0.5);
    if (verbose_diagnostics) {
      RCLCPP_INFO(get_logger(), "Verbose diagnostics enabled at %.2f Hz",
                  verbose_diagnostics_hz);
    } else {
      RCLCPP_INFO(get_logger(), "Verbose diagnostics disabled");
    }
    diag_.add(
        "eskf/driver", [this, verbose_diagnostics, verbose_diagnostics_hz](
                           diagnostic_updater::DiagnosticStatusWrapper& stat) {
          const auto ws = driver_.getWindowStats();
          const auto st = driver_.status();

          if (verbose_diagnostics && driver_.running()) {
            RCLCPP_INFO_THROTTLE(
                get_logger(), *get_clock(), 1000.0 / verbose_diagnostics_hz,
                "proc=%.2f Hz, pred=%.2f Hz (fail %.2f Hz), corr=%.2f Hz "
                "(reject %.2f Hz), "
                "cpu(proc/pred/corr/reb)=%.3f/%.3f/%.3f/%.3f, mean(us) "
                "pred/corr/proc=%.1f/%.1f/%.1f",
                ws.process_hz, ws.predict_ok_hz, ws.predict_fail_hz,
                ws.correct_ok_hz, ws.correct_reject_hz, ws.process_cpu,
                ws.predict_cpu, ws.correct_cpu, ws.rebuild_cpu,
                ws.mean_predict_us, ws.mean_correct_us, ws.mean_process_us);
          }

          if (driver_.halted()) {
            const auto h = driver_.getHaltedOutcome();
            stat.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                         "HALTED: predict failure (or fatal)");
            stat.add("halted_t", h.t);
            stat.add("halted_reason", static_cast<int>(h.status));
            stat.add("halted_msg", std::string(h.message));
          } else {
            // Pick your own thresholds later; these are conservative
            // defaults.
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

          // Window stats
          stat.add("window_s", ws.window_s);
          stat.add("sample_age_s", ws.sample_age_s);

          stat.add("process_hz", ws.process_hz);
          stat.add("predict_ok_hz", ws.predict_ok_hz);
          stat.add("predict_fail_hz", ws.predict_fail_hz);
          stat.add("correct_ok_hz", ws.correct_ok_hz);
          stat.add("correct_reject_hz", ws.correct_reject_hz);

          stat.add("cpu_process", ws.process_cpu);
          stat.add("cpu_predict", ws.predict_cpu);
          stat.add("cpu_correct", ws.correct_cpu);
          stat.add("cpu_rebuild", ws.rebuild_cpu);

          stat.add("mean_predict_us", ws.mean_predict_us);
          stat.add("mean_correct_us", ws.mean_correct_us);
          stat.add("mean_process_us", ws.mean_process_us);

          // Staleness
          stat.add("imu_head_t", st.imu_head_t);
          stat.add("post_t", st.post_t);
          stat.add("processed_up_to_t", st.processed_up_to_t);
          stat.add("imu_lag", st.imu_lag);
          stat.add("meas_lag", st.meas_lag);
          stat.add("rebuilding", st.rebuilding);
          stat.add("trigger_age", st.trigger_age);

          // Last outcomes
          const auto lp = driver_.last_predict_outcome();
          stat.add("last_predict_t", lp.t);
          stat.add("last_predict_code", static_cast<int>(lp.status));
          stat.add("last_predict_msg", std::string(lp.message));

          const auto lc = driver_.last_correct_outcome();
          stat.add("last_correct_t", lc.t);
          stat.add("last_correct_code", static_cast<int>(lc.status));
          stat.add("last_correct_msg", std::string(lc.message));
        });

    // Publish diagnostics at a fixed rate (you said “yes” to fixing a rate).
    const double diag_hz = declare_parameter("diag_hz", 1.0);
    diag_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / std::max(0.1, diag_hz)),
        [this]() { diag_.force_update(); });
  }

  void tryInitDriver(double time, const Eskf::Measurement& meas0) {
    if (!driver_.running()) {
      RCLCPP_INFO(get_logger(),
                  "Received first odometry measurement; resetting pose to "
                  "Pose(position=[%f, %f, %f], orientation=[%f, %f, %f, %f]) "
                  "and starting driver.",
                  meas0.data.p.x(), meas0.data.p.y(), meas0.data.p.z(),
                  meas0.data.q.x(), meas0.data.q.y(), meas0.data.q.z(),
                  meas0.data.q.w());
      Eskf::Estimate post0;
      post0.x.p = meas0.data.p;
      post0.x.q = meas0.data.q;
      post0.x.v.setZero();
      post0.P.setIdentity();
      driver_.reset(time, post0);
      driver_.start();
    }
  }

 private:
  std::string frame_id_ = "map";
  rclcpp::SubscriptionBase::SharedPtr imu_sub_;
  Eigen::Vector3d angular_velocity_ = Eigen::Vector3d::Zero();
  rclcpp::SubscriptionBase::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr grav_pub_;
  eskf::InertialOdometryDriver<Eskf> driver_;
  rclcpp::TimerBase::SharedPtr timer_;
  diagnostic_updater::Updater diag_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  Eigen::Matrix<double, 6, 6> rcov_ = Eigen::Matrix<double, 6, 6>::Identity();
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EskfNode>());
  rclcpp::shutdown();
  return 0;
}
