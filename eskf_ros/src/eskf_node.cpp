#include "Eigen/Dense"
#include "boost/math/distributions/chi_squared.hpp"
#include "eskf_baseline/eskf_baseline.hpp"
#include "eskf_baseline/inertial_odometry_driver.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2_eigen/tf2_eigen.hpp"

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime,
              Derived::RowsAtCompileTime>
EnsureSymmetric(const Eigen::MatrixBase<Derived>& m) {
  return (m + m.transpose()) / 2;
}

static constexpr int kTangentDim = eskf::NominalState<double>::kTangentDim;
using Covariance = Eigen::Matrix<double, kTangentDim, kTangentDim>;

enum class OutlierClassification { kNormal, kWarning, kError };

class OutlierClassifier {
 public:
  OutlierClassifier(std::uint32_t dof, double warning_threshold_percent,
                    double error_threshold_percent)
      : warning_threshold_(boost::math::quantile(boost::math::chi_squared(dof),
                                                 warning_threshold_percent)),
        error_threshold_(boost::math::quantile(boost::math::chi_squared(dof),
                                               error_threshold_percent)) {}

  OutlierClassification classify(double mahalanobis_distance) const {
    if (mahalanobis_distance > error_threshold_) {
      return OutlierClassification::kError;
    }

    if (mahalanobis_distance > warning_threshold_) {
      return OutlierClassification::kWarning;
    }

    return OutlierClassification::kNormal;
  }

  double warning_threshold() const { return warning_threshold_; }

  double error_threshold() const { return error_threshold_; }

 private:
  double warning_threshold_ = 0.0;
  double error_threshold_ = 0.0;
};

class Eskf {
 public:
  struct Estimate {
    eskf::NominalState<double> x;
    Covariance P;
  };

  struct Input {
    double t;
    eskf::ImuInput<double> data;
  };

  struct Measurement {
    double t;
    eskf::Pose<double> data;
    Eigen::Matrix<double, 6, 6> R;
  };

  eskf::Config<double> cfg_{.accel_noise_density = 1,
                            .gyro_noise_density = 0.01};

  eskf::BasicErrorContext predict(Estimate& ctx, const Input& u,
                                  double dt) const {
    if (dt <= 0.0) {
      return {.ec = eskf::Errc::kFatalNonPositiveTimeStep,
              .custom_message = "Non-positive time delta in time update"};
    }

    const auto [F, Q] = eskf::MotionJacobians(ctx.x, u.data, dt, cfg_);
    ctx.P = EnsureSymmetric(F * ctx.P * F.transpose() + Q);
    ctx.x = eskf::Motion(ctx.x, u.data, dt);
    ctx.x.q.normalize();

    return {.ec = eskf::Errc::kSuccess};
  }

  eskf::BasicErrorContext correct(Estimate& ctx,
                                  const Measurement& meas) const {
    const Eigen::Vector<double, 6> y =
        meas.data.boxminus(eskf::PoseObservation(ctx.x));
    const Eigen::Matrix<double, 6, kTangentDim> H =
        eskf::PoseObservationJacobian(ctx.x);
    const Eigen::Matrix<double, 6, 6> S = H * ctx.P * H.transpose() + meas.R;

    auto llt_fac = S.llt();
    double mahalanobis_distance = 0;
    const bool llt_success = llt_fac.info() == Eigen::Success;
    if (llt_success) {
      mahalanobis_distance = y.dot(llt_fac.solve(y));
    } else {
      mahalanobis_distance = y.dot(
          S.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y));
    }

    const auto classification =
        outlier_classifier_.classify(mahalanobis_distance);
    if (classification == OutlierClassification::kError) {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("Eskf"),
                          "Fatal Outlier detected with Mahalanobis distance "
                              << mahalanobis_distance << " (threshold: "
                              << outlier_classifier_.error_threshold() << ")");
      return {.ec = eskf::Errc::kOutlierDetected,
              .custom_message = "Fatal Outlier detected"};
    }
    auto errc = eskf::Errc::kSuccess;
    if (classification == OutlierClassification::kWarning) {
      RCLCPP_WARN_STREAM(rclcpp::get_logger("Eskf"),
                         "Outlier detected with Mahalanobis distance "
                             << mahalanobis_distance << " (threshold: "
                             << outlier_classifier_.warning_threshold() << ")");
      errc = eskf::Errc::kOutlierDetected;
    }

    if (!llt_success) {
      return {.ec = eskf::Errc::kFatalLinalgFailure,
              .custom_message =
                  "LLT decomposition failed for innovation covariance"};
    }

    const Eigen::Matrix<double, kTangentDim, 6> PHt = ctx.P * H.transpose();
    const Eigen::Matrix<double, kTangentDim, 6> K =
        llt_fac.solve(PHt.transpose()).transpose();

    Covariance i_m_km = -K * H;
    i_m_km.diagonal().array() += 1.0;
    ctx.P = EnsureSymmetric(i_m_km * ctx.P * i_m_km.transpose() +
                            K * meas.R * K.transpose());
    ctx.x = ctx.x.boxplus(K * y);
    ctx.x.q.normalize();

    return {.ec = errc};
  }

 private:
  OutlierClassifier outlier_classifier_ = {6, 0.95, 0.99};
};

class EskfNode : public rclcpp::Node {
 public:
  EskfNode() : Node("eskf_node") {
    auto accelerometer_unit_is_g =
        declare_parameter("accelerometer_unit_is_g", false);

    RCLCPP_INFO(get_logger(), "Accelerometer unit is %s",
                accelerometer_unit_is_g ? "g" : "m/s^2");

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

          driver_.push_imu(imu);
        });
    RCLCPP_INFO(this->get_logger(), "Subscribing to IMU on: %s",
                imu_sub_->get_topic_name());

    const bool pose_format_is_odometry =
        declare_parameter("pose_format_is_odometry", false);

    if (pose_format_is_odometry) {
      odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
          "eskf/meas", 10,
          [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
            auto time = rclcpp::Time(msg->header.stamp).seconds();
            Eskf::Measurement meas = {.t = time};
            tf2::fromMsg(msg->pose.pose.position, meas.data.p);
            tf2::fromMsg(msg->pose.pose.orientation, meas.data.q);
            tryInitDriver(time, meas);
            driver_.push_pose(meas);
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
            meas.R = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;

            tryInitDriver(time, meas);
            driver_.push_pose(meas);
          });
    }
    RCLCPP_INFO(this->get_logger(),
                "Subscribing to measurement in %s format on: %s",
                pose_format_is_odometry ? "nav_msgs/Odometry"
                                        : "geometry_msgs/PoseStamped",
                odom_sub_->get_topic_name());

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("eskf/estimate", 1);

    timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() {
      if (driver_.running()) {
        auto t = get_clock()->now().seconds();
        const auto& [ctx, outcome] = driver_.getEstimate(t);
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = get_clock()->now();
        odom.header.frame_id = frame_id_;
        odom.pose.pose.position = tf2::toMsg(ctx.est.x.p);
        odom.pose.pose.orientation = tf2::toMsg(ctx.est.x.q);
        tf2::toMsg(ctx.est.x.v, odom.twist.twist.linear);
        tf2::toMsg(angular_velocity_, odom.twist.twist.angular);
        odom_pub_->publish(odom);
      }
    });
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
  eskf::InertialOdometryDriver<Eskf> driver_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EskfNode>());
  rclcpp::shutdown();
  return 0;
}
