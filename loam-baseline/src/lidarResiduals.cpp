// Ported from arise_slam_mid360/src/LaserMapping/lidarOptimization.cpp
// Ceres classes (EdgeAnalyticCostFunction, SurfNormAnalyticCostFunction,
// PoseSE3Parameterization) removed; math extracted as free functions in SE(3)
// tangent space.

#include "loam_baseline/lidarResiduals.hpp"

#include <cmath>

namespace loam_baseline {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m.setZero();
  m(0, 1) = -v(2);
  m(0, 2) = v(1);
  m(1, 0) = v(2);
  m(1, 2) = -v(0);
  m(2, 0) = -v(1);
  m(2, 1) = v(0);
  return m;
}

void getTransformFromSe3(const Eigen::Matrix<double, 6, 1>& se3,
                         Eigen::Quaterniond& q, Eigen::Vector3d& t) {
  // se3 = [upsilon (translation); omega (rotation)]
  const Eigen::Vector3d upsilon(se3.data());
  const Eigen::Vector3d omega(se3.data() + 3);
  const Eigen::Matrix3d Omega = skew(omega);

  const double theta = omega.norm();
  const double half_theta = 0.5 * theta;
  const double real_factor = std::cos(half_theta);
  double imag_factor;

  if (theta < 1e-10) {
    const double theta_sq = theta * theta;
    const double theta_po4 = theta_sq * theta_sq;
    imag_factor = 0.5 - 0.0208333 * theta_sq + 0.000260417 * theta_po4;
  } else {
    imag_factor = std::sin(half_theta) / theta;
  }

  q = Eigen::Quaterniond(real_factor, imag_factor * omega.x(),
                         imag_factor * omega.y(), imag_factor * omega.z());

  Eigen::Matrix3d J;
  if (theta < 1e-10) {
    J = Eigen::Matrix3d::Identity() + 0.5 * Omega;
  } else {
    const Eigen::Matrix3d Omega2 = Omega * Omega;
    J = Eigen::Matrix3d::Identity() +
        (1.0 - std::cos(theta)) / (theta * theta) * Omega +
        (theta - std::sin(theta)) / (theta * theta * theta) * Omega2;
  }

  t = J * upsilon;
}

void retractPose(Eigen::Quaterniond& q, Eigen::Vector3d& t,
                 const Eigen::Matrix<double, 6, 1>& delta) {
  Eigen::Quaterniond delta_q;
  Eigen::Vector3d delta_t;
  getTransformFromSe3(delta, delta_q, delta_t);

  q = (delta_q * q).normalized();
  t = delta_q * t + delta_t;
}

double tukeyWeight(double r, double c) {
  if (std::abs(r) >= c) return 0.0;
  const double x = r / c;
  const double tmp = 1.0 - x * x;
  return tmp * tmp;
}

void edgeResidualAndJacobian(const Eigen::Vector3d& p,
                             const Eigen::Vector3d& pa,
                             const Eigen::Vector3d& pb,
                             const Eigen::Quaterniond& q,
                             const Eigen::Vector3d& t, Eigen::Vector3d& r,
                             Eigen::Matrix<double, 3, 6>& H) {
  // Transformed query point in map frame
  const Eigen::Vector3d lp = q * p + t;

  // Cross-product distance: nu = (lp - pa) × (lp - pb), de = pa - pb
  const Eigen::Vector3d nu = (lp - pa).cross(lp - pb);
  const Eigen::Vector3d de = pa - pb;
  const double de_norm = de.norm();

  r = nu / de_norm;

  // ∂(q*p + t)/∂[δt, δω] = [I | -R·skew(p)]
  Eigen::Matrix<double, 3, 6> dp;
  dp.leftCols<3>().setIdentity();
  dp.rightCols<3>() = -q.toRotationMatrix() * skew(p);

  // J = skew(pb - pa) · dp / de_norm
  H = skew(pb - pa) * dp / de_norm;
}

void planeResidualAndJacobian(const Eigen::Vector3d& p,
                              const Eigen::Vector3d& n, double d,
                              const Eigen::Quaterniond& q,
                              const Eigen::Vector3d& t, double& r,
                              Eigen::Matrix<double, 1, 6>& H) {
  const Eigen::Vector3d point_w = q * p + t;
  r = n.dot(point_w) + d;

  Eigen::Matrix<double, 3, 6> dp;
  dp.leftCols<3>().setIdentity();
  dp.rightCols<3>() = -q.toRotationMatrix() * skew(p);

  H = n.transpose() * dp;
}

}  // namespace loam_baseline
