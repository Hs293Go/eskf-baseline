#pragma once
// Ported from arise_slam_mid360/LidarProcess/factor/lidarOptimization.h
// Ceres base classes removed; residuals and Jacobians expressed as free
// functions in SE(3) tangent space (translation-first 6-vector convention:
// [δt | δω], matching the original J layout).

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace loam_baseline {

// ---------------------------------------------------------------------------
// SO(3) / SE(3) utilities
// ---------------------------------------------------------------------------

/// Skew-symmetric matrix of a 3-vector.
Eigen::Matrix3d skew(const Eigen::Vector3d& v);

/// SO(3) exponential map + left Jacobian for a 6D se(3) tangent vector
/// [upsilon (translation part) | omega (rotation part)].
/// Computes the quaternion q and translation t such that
///   exp([upsilon; omega]) = (q, t).
void getTransformFromSe3(const Eigen::Matrix<double, 6, 1>& se3,
                          Eigen::Quaterniond& q, Eigen::Vector3d& t);

/// Apply a 6D tangent-space delta to (q, t) in place.
///   q ← deltaQ(δω) * q   (normalised)
///   t ← deltaQ(δω) * t + δt
void retractPose(Eigen::Quaterniond& q, Eigen::Vector3d& t,
                 const Eigen::Matrix<double, 6, 1>& delta);

// ---------------------------------------------------------------------------
// Robust loss
// ---------------------------------------------------------------------------

/// Tukey robust weight: (1 − (r/c)²)² for |r| < c, else 0.
double tukeyWeight(double r, double c);

// ---------------------------------------------------------------------------
// Residuals and Jacobians
// ---------------------------------------------------------------------------

/// Point-to-line residual and 3×6 Jacobian in SE(3) tangent space.
///
/// @param p    Query point in sensor frame.
/// @param pa   First map point defining the target edge line.
/// @param pb   Second map point defining the target edge line.
/// @param q    Current rotation estimate (world←sensor).
/// @param t    Current translation estimate (world←sensor).
/// @param r    Output 3×1 residual  (cross-product distance / |pa-pb|).
/// @param H    Output 3×6 Jacobian  ∂r/∂[δt, δω].
void edgeResidualAndJacobian(const Eigen::Vector3d& p,
                              const Eigen::Vector3d& pa,
                              const Eigen::Vector3d& pb,
                              const Eigen::Quaterniond& q,
                              const Eigen::Vector3d& t,
                              Eigen::Vector3d& r,
                              Eigen::Matrix<double, 3, 6>& H);

/// Point-to-plane residual and 1×6 Jacobian in SE(3) tangent space.
///
/// @param p    Query point in sensor frame.
/// @param n    Unit plane normal in map frame.
/// @param d    Scalar offset: −(n · A) where A is any point on the plane.
/// @param q    Current rotation estimate (world←sensor).
/// @param t    Current translation estimate (world←sensor).
/// @param r    Output scalar residual  n·(q*p + t) + d.
/// @param H    Output 1×6 Jacobian  ∂r/∂[δt, δω].
void planeResidualAndJacobian(const Eigen::Vector3d& p,
                               const Eigen::Vector3d& n, double d,
                               const Eigen::Quaterniond& q,
                               const Eigen::Vector3d& t,
                               double& r,
                               Eigen::Matrix<double, 1, 6>& H);

}  // namespace loam_baseline
