#pragma once
// Ported from arise_slam_mid360/include/arise_slam_mid360/LidarProcess/Utilities.h
// Retained: Rad2Deg, Deg2Rad, TransformPoint, TransformPointd, ComputePCA.
// Removed: ROS-specific helpers, arise_slam point-type conversions.

#include <cmath>
#include <Eigen/Eigenvalues>
#include <pcl/point_cloud.h>

#include "loam_baseline/utils/Twist.h"

// Anonymous namespace matches upstream — avoids ODR issues across TUs.
namespace {

template <typename T>
inline constexpr T Rad2Deg(const T& rad) { return rad / M_PI * 180.; }

template <typename T>
inline constexpr T Deg2Rad(const T& deg) { return deg / 180. * M_PI; }

template <typename PointT>
inline void TransformPoint(PointT& p, const Transformd& transform) {
    Eigen::Vector3d temp = p.getVector3fMap().template cast<double>();
    p.getVector3fMap() = (transform * temp).template cast<float>();
}

template <typename PointT>
inline PointT TransformPointd(const PointT& p, const Transformd& transform) {
    PointT out(p);
    TransformPoint(out, transform);
    return out;
}

inline Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
ComputePCA(const Eigen::Matrix<double, Eigen::Dynamic, 3>& data,
           Eigen::Vector3d& mean) {
    mean = data.colwise().mean();
    Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
    Eigen::Matrix3d varianceCovariance = centered.transpose() * centered;
    return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(varianceCovariance);
}

inline Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
ComputePCA(const Eigen::Matrix<double, Eigen::Dynamic, 3>& data) {
    Eigen::Vector3d mean;
    return ComputePCA(data, mean);
}

}  // anonymous namespace
