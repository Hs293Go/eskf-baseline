#ifndef LOAM_BASELINE_PCA_HPP_
#define LOAM_BASELINE_PCA_HPP_

#include "Eigen/Dense"

namespace loam_baseline {

template <typename T, Eigen::Index C>
struct PCAResult {
  using MatrixType = Eigen::Matrix<double, C, C>;
  Eigen::SelfAdjointEigenSolver<MatrixType> eig;
  Eigen::Vector<double, C> mean;
};

template <std::ranges::input_range R,
          typename Derived = std::ranges::range_value_t<R>,
          typename T = typename Derived::Scalar,
          Eigen::Index C = Derived::RowsAtCompileTime>
  requires(std::derived_from<Derived, Eigen::MatrixBase<Derived>>)
PCAResult<T, C> ComputePCA(R&& data) {
  Eigen::Vector<double, C> mean = Eigen::Vector<double, C>::Zero();
  for (const auto& vec : data) {
    mean += vec;
  }
  mean /= static_cast<double>(data.size());

  Eigen::Matrix<double, C, C> var_cov = Eigen::Matrix<double, C, C>::Zero();
  for (const auto& vec : data) {
    var_cov.template selfadjointView<Eigen::Lower>().rankUpdate(vec - mean);
  }
  return {Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, C, C>>(
              var_cov.template selfadjointView<Eigen::Lower>()),
          mean};
}

}  // namespace loam_baseline

#endif  // LOAM_BASELINE_PCA_HPP_
