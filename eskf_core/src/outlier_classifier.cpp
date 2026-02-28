#include "eskf_baseline/outlier_classifier.hpp"

#include "boost/math/distributions/chi_squared.hpp"

namespace eskf {

OutlierClassifier::OutlierClassifier(std::uint32_t dof,
                                     double warning_threshold_percent,
                                     double error_threshold_percent)
    : warning_threshold_(boost::math::quantile(boost::math::chi_squared(dof),
                                               warning_threshold_percent)),
      error_threshold_(boost::math::quantile(boost::math::chi_squared(dof),
                                             error_threshold_percent)) {}

OutlierClassification OutlierClassifier::classify(
    double mahalanobis_distance) const {
  if (mahalanobis_distance > error_threshold_) {
    return OutlierClassification::kError;
  }

  if (mahalanobis_distance > warning_threshold_) {
    return OutlierClassification::kWarning;
  }

  return OutlierClassification::kNormal;
}

}  // namespace eskf
