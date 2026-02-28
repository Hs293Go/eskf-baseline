#ifndef ESKF_BASELINE_OUTLIER_CLASSIFIER_HPP_
#define ESKF_BASELINE_OUTLIER_CLASSIFIER_HPP_

#include <cstdint>

namespace eskf {

enum class OutlierClassification { kNormal, kWarning, kError };

class OutlierClassifier {
 public:
  OutlierClassifier(std::uint32_t dof, double warning_threshold_percent,
                    double error_threshold_percent);

  OutlierClassification classify(double mahalanobis_distance) const;

  double warning_threshold() const { return warning_threshold_; }

  double error_threshold() const { return error_threshold_; }

 private:
  double warning_threshold_ = 0.0;
  double error_threshold_ = 0.0;
};

}  // namespace eskf

#endif  // ESKF_BASELINE_OUTLIER_CLASSIFIER_HPP_
