
#include "eskf_baseline/eskf.hpp"

namespace eskf {
eskf::BasicErrorContext Eskf::predict(Estimate& ctx, const Input& u,
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

eskf::BasicErrorContext Eskf::correct(Estimate& ctx,
                                      const Measurement& meas) const {
  const Eigen::Vector<double, 6> y =
      meas.data.boxminus(eskf::PoseObservation(ctx.x));
  const Eigen::Matrix<double, 6, kTangentDim> hjac =
      eskf::PoseObservationJacobian(ctx.x);
  const Eigen::Matrix<double, 6, 6> scov =
      hjac * ctx.P * hjac.transpose() + meas.R;

  auto llt_fac = scov.llt();
  const bool llt_success = llt_fac.info() == Eigen::Success;
  if (!llt_success) {
    return {
        .ec = eskf::Errc::kFatalLinalgFailure,
        .custom_message = "LLT decomposition failed for innovation covariance"};
  }

  const double mahalanobis_distance = y.dot(llt_fac.solve(y));
  const auto classification =
      outlier_classifier_.classify(mahalanobis_distance);
  if (classification == eskf::OutlierClassification::kError) {
    return {.ec = eskf::Errc::kOutlierDetected,
            .custom_message = "Fatal Outlier detected"};
  }
  auto errc = eskf::Errc::kSuccess;
  if (classification == eskf::OutlierClassification::kWarning) {
    errc = eskf::Errc::kOutlierDetected;
  }

  // Solve K.' = S.' / (P * H.').' instead of K = P * H.' \ S
  const Eigen::Matrix<double, kTangentDim, 6> kgain =
      llt_fac.solve(hjac * ctx.P.transpose()).transpose();

  Covariance i_m_km = -kgain * hjac;
  i_m_km.diagonal().array() += 1.0;
  ctx.P = EnsureSymmetric(i_m_km * ctx.P * i_m_km.transpose() +
                          kgain * meas.R * kgain.transpose());
  ctx.x = ctx.x.boxplus(kgain * y);
  ctx.x.q.normalize();

  return {.ec = errc};
}

bool Eskf::setConfig(const eskf::Config<double>& cfg) {
  if (cfg.accel_noise_density <= 0 || cfg.gyro_noise_density <= 0) {
    return false;
  }
  cfg_ = cfg;
  return true;
}
}  // namespace eskf
