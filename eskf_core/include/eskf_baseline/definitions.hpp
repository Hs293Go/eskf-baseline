#ifndef ESKF_BASELINE_DEFINITIONS_HPP_
#define ESKF_BASELINE_DEFINITIONS_HPP_

#include <string_view>

namespace eskf {
enum class Errc {
  kSuccess,
  kNoop,
  // Future proof: Distinguish between fatal and non fatal outliers
  kTimeStepTooLarge,  // anticipate
  kImuBufferOverflow,
  kOutlierDetected,

  kFatalNonPositiveTimeStep,
  kFatalOutlierDetected,
  kFatalLinalgFailure,
  kFatalNonFiniteState,
  kFailure,
  kUnknown,
};

constexpr bool IsSuccess(Errc e) { return e == Errc::kSuccess; }

constexpr bool IsReject(Errc e) {
  switch (e) {
    case Errc::kTimeStepTooLarge:
      [[fallthrough]];
    case Errc::kOutlierDetected:
      return true;
    default:
      return false;
  }
}

constexpr bool IsFatal(Errc e) {
  switch (e) {
    case Errc::kFatalOutlierDetected:
      [[fallthrough]];
    case Errc::kFatalLinalgFailure:
      [[fallthrough]];
    case Errc::kFatalNonPositiveTimeStep:
      [[fallthrough]];
    case Errc::kFatalNonFiniteState:
      return true;
    default:
      return false;
  }
}

struct BasicErrorContext {
  Errc ec = Errc::kUnknown;         // Success must be explicitly set
  std::string_view custom_message;  // Static strings only

  std::string_view message() const noexcept { return custom_message; }
  Errc errorCode() const noexcept { return ec; }
};

template <typename R>
concept ErrorContext = requires(const R& r) {
  { r.message() } noexcept -> std::convertible_to<std::string_view>;
  { r.errorCode() } noexcept -> std::same_as<Errc>;
};

static_assert(ErrorContext<BasicErrorContext>,
              "SimpleErrorContext should satisfy ErrorContext");

template <typename T>
concept TimeStamped = requires(T t) {
  { t.t } -> std::convertible_to<double>;
};

namespace utils {

template <typename It>
It prevOrBegin(It it, It begin) {
  // Use a cleverer branchless version in the helper
  return std::prev(it, static_cast<int>(it != begin));
}

template <typename F>
struct ScopeGuard {
  ScopeGuard(F&& f) : func_(std::forward<F>(f)) {}

  ~ScopeGuard() { func_(); }

 private:
  F func_;
};
}  // namespace utils

}  // namespace eskf

#endif  // ESKF_BASELINE_DEFINITIONS_HPP_
