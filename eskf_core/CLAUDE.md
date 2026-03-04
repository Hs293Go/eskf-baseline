# ESKF Baseline

## Overall context

We implemented a known-correct ESKF whose implementation of covariance
propagation laws is checked against automatic differentiation of the mean
propagation laws.

We are now integrating this ESKF with a general inertial odometry driver
`eskf_core/include/eskf_baseline/inertial_odometry_driver.hpp` which attempts to
achieve a reasonable baseline that addresses time-delayed measurements and
asynchronous operations. While it could not be algorithmically proven, the
inertial odometry driver is extensively tested, primarily in

```
eskf_core/tests/src/test_driver.cpp
eskf_core/tests/src/test_inertial_odometry_integration.cpp
```

## Building

This code can be built standalone and with ROS integration. To build standalone,
use the following command from the root of the repository:

```bash
cmake -S . -B build/default -DCMAKE_DISABLE_FIND_PACKAGE_ament_cmake=ON
```

where we disable the ROS build system since it is not needed for this code.

## Testing

To run the tests, use the following command from the root of the repository:

```bash
ctest --test-dir build/default/eskf_core/tests
```
