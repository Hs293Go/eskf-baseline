#include "eskf_baseline/eskf_baseline.hpp"

#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace pybind11::literals;

// Helper to bind the templated logic for a specific scalar type
template <typename T>
void declare_eskf_types(py::module& m, const std::string& suffix) {
  using Scalar = T;
  using eskf::ComputeJacobians;
  using eskf::Config;
  using eskf::Input;
  using eskf::Jacobians;
  using eskf::kinematics;
  using eskf::NominalState;

  // 1. Bind Structs
  py::class_<NominalState<Scalar>>(m, ("NominalState" + suffix).c_str())
      .def(py::init([](const Eigen::Vector3<Scalar>& p,
                       const Eigen::Vector4<Scalar>& q_vec,  // Accept Vector4
                       const Eigen::Vector3<Scalar>& v,
                       const Eigen::Vector3<Scalar>& accel_bias,
                       const Eigen::Vector3<Scalar>& gyro_bias) {
             // Convert Vector4 [x, y, z, w] to Quaternion
             Eigen::Quaternion<Scalar> q(q_vec);
             return new NominalState<Scalar>{p, q, v, accel_bias, gyro_bias};
           }),
           "p"_a = Eigen::Vector3<Scalar>::Zero(),
           "q"_a = Eigen::Vector4<Scalar>::UnitW(),
           "v"_a = Eigen::Vector3<Scalar>::Zero(),
           "accel_bias"_a = Eigen::Vector3<Scalar>::Zero(),
           "gyro_bias"_a = Eigen::Vector3<Scalar>::Zero())
      .def_readwrite("p", &NominalState<Scalar>::p)
      .def_property(
          "q", [](const NominalState<Scalar>& self) { return self.q.coeffs(); },
          [](NominalState<Scalar>& self, const Eigen::Vector3<Scalar>& v) {
            self.q = Eigen::Quaternion<Scalar>(v[3], v[0], v[1], v[2]);
          })
      .def_readwrite("v", &NominalState<Scalar>::v)
      .def_readwrite("accel_bias", &NominalState<Scalar>::accel_bias)
      .def_readwrite("gyro_bias", &NominalState<Scalar>::gyro_bias);

  py::class_<Input<Scalar>>(m, ("Input" + suffix).c_str())
      .def(py::init<Eigen::Vector3<Scalar>, Eigen::Vector3<Scalar>>(),
           "accel"_a = Eigen::Vector3<Scalar>::Zero(),
           "gyro"_a = Eigen::Vector3<Scalar>::Zero())
      .def_readwrite("accel", &Input<Scalar>::accel)
      .def_readwrite("gyro", &Input<Scalar>::gyro);

  py::class_<Config<Scalar>>(m, ("Config" + suffix).c_str())
      .def(py::init<Scalar, Scalar, Scalar, Scalar, Eigen::Vector3<Scalar>>(),
           "accel_noise_density"_a = static_cast<Scalar>(0.1),
           "gyro_noise_density"_a = static_cast<Scalar>(0.01),
           "accel_bias_random_walk"_a = static_cast<Scalar>(0.001),
           "gyro_bias_random_walk"_a = static_cast<Scalar>(0.0001),
           "grav_vector"_a = Eigen::Vector3<Scalar>{0, 0, -9.81})
      .def_readwrite("accel_noise_density",
                     &Config<Scalar>::accel_noise_density)
      .def_readwrite("gyro_noise_density", &Config<Scalar>::gyro_noise_density)
      .def_readwrite("accel_bias_random_walk",
                     &Config<Scalar>::accel_bias_random_walk)
      .def_readwrite("gyro_bias_random_walk",
                     &Config<Scalar>::gyro_bias_random_walk)
      .def_readwrite("grav_vector", &Config<Scalar>::grav_vector);

  py::class_<Jacobians<Scalar>>(m, ("Jacobians" + suffix).c_str())
      .def_readwrite("fjac", &Jacobians<Scalar>::fjac)
      .def_readwrite("qcov", &Jacobians<Scalar>::qcov);

  // 2. Bind Functions
  m.def("kinematics", &kinematics<Scalar>, py::arg("state"), py::arg("input"),
        py::arg("dt"), py::arg("cfg") = Config<Scalar>{});

  m.def("compute_jacobians", &ComputeJacobians<Scalar>, py::arg("state"),
        py::arg("input"), py::arg("dt"), py::arg("cfg") = Config<Scalar>{});
}

#ifndef PYBIND11_MODULE_NAME
#warning "PYBIND11_MODULE_NAME not defined, defaulting to 'eskf_baseline_cpp'"
#define PYBIND11_MODULE_NAME eskf_baseline_cpp
#endif

PYBIND11_MODULE(PYBIND11_MODULE_NAME, m) {
  m.doc() = "ESKF Math Bindings";

  // Bind for double (standard)
  declare_eskf_types<double>(m, "");
}
