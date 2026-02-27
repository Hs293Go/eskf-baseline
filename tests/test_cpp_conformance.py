from collections.abc import Generator, Sequence

import eskf_baseline_cpp as cpp
import numpy as np
import pytest
import torch
from conftest import OperatingPoints
from numpy import testing

import eskf_baseline


@pytest.fixture
def cpp_operating_points(operating_points: OperatingPoints):
    def _convert(args: Sequence[torch.Tensor]) -> Generator[np.ndarray]:
        return (it.cpu().numpy() for it in args)

    return tuple(
        [
            [cpp.NominalState(*_convert(args)) for args in operating_points[0]],
            [cpp.ImuInput(*_convert(args)) for args in operating_points[1]],
        ]
    )


@pytest.fixture
def cpp_config(config: eskf_baseline.Config) -> cpp.Config:
    return cpp.Config()


def test_motion_model_cpp_conformance(
    config, operating_points, cpp_config, cpp_operating_points
):
    dt = torch.tensor(0.01, dtype=torch.float32)

    for x, u, xc, uc in zip(*operating_points, *cpp_operating_points):
        x_new = torch.compile(eskf_baseline.motion)(x, u, dt, config)
        x_new_cpp = cpp.motion(xc, uc, dt, cpp_config)
        testing.assert_allclose(
            x_new.p.cpu().numpy(), x_new_cpp.p, rtol=1e-4, atol=1e-4
        )
        testing.assert_allclose(
            x_new.q.cpu().numpy(), x_new_cpp.q, rtol=1e-4, atol=1e-4
        )
        testing.assert_allclose(
            x_new.v.cpu().numpy(), x_new_cpp.v, rtol=1e-4, atol=1e-4
        )

        fjac, qcov = torch.compile(eskf_baseline.motion_jacobians)(x, u, dt, config)
        jacs = cpp.compute_jacobians(xc, uc, dt, cpp_config)
        testing.assert_allclose(fjac.cpu().numpy(), jacs.fjac, rtol=1e-4, atol=1e-4)
        testing.assert_allclose(qcov.cpu().numpy(), jacs.qcov, rtol=1e-4, atol=1e-4)


def test_pose_observation_model_cpp_conformance(
    config, operating_points, cpp_operating_points
):
    for x, _, xc, _ in zip(*operating_points, *cpp_operating_points):
        z = torch.compile(eskf_baseline.pose_observation)(x)
        z_cpp = cpp.pose_observation(xc)
        testing.assert_allclose(z.p.cpu().numpy(), z_cpp.p, rtol=1e-4, atol=1e-4)
        testing.assert_allclose(z.q.cpu().numpy(), z_cpp.q, rtol=1e-4, atol=1e-4)

        hjac = torch.compile(eskf_baseline.pose_observation_jacobian)(x)
        hjac_cpp = cpp.pose_observation_jacobian(xc)
        testing.assert_allclose(hjac.cpu().numpy(), hjac_cpp, rtol=1e-4, atol=1e-4)


def test_compass_observation_model_cpp_conformance(
    dtype_device, config, operating_points, cpp_operating_points
):
    grav_vector = torch.tensor([0.0, 0.0, 9.81], **dtype_device)
    b = torch.tensor([0.2, 0.3, 0.4], **dtype_device)  # Example magnetic field vector
    b -= b.dot(grav_vector) / grav_vector.norm() ** 2 * grav_vector

    for x, _, xc, _ in zip(*operating_points, *cpp_operating_points):
        z = torch.compile(eskf_baseline.compass_observation)(x, b)
        z_cpp = cpp.compass_observation(xc, b.cpu().numpy())
        testing.assert_allclose(z.b.cpu().numpy(), z_cpp.b, rtol=1e-4, atol=1e-4)

        hjac = torch.compile(eskf_baseline.compass_observation_jacobian)(x, b)
        hjac_cpp = cpp.compass_observation_jacobian(xc, b.cpu().numpy())
        testing.assert_allclose(hjac.cpu().numpy(), hjac_cpp, rtol=1e-4, atol=1e-4)
