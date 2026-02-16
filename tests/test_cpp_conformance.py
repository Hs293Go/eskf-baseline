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
            [cpp.Input(*_convert(args)) for args in operating_points[1]],
        ]
    )


@pytest.fixture
def cpp_config(config: eskf_baseline.Config) -> cpp.Config:
    return cpp.Config(grav_vector=config.grav_vector)


def test_cpp_conformance(config, operating_points, cpp_config, cpp_operating_points):
    dt = torch.tensor(0.01, dtype=torch.float32)

    for x, u, xc, uc in zip(*operating_points, *cpp_operating_points):
        fjac, qcov = torch.compile(eskf_baseline.jacobians)(x, u, dt, config)
        jacs = cpp.compute_jacobians(xc, uc, dt, cpp_config)
        testing.assert_allclose(fjac.cpu().numpy(), jacs.fjac, rtol=1e-4, atol=1e-4)
        testing.assert_allclose(qcov.cpu().numpy(), jacs.qcov, rtol=1e-4, atol=1e-4)
