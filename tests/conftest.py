import functools
from typing import TypedDict

import pytest
import torch

import eskf_baseline

OperatingPoints = tuple[
    list[eskf_baseline.NominalState],
    list[eskf_baseline.ImuInput],
]

DtypeDevice = TypedDict("DtypeDevice", {"dtype": torch.dtype, "device": str})


@pytest.fixture
def config(dtype_device: DtypeDevice) -> eskf_baseline.Config:
    return eskf_baseline.Config(
        grav_vector=torch.as_tensor([0.0, 0.0, -9.81], **dtype_device)
    )


@pytest.fixture
def dtype_device() -> DtypeDevice:
    return {"dtype": torch.float32, "device": "cpu"}


def random_quaternion(
    batch_size: int = 1, dtype=torch.float32, device="cpu"
) -> torch.Tensor:
    u1 = torch.rand(batch_size, dtype=dtype, device=device)
    u2 = torch.rand(batch_size, dtype=dtype, device=device) * 2 * torch.pi
    u3 = torch.rand(batch_size, dtype=dtype, device=device) * 2 * torch.pi
    a = torch.sqrt(1 - u1)
    b = torch.sqrt(u1)
    q = torch.empty((batch_size, 4), dtype=dtype, device=device)
    q[:, 0] = a * torch.cos(u2)
    q[:, 1] = b * torch.sin(u3)
    q[:, 2] = b * torch.cos(u3)
    q[:, 3] = a * torch.sin(u2)
    return q


@pytest.fixture
def operating_points(dtype_device: DtypeDevice) -> OperatingPoints:
    rand = functools.partial(torch.rand, **dtype_device)
    randn = functools.partial(torch.randn, **dtype_device)
    num_trials = 50
    ps = -10 + 20 * rand(num_trials, 3)
    qs = random_quaternion(batch_size=num_trials, **dtype_device)
    vs = -5 + 10 * rand(num_trials, 3)
    abiases = randn(num_trials, 3) * 0.1
    gbiases = randn(num_trials, 3) * 0.01
    # Generate random valid nominal state
    accs = randn(num_trials, 3) * 0.5 + torch.tensor([0.0, 0.0, 9.81], **dtype_device)
    gyros = randn(num_trials, 3) * 0.1

    return (
        list(map(eskf_baseline.NominalState, ps, qs, vs, abiases, gbiases)),
        list(map(eskf_baseline.ImuInput, accs, gyros)),
    )
