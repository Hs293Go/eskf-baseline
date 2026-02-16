import functools

import pytest
import torch

import eskf_baseline

OperatingPoints = tuple[list[eskf_baseline.NominalState], list[eskf_baseline.Input]]


@pytest.fixture
def config():
    return eskf_baseline.Config(
        grav_vector=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
    )


def _random_quaternion(
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
def operating_points() -> OperatingPoints:
    rand = functools.partial(torch.rand, dtype=torch.float32, device="cpu")
    randn = functools.partial(torch.randn, dtype=torch.float32, device="cpu")
    num_trials = 50
    ps = -10 + 20 * rand(num_trials, 3)
    qs = _random_quaternion(batch_size=num_trials, dtype=torch.float32, device="cpu")
    vs = -5 + 10 * rand(num_trials, 3)
    abiases = randn(num_trials, 3) * 0.1
    gbiases = randn(num_trials, 3) * 0.01
    # Generate random valid nominal state
    accs = randn(num_trials, 3) * 0.5 + torch.tensor(
        [0.0, 0.0, 9.81], dtype=torch.float32, device="cpu"
    )
    gyros = randn(num_trials, 3) * 0.1

    return (
        list(map(eskf_baseline.NominalState, ps, qs, vs, abiases, gbiases)),
        list(map(eskf_baseline.Input, accs, gyros)),
    )
