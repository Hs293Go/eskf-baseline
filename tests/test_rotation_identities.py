import pytest
import torch
from conftest import random_quaternion

from eskf_baseline import (
    angle_axis_to_quaternion,
    inversely_rotated_vector_by_perturbation_jacobian,
    quaternion_product,
    quaternion_rotate_point,
    rotated_vector_by_perturbation_jacobian,
)


@pytest.fixture
def test_data():
    NUM_TESTS = 500
    q = random_quaternion(NUM_TESTS)
    v = torch.randn(NUM_TESTS, 3)
    return q, v


def test_differentiate_rotated_vector_by_perturbation(
    test_data: tuple[torch.Tensor, torch.Tensor],
):
    q, v = test_data

    def error_fn(error: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        perturbed_q = quaternion_product(q, angle_axis_to_quaternion(error))
        perturbed = quaternion_rotate_point(perturbed_q, v)
        return perturbed

    delta_x_zero = torch.zeros([len(q), 3], dtype=q.dtype, device=q.device)
    result = torch.vmap(torch.func.jacrev(error_fn, argnums=0))(delta_x_zero, q, v)
    expected = torch.vmap(rotated_vector_by_perturbation_jacobian)(q, v)

    assert torch.allclose(result, expected, atol=1e-5)


def test_differentiate_inversely_rotated_vector_by_perturbation(
    test_data: tuple[torch.Tensor, torch.Tensor],
):
    q, v = test_data

    def error_fn(error: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        perturbed_q = quaternion_product(q, angle_axis_to_quaternion(error))
        perturbed = quaternion_rotate_point(perturbed_q, v, invert=True)
        return perturbed

    delta_x_zero = torch.zeros([len(q), 3], dtype=q.dtype, device=q.device)
    result = torch.vmap(torch.func.jacrev(error_fn, argnums=0))(delta_x_zero, q, v)
    expected = torch.vmap(inversely_rotated_vector_by_perturbation_jacobian)(q, v)

    print(result[0, ...], expected[0, ...])
    assert torch.allclose(result, expected, atol=1e-5)
