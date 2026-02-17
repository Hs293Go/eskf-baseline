import torch
from conftest import DtypeDevice, OperatingPoints

import eskf_baseline


def error_dynamics_wrapper(
    delta_x: torch.Tensor,
    x: eskf_baseline.NominalState,
    u: eskf_baseline.Input,
    dt: torch.Tensor,
    config: eskf_baseline.Config,
) -> torch.Tensor:
    # 1. Unpack error state
    out_pert = eskf_baseline.motion(x.boxplus(delta_x), u, dt, config)
    out_nom = eskf_baseline.motion(x, u, dt, config)

    return out_pert.boxminus(out_nom)


def test_jacobian_derivation(
    config: eskf_baseline.Config,
    operating_points: OperatingPoints,
) -> None:
    dt = torch.tensor(0.01, dtype=torch.float32)

    for x, u in zip(*operating_points):
        # 1. Get hand-derived Jacobian
        fjac_hand, qcov_hand = torch.compile(eskf_baseline.jacobians)(x, u, dt, config)

        # 2. Get AutoDiff Jacobian evaluated at exactly zero error
        delta_x_zero = torch.zeros(15, dtype=x.p.dtype, device=x.p.device)

        # jacrev with respect to argument 0 (delta_x)
        fjac_ad = torch.compile(torch.func.jacrev(error_dynamics_wrapper, argnums=0))(
            delta_x_zero, x, u, dt, config
        )

        torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)

        result = fjac_hand
        expected = fjac_ad
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
        )


def test_jvp(
    config: eskf_baseline.Config,
    operating_points: OperatingPoints,
    dtype_device: DtypeDevice,
) -> None:
    dt = torch.tensor(0.01, dtype=torch.float32)

    num_vecs = 10

    vs = torch.randn((len(operating_points[0]), num_vecs, 15), **dtype_device)
    for x, u, ith_v in zip(*operating_points, vs):
        # 1. Get hand-derived Jacobian
        fjac_hand, qcov_hand = torch.compile(eskf_baseline.jacobians)(x, u, dt, config)

        delta0 = torch.zeros(15, dtype=x.p.dtype, device=x.p.device)

        # Function of delta only (so jvp sees a single input)
        def g(delta: torch.Tensor) -> torch.Tensor:
            return error_dynamics_wrapper(delta, x, u, dt, config)

        # Random direction vectors
        for v in ith_v:
            # JVP via forward-mode AD
            _, jvp = torch.func.jvp(g, (delta0,), (v,))

            # Hand linearization action
            jvp_expected = fjac_hand @ v

            torch.testing.assert_close(jvp, jvp_expected, rtol=1e-4, atol=1e-4)
