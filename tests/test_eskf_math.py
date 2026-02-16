import functools

import pytest
import torch

import eskf_baseline


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
def operating_points() -> (
    tuple[list[eskf_baseline.NominalState], list[eskf_baseline.Input]]
):
    rand = functools.partial(torch.rand, dtype=torch.float32, device="cpu")
    randn = functools.partial(torch.randn, dtype=torch.float32, device="cpu")
    num_trials = 50
    ps = -10 + 20 * rand(num_trials, 3)
    qs = random_quaternion(batch_size=num_trials, dtype=torch.float32, device="cpu")
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


def error_dynamics_wrapper(
    delta_x: torch.Tensor,
    x: eskf_baseline.NominalState,
    u: eskf_baseline.Input,
    dt: torch.Tensor,
    config: eskf_baseline.Config,
) -> torch.Tensor:
    # 1. Unpack error state
    dp, dth, dv, dba, dbg = torch.split(delta_x, 3)

    # 2. Inject error (Retraction / ⊕)
    # Perturb the nominal state

    x_pert = x.perturb(delta_x)

    # 3. Propagate Perturbed & Nominal States
    out_pert = eskf_baseline.kinematics(x_pert, u, dt, config)
    out_nom = eskf_baseline.kinematics(x, u, dt, config)

    p_next_pert, q_next_pert, v_next_pert, *_ = out_pert
    p_next_nom, q_next_nom, v_next_nom, *_ = out_nom

    # 4. Extract Error (Local Coordinates / ⊖)
    dp_next = p_next_pert - p_next_nom
    dth_next = eskf_baseline.quaternion_to_angle_axis(
        eskf_baseline.quaternion_product(
            eskf_baseline.quaternion_inverse(q_next_nom), q_next_pert
        )
    )
    dv_next = v_next_pert - v_next_nom

    # Biases are deterministically constant in process model
    dba_next = dba
    dbg_next = dbg

    return torch.concat([dp_next, dth_next, dv_next, dba_next, dbg_next])


def test_jacobian_derivation(
    operating_points: tuple[
        list[eskf_baseline.NominalState], list[eskf_baseline.Input]
    ],
) -> None:
    config = eskf_baseline.Config(
        grav_vector=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
    )
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
    operating_points: tuple[
        list[eskf_baseline.NominalState], list[eskf_baseline.Input]
    ],
) -> None:
    config = eskf_baseline.Config(
        grav_vector=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
    )
    dt = torch.tensor(0.01, dtype=torch.float32)

    num_vecs = 10
    with torch.autograd.set_detect_anomaly(True):
        for x, u in zip(*operating_points):
            # 1. Get hand-derived Jacobian
            fjac_hand, qcov_hand = torch.compile(eskf_baseline.jacobians)(
                x, u, dt, config
            )

            delta0 = torch.zeros(15, dtype=x.p.dtype, device=x.p.device)

            # Function of delta only (so jvp sees a single input)
            def g(delta: torch.Tensor) -> torch.Tensor:
                return error_dynamics_wrapper(delta, x, u, dt, config)

            # Random direction vectors
            for _ in range(num_vecs):
                v = torch.randn(15, dtype=x.p.dtype, device=x.p.device)

                # JVP via forward-mode AD
                _, jvp = torch.func.jvp(g, (delta0,), (v,))

                # Hand linearization action
                jvp_expected = fjac_hand @ v

                torch.testing.assert_close(jvp, jvp_expected, rtol=1e-4, atol=1e-4)
