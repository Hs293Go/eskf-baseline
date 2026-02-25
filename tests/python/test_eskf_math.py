import torch
from conftest import DtypeDevice, OperatingPoints

import eskf_baseline


def error_dynamics_wrapper(
    delta_x: torch.Tensor,
    x: eskf_baseline.NominalState,
    u: eskf_baseline.ImuInput,
    dt: torch.Tensor,
    config: eskf_baseline.Config,
) -> torch.Tensor:
    # 1. Unpack error state
    out_pert = eskf_baseline.motion(x.boxplus(delta_x), u, dt, config)
    out_nom = eskf_baseline.motion(x, u, dt, config)

    return out_pert.boxminus(out_nom)


def test_motion_jacobian(
    config: eskf_baseline.Config,
    operating_points: OperatingPoints,
    dtype_device: DtypeDevice,
) -> None:
    dt = torch.tensor(0.01, dtype=torch.float32)

    num_vecs = 10

    vs = torch.randn(
        (len(operating_points[0]), num_vecs, eskf_baseline.NominalState.TANGENT_DIM),
        **dtype_device,
    )

    for x, u, vb in zip(*operating_points, vs):
        # 1. Get hand-derived Jacobian
        fjac_hand, qcov_hand = torch.compile(eskf_baseline.motion_jacobians)(
            x, u, dt, config
        )

        # 2. Get AutoDiff Jacobian evaluated at exactly zero error
        delta_x_zero = torch.zeros(
            eskf_baseline.NominalState.TANGENT_DIM, dtype=x.p.dtype, device=x.p.device
        )

        # jacrev with respect to argument 0 (delta_x)
        fjac_ad = torch.compile(torch.func.jacrev(error_dynamics_wrapper, argnums=0))(
            delta_x_zero, x, u, dt, config
        )

        torch.testing.assert_close(fjac_hand, fjac_ad, rtol=1e-4, atol=1e-4)

        delta0 = torch.zeros(
            eskf_baseline.NominalState.TANGENT_DIM, dtype=x.p.dtype, device=x.p.device
        )

        # Function of delta only (so jvp sees a single input)
        def g(delta: torch.Tensor) -> torch.Tensor:
            return error_dynamics_wrapper(delta, x, u, dt, config)

        # Random direction vectors
        for v in vb:
            # JVP via forward-mode AD
            _, jvp = torch.func.jvp(g, (delta0,), (v,))

            # Hand linearization action
            jvp_expected = fjac_hand @ v

            torch.testing.assert_close(jvp, jvp_expected, rtol=1e-4, atol=1e-4)


def pose_observation_wrapper(
    delta_x: torch.Tensor,
    x: eskf_baseline.NominalState,
):
    # 1. Unpack error state
    obs_pert = eskf_baseline.pose_observation(x.boxplus(delta_x))
    obs_nom = eskf_baseline.pose_observation(x)
    return obs_pert.boxminus(obs_nom)


def test_pose_observation_jacobian(
    operating_points: OperatingPoints,
    dtype_device: DtypeDevice,
) -> None:
    num_vecs = 10
    vs = torch.randn(
        (len(operating_points[0]), num_vecs, eskf_baseline.NominalState.TANGENT_DIM),
        **dtype_device,
    )

    for x, _, vb in zip(*operating_points, vs):
        # 1. Get hand-derived Jacobian
        obs_jac_hand = torch.compile(eskf_baseline.pose_observation_jacobian)(x)

        # 2. Get AutoDiff Jacobian evaluated at exactly zero error
        delta_x_zero = torch.zeros(
            eskf_baseline.NominalState.TANGENT_DIM, dtype=x.p.dtype, device=x.p.device
        )

        # jacrev with respect to argument 0 (delta_x)
        obs_jac_ad = torch.compile(
            torch.func.jacrev(pose_observation_wrapper, argnums=0)
        )(delta_x_zero, x)

        torch.testing.assert_close(obs_jac_hand, obs_jac_ad, rtol=1e-4, atol=1e-4)

        delta0 = torch.zeros(
            eskf_baseline.NominalState.TANGENT_DIM, dtype=x.p.dtype, device=x.p.device
        )

        # Function of delta only (so jvp sees a single input)
        def g(delta: torch.Tensor) -> torch.Tensor:
            return pose_observation_wrapper(delta, x)

        # Random direction vectors
        for v in vb:
            # JVP via forward-mode AD
            _, jvp = torch.func.jvp(g, (delta0,), (v,))

            # Hand linearization action
            jvp_expected = obs_jac_hand @ v

            torch.testing.assert_close(jvp, jvp_expected, rtol=1e-4, atol=1e-4)


def compass_observation_wrapper(
    delta_x: torch.Tensor,
    x: eskf_baseline.NominalState,
    b: torch.Tensor,
):
    # 1. Unpack error state
    obs_pert = eskf_baseline.compass_observation(x.boxplus(delta_x), b)
    obs_nom = eskf_baseline.compass_observation(x, b)
    return obs_pert.boxminus(obs_nom)


def test_compass_observation_jacobian(
    dtype_device: DtypeDevice,
    config: eskf_baseline.Config,
    operating_points: OperatingPoints,
) -> None:
    b = torch.tensor([0.2, 0.3, 0.4], **dtype_device)  # Example magnetic field vector
    grav_vector = torch.tensor(
        [0.0, 0.0, 9.81], **dtype_device
    )  # Example gravity vector
    b -= b.dot(grav_vector) / grav_vector.norm() ** 2 * grav_vector
    num_vecs = 10
    vs = torch.randn(
        (len(operating_points[0]), num_vecs, eskf_baseline.NominalState.TANGENT_DIM),
        **dtype_device,
    )
    for x, _, vb in zip(*operating_points, vs):
        # 1. Get hand-derived Jacobian
        obs_jac_hand = torch.compile(eskf_baseline.compass_observation_jacobian)(x, b)

        # 2. Get AutoDiff Jacobian evaluated at exactly zero error
        delta_x_zero = torch.zeros(
            eskf_baseline.NominalState.TANGENT_DIM, dtype=x.p.dtype, device=x.p.device
        )

        # jacrev with respect to argument 0 (delta_x)
        obs_jac_ad = torch.compile(
            torch.func.jacrev(compass_observation_wrapper, argnums=0)
        )(delta_x_zero, x, b)

        torch.testing.assert_close(obs_jac_hand, obs_jac_ad, rtol=1e-4, atol=1e-4)

        delta0 = torch.zeros(
            eskf_baseline.NominalState.TANGENT_DIM, dtype=x.p.dtype, device=x.p.device
        )

        # Function of delta only (so jvp sees a single input)
        def g(delta: torch.Tensor) -> torch.Tensor:
            return compass_observation_wrapper(delta, x, b)

        for v in vb:
            # JVP via forward-mode AD
            _, jvp = torch.func.jvp(g, (delta0,), (v,))

            # Hand linearization action
            jvp_expected = obs_jac_hand @ v

            torch.testing.assert_close(jvp, jvp_expected, rtol=1e-4, atol=1e-4)
