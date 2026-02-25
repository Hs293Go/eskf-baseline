import dataclasses
from typing import NamedTuple, Self

import torch


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the hat operator for a 3D vector.

    Args:
        v (torch.Tensor): Input vector (3,)
    Returns:
        torch.Tensor: Hat operator of the input vector (3, 3)
    """
    zero = torch.zeros_like(v[..., 0], dtype=v.dtype, device=v.device)
    res = torch.stack(
        [zero, -v[2], v[1], v[2], zero, -v[0], -v[1], v[0], zero]
    ).reshape(3, 3)
    return res


def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a quaternion.

    Args:
        q (torch.Tensor): Quaternion (4,)

    Returns:
        torch.Tensor: Inverse of the quaternion (4,)
    """
    return torch.concat((-q[:3], q[3:4]), dim=0)


def quaternion_rotate_point(
    q: torch.Tensor, v: torch.Tensor, invert: bool = False
) -> torch.Tensor:
    """
    Rotate a point by a quaternion.

    Args
    ----
    q (torch.Tensor): Quaternion (4,)
    v (torch.Tensor): Point to rotate (3,)
    invert (bool): If True, rotate by the inverse of the quaternion

    Returns
    -------
    torch.Tensor: Rotated point (3,)
    """
    w = q[3]
    vec = torch.where(torch.as_tensor(invert, dtype=torch.bool), -q[:3], q[:3])
    vx = torch.cross(vec, v, dim=0)
    return v + 2 * (w * vx + torch.cross(vec, vx, dim=0))


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Convert an angle-axis representation to a quaternion.

    Args:
        angle_axis (torch.Tensor): Angle-axis representation (3,)
    Returns:
        torch.Tensor: Quaternion (4,)
    """
    eps = torch.as_tensor([1e-6], dtype=angle_axis.dtype, device=angle_axis.device)
    theta_sq = torch.dot(angle_axis, angle_axis)  # scalar
    theta = torch.sqrt(torch.max(theta_sq, eps))

    half = 0.5 * theta
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)

    # series expansions around 0
    theta_po4 = theta_sq**2
    imag_series = (
        torch.as_tensor([0.5], dtype=angle_axis.dtype, device=angle_axis.device)
        - (1.0 / 48.0) * theta_sq
        + (1.0 / 3840.0) * theta_po4
    )
    real_series = (
        torch.as_tensor([1.0], dtype=angle_axis.dtype, device=angle_axis.device)
        - (1.0 / 8.0) * theta_sq
        + (1.0 / 384.0) * theta_po4
    )

    imag_general = sin_half / theta
    real_general = cos_half

    imag = torch.where(theta_sq < eps, imag_series, imag_general)
    real = torch.where(theta_sq < eps, real_series, real_general)

    return torch.concat((imag * angle_axis, real), dim=0)


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Convert an angle-axis representation to a rotation matrix.

    Args:
        angle_axis (torch.Tensor): Angle-axis representation (3,)
    Returns:
        torch.Tensor: Rotation matrix (3, 3)
    """

    eps = torch.as_tensor([1e-6], dtype=angle_axis.dtype, device=angle_axis.device)
    theta_sq = torch.vdot(angle_axis, angle_axis)
    hat_phi = hat(angle_axis)

    theta = torch.sqrt(theta_sq)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    eye = torch.eye(3, dtype=angle_axis.dtype, device=angle_axis.device)
    return torch.where(
        theta_sq < eps,
        eye + hat_phi,
        eye
        + (1 - cos_theta) / theta_sq * hat_phi @ hat_phi
        + sin_theta / theta * hat_phi,
    )


def quaternion_to_angle_axis(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to an angle-axis representation.

    Args:
        q (torch.Tensor): Quaternion (4,)
    Returns:
        torch.Tensor: Angle-axis representation (3,)
    """

    eps = torch.as_tensor(1e-6, dtype=q.dtype, device=q.device)
    squared_n = torch.vdot(q[:3], q[:3])
    w = q[3]
    n = torch.sqrt(torch.max(squared_n, eps))

    atan_nbyw = torch.where(w < 0, torch.atan2(-n, -w), torch.atan2(n, w))
    two_atan_nbyw_by_n = torch.where(
        squared_n < eps,
        (2.0) / w - (2.0 / 3.0) * (squared_n) / w**3,
        2 * atan_nbyw / n,
    )

    return two_atan_nbyw_by_n * q[:3]


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to a rotation matrix.

    Args:
        q (torch.Tensor): Quaternion (4,)
    Returns:
        torch.Tensor: Rotation matrix (3, 3)
    """
    tx = 2 * q[0]
    ty = 2 * q[1]
    tz = 2 * q[2]
    twx = tx * q[3]
    twy = ty * q[3]
    twz = tz * q[3]
    txx = tx * q[0]
    txy = ty * q[0]
    txz = tz * q[0]
    tyy = ty * q[1]
    tyz = tz * q[1]
    tzz = tz * q[2]

    res = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
    ).reshape(3, 3)
    return res


def quaternion_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.

    Args:
        q1 (torch.Tensor): First quaternion (4,)
        q2 (torch.Tensor): Second quaternion (4,)

    Returns:
        torch.Tensor: Product of the two quaternions (4,)
    """

    return torch.concat(
        [
            a[:3] * b[3] + a[3] * b[:3] + torch.cross(a[:3], b[:3], dim=0),
            torch.atleast_1d(a[3] * b[3] - torch.vdot(a[:3], b[:3])),
        ],
    )


def rotation_error(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the rotation error between two quaternions.

    This computes q_ab = q_{ia}^{-1} * q_{ib}, where q_{ia} and q_{ib} are
    orientations in the passive body-to-inertial convention.

    Args:
        q1 (torch.Tensor): First quaternion (4,)
        q2 (torch.Tensor): Second quaternion (4,)

    Returns:
        torch.Tensor: Rotation error in angle-axis representation (3,)
    """
    return quaternion_to_angle_axis(quaternion_product(quaternion_inverse(q_a), q_b))


def rotated_vector_by_perturbation_jacobian(
    q: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return -quaternion_to_rotation_matrix(q) @ hat(v)


def inversely_rotated_vector_by_perturbation_jacobian(
    q: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return hat(quaternion_rotate_point(q, v, invert=True))


class NominalState(NamedTuple):
    TANGENT_DIM = 18

    p: torch.Tensor  # Position vector (3,)
    q: torch.Tensor  # Orientation quaternion (4,)
    v: torch.Tensor  # Velocity vector (3,)
    accel_bias: torch.Tensor  # Accelerometer bias (3,)
    gyro_bias: torch.Tensor  # Gyroscope bias (3,)
    grav_vector: torch.Tensor  # Gravity vector (3,)

    def boxplus(self, delta: torch.Tensor) -> "NominalState":
        """
        Perturb the nominal state by a vector in the tangent space.

        Args:
            delta (torch.Tensor): Error state (15,)
                [dp (3), dth (3), dv (3), dba (3), dbg (3)]
        Returns:
            NominalState: Perturbed nominal state
        """
        dp, dth, dv, dba, dbg, dg = torch.split(delta, 3)
        return NominalState(
            p=self.p + dp,
            q=quaternion_product(self.q, angle_axis_to_quaternion(dth)),
            v=self.v + dv,
            accel_bias=self.accel_bias + dba,
            gyro_bias=self.gyro_bias + dbg,
            grav_vector=self.grav_vector + dg,
        )

    def boxminus(self, other: "NominalState") -> torch.Tensor:
        """
        Compute the error state between this nominal state and another nominal state.

        Args:
            other (NominalState): Other nominal state
        Returns:
            torch.Tensor: Error state (15,)
                [dp (3), dth (3), dv (3), dba (3), dbg (3)]
        """
        dp = self.p - other.p
        dth = rotation_error(other.q, self.q)
        dv = self.v - other.v
        dba = self.accel_bias - other.accel_bias
        dbg = self.gyro_bias - other.gyro_bias
        dg = self.grav_vector - other.grav_vector

        return torch.concat((dp, dth, dv, dba, dbg, dg), dim=0)


class ImuInput(NamedTuple):
    accel: torch.Tensor  # Unbiased acceleration vector (3,)
    gyro: torch.Tensor  # Unbiased gyroscope vector (3,)


class Noise(NamedTuple):
    accel_noise: torch.Tensor  # Accelerometer noise (3,)
    gyro_noise: torch.Tensor  # Gyroscope noise (3,)
    accel_bias_noise: torch.Tensor  # Accelerometer bias random walk noise (3,)
    gyro_bias_noise: torch.Tensor  # Gyroscope bias random walk noise (3,)


@dataclasses.dataclass
class Config:
    accel_noise_density: float = 0.005
    gyro_noise_density: float = 5e-5
    accel_bias_random_walk: float = 0.001
    gyro_bias_random_walk: float = 0.0001


def motion(
    x: NominalState, u: ImuInput, dt: torch.Tensor, config: Config
) -> NominalState:
    """
    Motion model for the ESKF.

    Args
    ----
    x (NominalState): Nominal state
    u (ImuInput): Input
    dt (torch.Tensor): Time step (scalar)
    config (Config): Configuration parameters
    """
    p, q, v, accel_bias, gyro_bias, grav_vector = x
    accel, gyro = u
    accel_unbiased = accel - accel_bias
    accel_world = quaternion_rotate_point(q, accel_unbiased) + torch.as_tensor(
        grav_vector, dtype=p.dtype, device=p.device
    )
    delta_velocity = accel_world * dt
    gyro_unbiased = gyro - gyro_bias
    delta_angle = gyro_unbiased * dt

    return NominalState(
        p=p + v * dt,
        q=quaternion_product(q, angle_axis_to_quaternion(delta_angle)),
        v=v + delta_velocity,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
        grav_vector=grav_vector,
    )


def motion_jacobians(
    x: NominalState, u: ImuInput, dt: torch.Tensor, config: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Jacobians of the motion model.

    Args
    ----
    x (NominalState): Nominal state
    u (ImuInput): Input
    dt (torch.Tensor): Time step (scalar)
    config (Config): Configuration parameters

    """
    p, q, v, accel_bias, gyro_bias, _ = x
    accel, gyro = u
    accel_unbiased = accel - accel_bias
    gyro_unbiased = gyro - gyro_bias
    delta_angle = gyro_unbiased * dt
    fjac = torch.zeros(
        (NominalState.TANGENT_DIM, NominalState.TANGENT_DIM),
        dtype=p.dtype,
        device=p.device,
    )

    eye3 = torch.eye(3, dtype=p.dtype, device=p.device)
    fjac[0:3, 0:3] = eye3
    fjac[0:3, 6:9] = dt * eye3

    fjac[3:6, 3:6] = angle_axis_to_rotation_matrix(-delta_angle)
    fjac[3:6, 12:15] = -dt * eye3

    fjac[6:9, 3:6] = dt * rotated_vector_by_perturbation_jacobian(q, accel_unbiased)
    fjac[6:9, 6:9] = eye3
    fjac[6:9, 9:12] = -dt * quaternion_to_rotation_matrix(q)
    fjac[6:9, 15:18] = dt * eye3

    fjac[9:12, 9:12] = eye3
    fjac[12:15, 12:15] = eye3
    fjac[15:18, 15:18] = eye3

    qcov = torch.zeros(
        (NominalState.TANGENT_DIM, NominalState.TANGENT_DIM),
        dtype=p.dtype,
        device=p.device,
    )
    qcov[3:6, 3:6] = (config.gyro_noise_density * dt) ** 2 * eye3
    qcov[6:9, 6:9] = (config.accel_noise_density * dt) ** 2 * eye3
    qcov[9:12, 9:12] = (config.accel_bias_random_walk * dt) ** 2 * eye3
    qcov[12:15, 12:15] = (config.gyro_bias_random_walk * dt) ** 2 * eye3

    return fjac, qcov


class PoseObservation(NamedTuple):
    TANGENT_DIM = 6
    p: torch.Tensor  # Position vector (3,)
    q: torch.Tensor  # Orientation quaternion (4,)

    def boxminus(self, other: "PoseObservation") -> torch.Tensor:
        """
        Compute the innovation between this pose observation and another pose observation.

        Args
        ----
        other (PoseObservation): Other pose observation

        Returns
        -------
        torch.Tensor: Innovation (6,)
            [dp (3), dth (3)]
        """
        dp = self.p - other.p
        dth = rotation_error(other.q, self.q)
        return torch.concat((dp, dth), dim=0)


def pose_observation(x: NominalState) -> PoseObservation:
    """
    Pose observation function.

    Args
    ----
    x (NominalState): Nominal state

    Returns
    -------
    PoseObservation: Pose observation
    """

    return PoseObservation(p=x.p, q=x.q)


def pose_observation_jacobian(x: NominalState) -> torch.Tensor:
    """
    Compute the Jacobian of the pose observation function.

    Args
    ----
    x (NominalState): Nominal state

    Returns
    -------
    torch.Tensor: Jacobian of the pose observation function (7, 15)
    """
    jac = torch.zeros(
        (PoseObservation.TANGENT_DIM, NominalState.TANGENT_DIM),
        dtype=x.p.dtype,
        device=x.p.device,
    )
    jac[0:3, 0:3] = torch.eye(3, dtype=x.p.dtype, device=x.p.device)
    jac[3 : PoseObservation.TANGENT_DIM, 3 : PoseObservation.TANGENT_DIM] = torch.eye(
        3, dtype=x.p.dtype, device=x.p.device
    )
    return jac


class CompassObservation(NamedTuple):
    TANGENT_DIM = 3

    b: torch.Tensor  # Body-frame magnetic field vector (3,)

    def boxminus(self, other: "CompassObservation") -> torch.Tensor:
        """
        Compute the innovation between this compass observation and another compass observation

        Args
        ----
        other (CompassObservation): Other compass observation

        Returns
        -------
        torch.Tensor: Innovation (3,)
        """
        return self.b - other.b


def compass_observation(
    x: NominalState, mag_inertial: torch.Tensor
) -> CompassObservation:
    """
    Compass observation function.

    Args
    ----
    x (NominalState): Nominal state
    mag_inertial (torch.Tensor): Inertial magnetic field vector (3,)

    Returns
    -------
    CompassObservation: Compass observation
    """
    b = quaternion_rotate_point(x.q, mag_inertial, invert=True)
    return CompassObservation(b=b)


def compass_observation_jacobian(
    x: NominalState, mag_inertial: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Jacobian of the compass observation function.

    Args
    ----
    x (NominalState): Nominal state
    mag_inertial (torch.Tensor): Inertial magnetic field vector (3,)

    Returns
    -------
    torch.Tensor: Jacobian of the compass observation function (3, 15)
    """
    jac = torch.zeros(
        (CompassObservation.TANGENT_DIM, NominalState.TANGENT_DIM),
        dtype=x.p.dtype,
        device=x.p.device,
    )
    jac[:, 3:6] = inversely_rotated_vector_by_perturbation_jacobian(x.q, mag_inertial)
    return jac
