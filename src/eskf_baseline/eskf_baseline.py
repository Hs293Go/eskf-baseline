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
    res = torch.zeros((3, 3), dtype=v.dtype, device=v.device)
    res[0, 1] = -v[2]
    res[0, 2] = v[1]
    res[1, 0] = v[2]
    res[1, 2] = -v[0]
    res[2, 0] = -v[1]
    res[2, 1] = v[0]
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


def quaternion_rotate_point(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate a point by a quaternion.

    Args:
        q (torch.Tensor): Quaternion (4,)
        v (torch.Tensor): Point to rotate (3,)

    Returns:
        torch.Tensor: Rotated point (3,)
    """
    w = q[3]
    vec = q[:3]
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
    imag_series = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4
    real_series = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_po4

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

    res = torch.empty((3, 3), dtype=q.dtype, device=q.device)
    res[0, 0] = 1 - (tyy + tzz)
    res[0, 1] = txy - twz
    res[0, 2] = txz + twy
    res[1, 0] = txy + twz
    res[1, 1] = 1 - (txx + tzz)
    res[1, 2] = tyz - twx
    res[2, 0] = txz - twy
    res[2, 1] = tyz + twx
    res[2, 2] = 1 - (txx + tyy)
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


class NominalState(NamedTuple):
    p: torch.Tensor  # Position vector (3,)
    q: torch.Tensor  # Orientation quaternion (4,)
    v: torch.Tensor  # Velocity vector (3,)
    accel_bias: torch.Tensor  # Accelerometer bias (3,)
    gyro_bias: torch.Tensor  # Gyroscope bias (3,)

    def perturb(self, delta: torch.Tensor) -> "NominalState":
        """
        Perturb the nominal state by a small error state.

        Args:
            delta (torch.Tensor): Error state (15,)
                [dp (3), dth (3), dv (3), dba (3), dbg (3)]
        Returns:
            NominalState: Perturbed nominal state
        """
        dp, dth, dv, dba, dbg = torch.split(delta, 3)
        return NominalState(
            p=self.p + dp,
            q=quaternion_product(self.q, angle_axis_to_quaternion(dth)),
            v=self.v + dv,
            accel_bias=self.accel_bias + dba,
            gyro_bias=self.gyro_bias + dbg,
        )


class Input(NamedTuple):
    accel: torch.Tensor  # Unbiased acceleration vector (3,)
    gyro: torch.Tensor  # Unbiased gyroscope vector (3,)


class Noise(NamedTuple):
    accel_noise: torch.Tensor  # Accelerometer noise (3,)
    gyro_noise: torch.Tensor  # Gyroscope noise (3,)
    accel_bias_noise: torch.Tensor  # Accelerometer bias random walk noise (3,)
    gyro_bias_noise: torch.Tensor  # Gyroscope bias random walk noise (3,)


@dataclasses.dataclass
class Config:
    grav_vector: torch.Tensor = dataclasses.field(
        default_factory=lambda: torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
    )
    accel_noise_density: float = 0.005
    gyro_noise_density: float = 5e-5
    accel_bias_random_walk: float = 0.001
    gyro_bias_random_walk: float = 0.0001


def kinematics(
    x: NominalState,
    u: Input,
    dt: torch.Tensor,
    config: Config,
) -> NominalState:
    """
    Kinematics function for the ESKF.

    Args
    ----
    x (NominalState): Nominal state
    u (Input): Input
    dt (torch.Tensor): Time step (scalar)
    config (Config): Configuration parameters
    """
    p, q, v, accel_bias, gyro_bias = x
    accel, gyro = u
    accel_unbiased = accel - accel_bias
    accel_world = quaternion_rotate_point(q, accel_unbiased) + torch.as_tensor(
        config.grav_vector, dtype=p.dtype, device=p.device
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
    )


def jacobians(
    x: NominalState, u: Input, dt: torch.Tensor, config: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Jacobians of the kinematics function.

    Args
    ----
    x (NominalState): Nominal state
    u (Input): Input
    dt (torch.Tensor): Time step (scalar)
    config (Config): Configuration parameters

    """
    p, q, v, accel_bias, gyro_bias = x
    accel, gyro = u
    accel_unbiased = accel - accel_bias
    gyro_unbiased = gyro - gyro_bias
    delta_angle = gyro_unbiased * dt
    fjac = torch.zeros((15, 15), dtype=p.dtype, device=p.device)

    eye3 = torch.eye(3, dtype=p.dtype, device=p.device)
    fjac[0:3, 0:3] = eye3
    fjac[0:3, 6:9] = dt * eye3

    fjac[3:6, 3:6] = angle_axis_to_rotation_matrix(-delta_angle)
    fjac[3:6, 12:15] = -dt * eye3

    rmat = quaternion_to_rotation_matrix(q)
    fjac[6:9, 3:6] = -dt * rmat @ hat(accel_unbiased)
    fjac[6:9, 6:9] = eye3
    fjac[6:9, 9:12] = -dt * rmat

    fjac[9:12, 9:12] = eye3
    fjac[12:15, 12:15] = eye3

    qcov = torch.zeros((15, 15), dtype=p.dtype, device=p.device)
    qcov[3:6, 3:6] = (config.gyro_noise_density * dt) ** 2 * eye3
    qcov[6:9, 6:9] = (config.accel_noise_density * dt) ** 2 * eye3
    qcov[9:12, 9:12] = (config.accel_bias_random_walk * dt) ** 2 * eye3
    qcov[12:15, 12:15] = (config.gyro_bias_random_walk * dt) ** 2 * eye3

    return fjac, qcov
