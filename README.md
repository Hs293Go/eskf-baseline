# Unit-Tested ESKF baseline

[Python](./src/eskf_baseline/eskf_baseline.py) and
[C++](./include/eskf_baseline/eskf_baseline.hpp) implementations of the
[Error State Kalman Filter (ESKF)](https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf)
for testing and comparison purposes.

The Python version is implemented with pytorch, so that the linearized error
state dynamics is verified against the automatic differentiation of the
error-state dynamics.

The C++ version is implemented with Eigen, and a python binding via pybind11 is
provided to verify the C++ implementation against the Python one.

## Formulas

### Nominal State

```math
\mathbf{x}_k = \begin{bmatrix}
\mathbf{p}_l \\
\mathbf{q}_k \\
\mathbf{v}_k \\
\mathbf{b}_{a,k} \\
\mathbf{b}_{g,k}
\end{bmatrix}
```

> [!NOTE] <!-- break line -->
>
> Compared to Sola's original formulation, we put the orientation `q` right
> after the position `p` in the state vector, instead of after the velocity `v`.
> This is because position and orientation are usually stored together, c.f.
> [`geometry_msgs/Pose`](https://docs.ros2.org/jazzy/api/geometry_msgs/msg/Pose.html)
> in ROS and
> [`sophus::SE3`](https://docs.ros.org/en/noetic/api/sophus/html/classSophus_1_1SE3.html)

### Input

```math
\mathbf{u} = \begin{bmatrix}
\mathbf{a}_{m,k} \\
\boldsymbol{\omega}_{m,k}
\end{bmatrix}
```

### Nominal dynamics

```math
\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_{k}, \mathbf{u}_k) \triangleq \begin{bmatrix}
\mathbf{p}_{k} + \mathbf{v}_{k} \Delta t \\
\mathbf{q}_{k} \otimes \exp\left\{\boldsymbol{\omega}_k \Delta t\right\} \\
\mathbf{v}_{k} + \left(\mathbf{R}(\mathbf{q}_{k}) \mathbf{a}_k - \mathbf{g}\right) \Delta t \\
\mathbf{b}_{a,k} \\
\mathbf{b}_{g,k}
\end{bmatrix},
```

where $\boldsymbol{\omega}\_k = \boldsymbol{\omega}\_{m,k} - \mathbf{b}\_{g,k}$
and $\mathbf{a}\_k = \mathbf{a}\_{m,k} - \mathbf{b}\_{a,k}$, and
$\exp\lbrace\cdot\rbrace$ is the quaternion exponential map.

### Linearized error-state dynamics

```math
\mathbf{F}_k = \begin{bmatrix}
\mathbf{1} & \mathbf{0} & \Delta t \mathbf{1} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \exp(-\Delta t \boldsymbol{\omega}_k) & \mathbf{0} & \mathbf{0} & -\Delta t \mathbf{1} \\
\mathbf{0} & -\Delta t\mathbf{R}(\mathbf{q}_{k}) \mathbf{a}_k^\times & \mathbf{1} & -\Delta t \mathbf{R}(\mathbf{q}_{k}) & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{1} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{1}
\end{bmatrix},
```

where $\exp(\cdot)$ is the conventional SO(3) exponential map and
$(\cdot)^\times$ is the skew-symmetric operator.

### Pose observation

Full pose observation may be sourced from a visual SLAM system, a LIDAR SLAM
system, or a motion capture system.

```math
\mathbf{z}_k = \mathbf{h}(\mathbf{x}_k) \triangleq \begin{bmatrix}
\mathbf{p}_k \\
\mathbf{q}_k
\end{bmatrix}
```

Its linearization is trivial

```math
\mathbf{H}_k = \begin{bmatrix}
\mathbf{1} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{1} & \mathbf{0} & \mathbf{0} & \mathbf{0}
\end{bmatrix}.
```

### Compass observation

Compass observation may be sourced from a magnetometer.

```math
\mathbf{z}_k = \mathbf{h}(\mathbf{x}_k) \triangleq \mathbf{R}^\top(\mathbf{q}_k) \mathbf{m}
```

where $\mathbf{m}$ is the magnetic field vector in the world frame. Its
linearization is:

```math
\mathbf{H}_k = \begin{bmatrix}
\mathbf{0} & {\left(\mathbf{R}^\top(\mathbf{q}_k)\right)}^\times & \mathbf{0} & \mathbf{0} & \mathbf{0}
\end{bmatrix}.
```

The compass observation is generalizable to all vector observations in the world
frame, as the relevant Jacobian block is only the derivative of a vector rotated
into the body frame with respect to the orientation:

```math
\frac{\partial \mathbf{R}^\top(\mathbf{q}_k) \mathbf{v}}{\partial \boldsymbol{\theta}} = {\left(\mathbf{R}^\top(\mathbf{q}_k)\mathbf{v}\right)}^\times.
```

## Dependencies

We use `uv` to manage our python dependencies.
[Follow instructions](https://docs.astral.sh/uv/getting-started/installation/)
to install `uv` itself, or run

```bash
sudo apt-get install pipx
pipx install uv
```

if you are on a Debian-based system.

Then, to install the dependencies, run

```bash
uv sync
```

## Running the tests

To run the tests, simply execute

```bash
uv run pytest
```

## Further development

Use C++-native autodiff techniques to verify the C++ implementation.
