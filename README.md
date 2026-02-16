# Unit-Tested ESKF baseline

[Python](./src/eskf_baseline/eskf_baseline.py) and
[C++](./include/eskf_baseline/eskf_baseline.hpp) implementations of the
[Error State Kalman Filter (ESKF)](https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf)
for testing and comparison purposes.

The Python version is implemented with pytorch and the linearized system
Jacobian is compared against the automatic differentiation of the error-state
dynamics.

## Implementation divergence

Compared to Sola's original formulation, we have made the following changes:

We order the state as `[p, q, v, b_a, b_g]` instead of `[p, v, q, b_a, b_g]`.
This is because position and orientation are usually contiguously stored
together.

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

Also compare the C++ implementation against the Python one, or potentially use
ceres::Jet to autodiff the C++ implementation and compare against the analytical
Jacobian.
