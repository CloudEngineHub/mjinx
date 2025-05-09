<!-- # MJINX -->
[![mypy](https://img.shields.io/github/actions/workflow/status/based-robotics/mjinx/mypy.yml?branch=main&label=mypy)](https://github.com/based-robotics/mjinx/actions)
[![ruff](https://img.shields.io/github/actions/workflow/status/based-robotics/mjinx/ruff.yml?branch=main&label=ruff)](https://github.com/based-robotics/mjinx/actions)
[![docs](https://img.shields.io/github/actions/workflow/status/based-robotics/mjinx/docs.yml?branch=main&label=docs)](https://based-robotics.github.io/mjinx/)
[![PyPI version](https://img.shields.io/pypi/v/mjinx?color=blue)](https://pypi.org/project/mjinx/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mjinx?color=blue)](https://pypistats.org/packages/mjinx)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/based-robotics/mjinx/blob/main/examples/notebooks/turoial.ipynb)


<p align="center">
  <img src="img/logo.svg" style="width: 500px" />
</p>


**MJINX** is a python library for auto-differentiable numerical inverse kinematics built on **JAX** and **Mujoco MJX**. It draws inspiration from similar tools like the Pinocchio-based [PINK](https://github.com/stephane-caron/pink/tree/main) and Mujoco-based [MINK](https://github.com/kevinzakka/mink/tree/main).

<p align="center">
  <img src="img/local_ik_output.gif" style="width: 300px" />
  <img src="img/go2_stance.gif" style="width: 300px" /> 
  <img src="img/g1_heart.gif" style="width: 300px"/>
  <img src="img/cassie_caravan.gif" style="width: 300px"/>
</p>

## Key Features
1. **Flexibility**. Problems are constructed using modular `Components` that enforce desired behaviors or maintain system safety constraints.
2. **Multiple Solution Strategies**. Leveraging JAX's efficient sampling and automatic differentiation capabilities, MJINX implements various solvers optimized for different robotics scenarios.
3. **Full JAX Compatibility**. Both the optimal control formulation and solvers are fully JAX-compatible, enabling JIT compilation and automatic vectorization across the entire pipeline.
4. **User-Friendly Interface**. The API is designed with a clean, intuitive interface that simplifies complex inverse kinematics tasks while maintaining advanced functionality.

## Installation
The package is available in PyPI registry, and could be installed via `pip`:
```bash
pip install mjinx
```

Different installation versions:
1. Visualization tool `mjinx.visualization.BatchVisualizer` is available in `mjinx[visual]` 
2. To run examples, install `mjinx[examples]`
3. To install development version, install `mjinx[dev]` (preferably in editable mode)
4. To build docs, install `mjinx[docs]`
5. To install the repository with all dependencies, install `mjinx[all]`

Note that by default installation of `mjinx` installs `jax` without cuda support. If you need it, please install `jax>=0.5.0` with CUDA support manually.

## Usage
Here is the example of `mjinx` usage:

```python
from mujoco import mjx mjx
from mjinx.problem import Problem

# Initialize the robot model using MuJoCo
MJCF_PATH = "path_to_mjcf.xml"
mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)

# Create instance of the problem
problem = Problem(mjx_model)

# Add tasks to track desired behavior
frame_task = FrameTask("ee_task", cost=1, gain=20, body_name="link7")
problem.add_component(frame_task)

# Add barriers to keep robot in a safety set
joints_barrier = JointBarrier("jnt_range", gain=10)
problem.add_component(joints_barrier)

# Initialize the solver
solver = LocalIKSolver(mjx_model)

# Initializing initial condition
q0 = np.zeros(7)

# Initialize solver data
solver_data = solver.init()

# jit-compiling solve and integrate 
solve_jit = jax.jit(solver.solve)
integrate_jit = jax.jit(integrate, static_argnames=["dt"])

# === Control loop ===
for t in np.arange(0, 5, 1e-2):
    # Changing problem and compiling it
    frame_task.target_frame = np.array([0.1 * np.sin(t), 0.1 * np.cos(t), 0.1, 1, 0, 0,])
    problem_data = problem.compile()

    # Solving the instance of the problem
    opt_solution, solver_data = solve_jit(q, solver_data, problem_data)

    # Integrating
    q = integrate_jit(
        mjx_model,
        q,
        opt_solution.v_opt,
        dt,
    )
```

## Examples
The list of examples includes:
   1. `Kuka iiwa` local inverse kinematics ([single item](examples/local_ik.py), [vmap over desired trajectory](examples/local_ik_vmapped_output.py))
   2. `Kuka iiwa` global inverse kinematics ([single item](examples/global_ik.py), [vmap over desired trajectory](examples/global_ik_vmapped_output.py))
   3. `Go2` [batched squats](examples/go2_squat.py) example
   
> **Note:** The Global IK functionality is currently under development and not yet working properly as expected. It needs proper tuning and will be fixed in future updates. Use the Global IK examples with caution and expect suboptimal results.


## Citing MJINX

If you use MJINX in your research, please cite it as follows:

```bibtex
@software{mjinx25,
  author = {Domrachev, Ivan and Nedelchev, Simeon},
  license = {MIT},
  month = mar,
  title = {{MJINX: Differentiable GPU-accelerated inverse kinematics in JAX}},
  url = {https://github.com/based-robotics/mjinx},
  version = {0.1.1},
  year = {2025}
}
```

## Contributing
We welcome suggestions and contributions. Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## Acknowledgements
I am deeply grateful to Simeon Nedelchev, whose guidance and expertise were instrumental in bringing this project to life.

This work draws significant inspiration from [`pink`](https://github.com/stephane-caron/pink) by Stéphane Caron and [`mink`](https://github.com/kevinzakka/mink) by Kevin Zakka. Their pioneering work in robotics and open source has been a guiding light for this project.

The codebase incorporates utility functions from [`MuJoCo MJX`](https://github.com/google-deepmind/mujoco/tree/main/mjx). Beyond being an excellent tool for batched computations and machine learning, MJX's codebase serves as a masterclass in clean, informative implementation of physical simulations and JAX usage.

Special thanks to [IRIS lab](http://iris.kaist.ac.kr/) for their support.
