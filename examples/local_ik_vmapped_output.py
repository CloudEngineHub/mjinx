import time

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from mujoco import viewer
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()


# --- Mujoco visualization ---
# Initialize render window and launch it at the background
mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model)
mj_viewer = viewer.launch_passive(
    mj_model,
    mj_data,
    show_left_ui=False,
    show_right_ui=False,
)

# Initialize a sphere marker for end-effector task
renderer.scene.ngeom += 1
mj_viewer.user_scn.ngeom = 1
mj.mjv_initGeom(
    mj_viewer.user_scn.geoms[0],
    mj.mjtGeom.mjGEOM_SPHERE,
    0.05 * np.ones(3),
    np.array([0.2, 0.2, 0.2]),
    np.eye(3).flatten(),
    np.array([0.565, 0.933, 0.565, 0.4]),
)

# === Mjinx ===

# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-100, v_max=100)

# Creating components of interest and adding them to the problem
frame_task = FrameTask("ee_task", cost=1, gain=20, body_name="link7")
position_barrier = PositionBarrier(
    "ee_barrier",
    gain=100,
    body_name="link7",
    limit_type="max",
    p_max=0.3,
    safe_displacement_gain=1e-2,
    mask=[1, 0, 0],
)
joints_barrier = JointBarrier("jnt_range", gain=10)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=20)

# Initializing initial condition
N_batch = 10000
q0 = np.array(
    [
        -1.5878328,
        -2.0968683,
        -1.4339591,
        1.6550868,
        2.1080072,
        1.646142,
        -2.982619,
    ]
)
q = jnp.array([q0.copy() for _ in range(N_batch)])

# --- Batching ---
# First of all, data should be created via vmapped init function
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

# To create a batch w.r.t. desired component's attributes, library defines convinient wrapper
# That sets all elements to None and allows user to mutate dataclasses of interest.
# After exiting the Context Manager, you'll get immutable jax dataclass object.
with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["ee_task"].target_frame = 0

# Vmapping solve and integrate functions.
# Note that for batching w.r.t. q both q and solver_data should be batched.
# Other approaches might work, but it would be undefined behaviour, please stick to this format.
solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, 0, empty_problem_data),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])


# === Control loop ===
dt = 1e-2
ts = np.arange(0, 20, dt)

t_solve_avg = 0.0
n = 0

for t in ts:
    # Changing desired values
    frame_task.target_frame = np.array(
        [[0.2 + 0.2 * np.sin(t + np.pi * i / N_batch) ** 2, 0.2, 0.2, 1, 0, 0, 0] for i in range(N_batch)]
    )
    problem_data = problem.compile()
    t0 = time.perf_counter()

    # Solving the instance of the problem
    opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
    t1 = time.perf_counter()

    # Integrating
    q = integrate_jit(
        mjx_model,
        q,
        opt_solution.v_opt,
        dt,
    )
    t2 = time.perf_counter()

    # --- MuJoCo visualization ---
    mj_data.qpos = q[0]
    mj.mj_forward(mj_model, mj_data)
    print(f"Position barrier: {mj_data.xpos[position_barrier.body_id][0]} <= {position_barrier.p_max[0]}")
    mj.mjv_initGeom(
        mj_viewer.user_scn.geoms[0],
        mj.mjtGeom.mjGEOM_SPHERE,
        0.05 * np.ones(3),
        np.array(frame_task.target_frame.wxyz_xyz[0, -3:], dtype=np.float64),
        np.eye(3).flatten(),
        np.array([0.565, 0.933, 0.565, 0.4]),
    )

    # Run the forward dynamics to reflec
    # the updated state in the data
    mj.mj_forward(mj_model, mj_data)
    mj_viewer.sync()

    # --- Logging ---
    # Execution time
    t_solve = (t1 - t0) * 1e3
    # Ignore the first (compiling) iteration and calculate mean solution times
    if t > 0:
        t_solve_avg = t_solve_avg + (t_solve - t_solve_avg) / (n + 1)
        n += 1

print(f"Avg solving time: {t_solve_avg:0.3f}ms")
