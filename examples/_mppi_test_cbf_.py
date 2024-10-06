import time
from functools import partial

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from mujoco import viewer
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx import configuration
from mjinx.components.barriers import JointBarrier, PositionBarrier, SelfCollisionBarrier
from mjinx.components.barriers._base import JaxBarrier
from mjinx.components.tasks import FrameTask
from mjinx.components.tasks._base import JaxTask
from mjinx.configuration import integrate
from mjinx.problem import JaxProblemData, Problem


def log_barrier(x: jnp.ndarray, gain: jnp.ndarray):
    return jnp.sum(gain * jax.lax.map(jnp.log, x))


def cost_fn(model_data: mjx.Data, problem_data: JaxProblemData) -> float:
    loss = 0

    for component in problem_data.components.values():
        if isinstance(component, JaxTask):
            err = component(model_data)
            loss = loss + component.vector_gain * err.T @ err
        if isinstance(component, JaxBarrier):
            barrier = jnp.clip(component(model_data), min=1e-2)
            loss = loss - log_barrier(barrier, gain=component.vector_gain)
    return loss


ScanCarryType = tuple[float, mjx.Data]
SampleType = tuple[tuple[float, mjx.Data], jnp.ndarray]


@partial(jax.jit, static_argnames=["n_samples", "horizon"])
def jax_mppi(
    mjx_data0: mjx.Data,
    vel0: jnp.ndarray,
    rng_key,
    problem_data: JaxProblemData,
    model: mjx.Model,
    variance: float = 0.3,
    temperature: float = 1.0,
    dt: float = 1e-2,
    n_samples: int = 100,
    horizon: int = 10,
):
    def scan_fn(carry: ScanCarryType, vel: jnp.ndarray) -> tuple[ScanCarryType, None]:
        cost, model_data = carry
        cost += cost_fn(model_data, problem_data)

        # TODO: is sequence correct?
        model_data = model_data.replace(
            qpos=configuration.integrate(model, model_data.qpos, velocity=vel, dt=dt),
            # qvel=vel,
        )
        model_data = mjx.fwd_position(model, model_data)
        model_data = mjx.com_pos(model, model_data)

        return (cost, model_data), None

    def get_sample(key: jnp.ndarray) -> SampleType:
        noise = 1.0 * jax.random.normal(key, (horizon, model.nq))
        noise_scaled = noise * variance
        u_seq = vel0 + noise_scaled

        (cost, terminal_data), _ = jax.lax.scan(
            scan_fn,
            (0.0, mjx_data0),
            u_seq,
        )

        # Calculate terminal cost
        cost += cost_fn(model_data=terminal_data, problem_data=problem_data)
        return (cost, terminal_data), u_seq

    rng_keys = jax.random.split(rng_key, n_samples)
    (costs, _), all_seqs = jax.vmap(get_sample)(rng_keys)

    # get optimal control input
    # FIXME: why we substract from minimum?.. To map it from 1 to 0 for numerical stability?..
    exp_cost = jnp.exp(temperature * (jnp.min(costs) - costs))
    denom = jnp.sum(exp_cost) + 1e-7
    best_vel = jnp.sum(exp_cost[..., None, None] * all_seqs, axis=0) / denom

    # Remove the first element, and repeat last control input twice
    new_vel_init = jnp.roll(best_vel, shift=-1, axis=0)
    new_vel_init = new_vel_init.at[-1].set(new_vel_init[-2])
    return best_vel[0], new_vel_init, all_seqs


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
self_collision_barrier = SelfCollisionBarrier(
    "self_collision_barrier",
    gain=1.0,
    d_min=0.01,
)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)
# problem.add_component(self_collision_barrier)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initial condition
q = jnp.array(
    [
        -1.4238753,
        -1.7268502,
        -0.84355015,
        2.0962472,
        2.1339328,
        2.0837479,
        -2.5521986,
    ]
)
# Jit-compiling the key functions for better efficiency
integrate_jit = jax.jit(integrate, static_argnames=["dt"])
# === Control loop ===
dt = 1e-2
ts = np.arange(0, 0.1, dt)

t_solve_avg = 0.0
n = 0

horizon = 10
vel_init = jnp.zeros((horizon, mj_model.nv))

rng_key = jax.random.PRNGKey(0)
rng_key, subkey = jax.random.split(rng_key)
vel_des, vel_init, sampled_smth = jax_mppi(
    configuration.update(mjx_model, q),
    vel_init,
    subkey,
    problem_data,
    mjx_model,
    variance=1e-1,
    temperature=1.0,
    dt=dt,
    n_samples=100,
    horizon=horizon,
)
q = integrate_jit(
    mjx_model,
    q,
    velocity=vel_des,
    dt=dt,
)
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
for t in ts:
    # Changing desired values
    frame_task.target_frame = np.array([0.2 + 0.2 * jnp.sin(t) ** 2, 0.2, 0.2, 1, 0, 0, 0])
    # After changes, recompiling the model
    problem_data = problem.compile()
    t0 = time.perf_counter()

    # Solving the instance of the problem
    rng_key, subkey = jax.random.split(rng_key)
    vel_des, vel_init, sampled_smth = jax_mppi(
        configuration.update(mjx_model, q),
        vel_init,
        subkey,
        problem_data,
        mjx_model,
        variance=1e-1,
        temperature=1.0,
        dt=dt,
        n_samples=100,
        horizon=horizon,
    )
    t1 = time.perf_counter()

    # Integrating
    q = integrate_jit(
        mjx_model,
        q,
        velocity=vel_des,
        dt=dt,
    )

    # --- MuJoCo visualization ---
    mj_data.qpos = q
    mj.mj_forward(mj_model, mj_data)
    # print(f"Position barrier: {mj_data.xpos[position_barrier.body_id][0]} <= {position_barrier.p_max[0]}")
    mj.mjv_initGeom(
        mj_viewer.user_scn.geoms[0],
        mj.mjtGeom.mjGEOM_SPHERE,
        0.05 * np.ones(3),
        np.array(frame_task.target_frame.wxyz_xyz[-3:], dtype=np.float64),
        np.eye(3).flatten(),
        np.array([0.565, 0.933, 0.565, 0.4]),
    )

    # Run the forward dynamics to reflec
    # the updated state in the data
    mj.mj_forward(mj_model, mj_data)
    mj_viewer.sync()

    t2 = time.perf_counter()
    t_solve = (t1 - t0) * 1e3
    t_interpolate = (t2 - t1) * 1e3

    if t > 0:
        t_solve_avg = t_solve_avg + (t_solve - t_solve_avg) / (n + 1)
        n += 1

    print(f"Avg solving time: {t_solve_avg:0.3f}ms")

while True:
    time.sleep(10)
