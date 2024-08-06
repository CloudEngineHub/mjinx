import os
import time

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3
from matplotlib import pyplot as plt
from mujoco import viewer

from mjinx import solve_local_ik
from mjinx.barriers import JointBarrier, PositionUpperBarrier
from mjinx.configuration import update
from mjinx.tasks import FrameTask

model_path = os.path.abspath(os.path.dirname(__file__)) + "/../../robot_descriptions/kuka_iiwa_14/iiwa14.xml"
mj_model = mj.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)

mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model)
mj_viewer = viewer.launch_passive(
    mj_model,
    mj_data,
    show_left_ui=False,
    show_right_ui=False,
)
q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()
# mj_model.jnt_range = [(-20 * np.pi, 20 * np.pi) for _ in range(7)]
renderer.scene.ngeom += 1
mj_viewer.user_scn.ngeom = 1

N_batch = 100
cur_q0 = np.array(
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
cur_q = jnp.array(
    [
        cur_q0.copy()
        # + np.random.uniform(
        #     -1e-1,
        #     1e-1,
        #     size=(mj_model.nq),
        # )
        for _ in range(N_batch)
    ]
)

ee_id = 8

tasks = {
    "ee_task": FrameTask(
        model=mjx_model,
        cost=1 * jnp.eye(6),
        gain=10 * jnp.ones(6),
        frame_id=ee_id,
        target_frame=SE3.from_rotation_and_translation(SO3.identity(), np.array([0.2, 0.2, 0.2])),
    ),
}
barriers = {
    "joint_barrier": JointBarrier(
        model=mjx_model,
        qmin=jnp.array(q_min),
        qmax=jnp.array(q_max),
        gain=jnp.concatenate(
            [
                10 * jnp.ones(mj_model.nv),
                10 * jnp.ones(mj_model.nv),
            ]
        ),
    ),
    "position_barrier": PositionUpperBarrier(
        model=mjx_model,
        gain=jnp.array([100.0]),
        frame_id=ee_id,
        axes="x",
        p_max=np.array([0.3]),
        safe_displacement_gain=1e-2,
    ),
}

dt = 1e-2
ts = np.arange(0, 20, dt)

mj.mjv_initGeom(
    mj_viewer.user_scn.geoms[0],
    mj.mjtGeom.mjGEOM_SPHERE,
    0.1 * np.ones(3),
    np.array([0.2, 0.2, 0.2]),
    np.eye(3).flatten(),
    np.array([0.565, 0.933, 0.565, 0.4]),
)

solve_local_ik_vmap = jax.jit(jax.vmap(solve_local_ik, in_axes=(None, 0, None, None)), static_argnames="damping")

# try:
# Warm-up JIT
jnts = []
for t in ts:
    tasks["ee_task"] = tasks["ee_task"].copy_and_set(
        target_frame=SE3.from_rotation_and_translation(
            SO3.identity(),
            np.array(
                [
                    0.2 + 0.2 * jnp.sin(t) ** 2,
                    0.2,
                    0.2,
                ]
            ),
        )
    )
    t0 = time.perf_counter()
    vel = solve_local_ik_vmap(mjx_model, cur_q, tasks, barriers)
    print(f"Time: {(time.perf_counter() - t0)*1e3 :.3f} ms")
    if vel is None:
        raise ValueError("No solution found for IK")

    cur_q += vel * dt
    mj_data.qpos = cur_q[0]
    mj_data.qvel = vel[0]
    mj.mj_forward(mj_model, mj_data)
    print(mj_data.xpos[ee_id][0])
    mj.mjv_initGeom(
        mj_viewer.user_scn.geoms[0],
        mj.mjtGeom.mjGEOM_SPHERE,
        0.1 * np.ones(3),
        np.array(tasks["ee_task"].target_frame.translation(), dtype=np.float64),
        np.eye(3).flatten(),
        np.array([0.565, 0.933, 0.565, 0.4]),
    )

    # Run the forward dynamics to reflec
    # the updated state in the data
    mj.mj_forward(mj_model, mj_data)
    mj_viewer.sync()

    jnts.append(mj_data.qpos.copy())

# except Exception as e:
#     print(e.with_traceback(None))
#     mj_viewer.close()

#     jnts = np.array(jnts)

#     fig, ax = plt.subplots(7, 1, figsize=(10, 20))

#     for i in range(7):
#         ax[i].axhline(q_min[i], color="r", ls="--", label="Lower limit")
#         ax[i].axhline(q_max[i], color="r", ls="--", label="Upper limit")
#         ax[i].plot(ts, jnts[:, i], label=f"Joint {i}")
#         ax[i].set_title(f"Joint {i}")
#         ax[i].set_xlabel("Time (s)")
#         ax[i].set_ylabel("Joint position (rad)")
#         ax[i].legend()

#     plt.tight_layout()
#     plt.show()
