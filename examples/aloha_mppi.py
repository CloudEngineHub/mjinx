import os
from dataclasses import dataclass

import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer


@dataclass
class GeomData:
    name: str
    type: str
    size: np.ndarray
    pos: np.ndarray


def spawn_geom(vis: BatchVisualizer, spec: mj.MjSpec, geom_data: GeomData):
    vis.mjcf_model.worldbody.add(
        "geom",
        type=geom_data.type,
        size=geom_data.size,
        pos=geom_data.pos,
    )
    geom_type: mj.mjtGeom
    match geom_data.type:
        case "sphere":
            geom_type = mj.mjtGeom.mjGEOM_SPHERE
        case "cylinder":
            geom_type = mj.mjtGeom.mjGEOM_CYLINDER
        case "box":
            geom_type = mj.mjtGeom.mjGEOM_BOX
        case "capsule":
            geom_type = mj.mjtGeom.mjGEOM_CAPSULE
        case _:
            raise ValueError(f"geom {geom_data.type} not supported")
    size_extended = np.zeros(3)
    size_extended[: len(geom_data.size)] = geom_data.size
    pos_extended = np.zeros(3)
    pos_extended[: len(geom_data.pos)] = geom_data.pos
    g = spec.worldbody.add_geom()
    g.name = geom_data.name
    g.type = geom_type
    g.size = size_extended
    g.pos = pos_extended


aloha_mjcf: str = ""
filename: str = "mjx_aloha.xml"
for root, _, files in os.walk(os.path.abspath(os.path.dirname(__file__))):
    if filename in files:
        aloha_mjcf = os.path.join(root, filename)
        break

# === Mujoco ===


# --- Mujoco visualization ---
# Initialize render window and launch it at the background
vis = BatchVisualizer(aloha_mjcf, n_models=1, alpha=1, compile=False)
mj_spec = mj.MjSpec()
mj_spec.from_file(aloha_mjcf)


# Adding ground plane
vis.mjcf_model.asset.add(
    "texture",
    name="groundplane",
    type="2d",
    builtin="checker",
    rgb1=[0.2, 0.3, 0.4],
    rgb2=[0.1, 0.2, 0.3],
    width=320,
    height=320,
)
vis.mjcf_model.asset.add(
    "material",
    name="groundplane",
    texture="groundplane",
    texuniform="true",
    texrepeat=[5, 5],
)
vis.mjcf_model.worldbody.add(
    "geom",
    name="ground",
    type="plane",
    size=[0, 0, 0.05],
    pos=[0, 0, 0],
    material="groundplane",
)

# Spawn obstacles
obstacles = [
    GeomData(name="obstacle_0", type="cylinder", size=np.array([0.1, 0.15]), pos=np.array([0, 0, 0.15])),
    GeomData(name="obstacle_1", type="box", size=np.array([0.05, 0.05, 0.05]), pos=np.array([0.2, 0.3, 0.15])),
    GeomData(name="obstacle_2", type="box", size=np.array([0.08, 0.07, 0.04]), pos=np.array([-0.2, 0.0, 0.23])),
    GeomData(name="obstacle_3", type="box", size=np.array([0.1, 0.03, 0.13]), pos=np.array([0.15, 0.17, 0.5])),
    GeomData(name="obstacle_4", type="box", size=np.array([0.2, 0.2, 0.02]), pos=np.array([0.0, 0.4, 0.3])),
    GeomData(name="obstacle_5", type="sphere", size=np.array([0.07]), pos=np.array([-0.4, 0.25, 0.25])),
    GeomData(name="obstacle_6", type="sphere", size=np.array([0.09]), pos=np.array([-0.5, 0.3, 0.1])),
    GeomData(name="obstacle_7", type="sphere", size=np.array([0.03]), pos=np.array([-0.14, 0.5, 0.13])),
    GeomData(name="obstacle_8", type="sphere", size=np.array([0.03]), pos=np.array([-0.4, 0.1, 0.4])),
    GeomData(name="obstacle_9", type="sphere", size=np.array([0.03]), pos=np.array([-0.24, 0.2, 0.6])),
    GeomData(name="obstacle_10", type="sphere", size=np.array([0.05]), pos=np.array([0.0, 0.5, 0.5])),
]
for obstacle in obstacles:
    spawn_geom(vis, mj_spec, obstacle)

mjx_model = mjx.put_model(mj_spec.compile())


vis.compile_model()
vis.launch()

try:
    while True:
        pass
except Exception:
    vis.close()
