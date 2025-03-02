from __future__ import annotations

from collections.abc import Sequence
from typing import final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx import typing
from mjinx.components.constraints._obj_constraint import JaxObjConstraint, ObjConstraint
from mjinx.configuration import get_frame_jacobian_world_aligned


@jdc.pytree_dataclass
class JaxPositionConstraint(JaxObjConstraint):
    refpos: jnp.ndarray

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return self.get_pos(data)[self.mask_idxs,] - self.refpos

    @final
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        return get_frame_jacobian_world_aligned(self.model, data, self.obj_id, self.obj_type)[:, self.mask_idxs].T


class PositionConstraint(ObjConstraint[JaxPositionConstraint]):
    JaxComponentType: type = JaxPositionConstraint
    _refpos: jnp.ndarray

    def __init__(
        self,
        name: str,
        gain: typing.ArrayOrFloat,
        obj_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        refpos: typing.ArrayOrFloat | None = None,
        mask: Sequence[int] | None = None,
        hard_constraint: bool = False,
        soft_constraint_cost: typing.ArrayOrFloat | None = None,
    ):
        super().__init__(name, gain, obj_name, obj_type, mask, hard_constraint, soft_constraint_cost)
        self.update_refpos(refpos if refpos is not None else jnp.zeros(3))
        self._dim = len(self._mask_idxs) if mask is not None else 3

    @property
    def refpos(self) -> jnp.ndarray:
        return self._refpos

    @refpos.setter
    def refpos(self, value: typing.ArrayOrFloat):
        self.update_refpos(value)

    def update_refpos(self, refpos: typing.ArrayOrFloat):
        self._refpos = jnp.array(refpos)
