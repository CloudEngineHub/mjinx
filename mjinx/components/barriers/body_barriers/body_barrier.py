"""Frame task implementation."""

from typing import Callable

import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.barriers.base import Barrier, JaxBarrier
from mjinx.typing import Gain


@jdc.pytree_dataclass
class JaxBodyBarrier(JaxBarrier):
    r""""""

    body_id: jdc.Static[int]


class BodyBarrier[T: JaxBodyBarrier](Barrier[T]):
    __body_name: str
    __body_id: int

    def __init__(
        self,
        model: mjx.Model,
        gain: Gain,
        body_name: str,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
    ):
        super().__init__(model, gain, gain_fn, safe_displacement_gain)
        self.__body_name = body_name
        self.__body_id = mjx.name2id(
            model,
            mj.mjtObj.mjOBJ_BODY,
            self.__body_name,
        )
        if self.__body_id == -1:
            raise ValueError(f"body with name {self.__body_name} is not found.")

    @property
    def body_name(self) -> str:
        return self.__body_name

    @property
    def body_id(self) -> int:
        return self.__body_id
