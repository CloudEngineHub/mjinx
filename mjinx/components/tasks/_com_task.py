#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Stéphane Caron, Simeon Nedelchev, Ivan Domrachev

"""Center of mass task implementation."""

from typing import Callable, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import numpy as np
from typing_extensions import override

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxComTask(JaxTask):
    target_com: jnp.ndarray
    axes: jdc.Static[tuple[int, ...]]

    @final
    @override
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        r"""..."""
        error = data.subtree_com[self.model.body_rootid[0], self.axes] - self.target_com
        return error


class ComTask(Task[JaxComTask]):
    __target_com: jnp.ndarray
    __task_axes_str: str
    __task_axes_idx: tuple[int, ...]

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        axes: str = "xyz",
    ):
        super().__init__(name, cost, gain, gain_fn, lm_damping)
        self.target_com = jnp.zeros(3)
        self.__task_axes_str = axes
        self.__task_axes_idx = tuple([i for i in range(3) if "xyz"[i] in self.axes])
        self._dim = len(self.__task_axes_idx)

    @property
    def target_com(self) -> jnp.ndarray:
        return self.__target_com

    @target_com.setter
    def target_com(self, value: jnp.ndarray | np.ndarray):
        self.update_target_com(value)

    def update_target_com(self, target_com: jnp.ndarray | np.ndarray):
        if len(target_com) != len(self.__task_axes_idx):
            raise ValueError(
                "invalid dimension of the target CoM value: "
                f"{len(target_com)} given, expected {len(self.__task_axes_idx)} "
            )
        self._modified = True
        self.__target_com = target_com if isinstance(target_com, jnp.ndarray) else jnp.array(target_com)

    @property
    def task_axes(self) -> str:
        return self.__task_axes_str

    @final
    @override
    def _build_component(self) -> JaxComTask:
        return JaxComTask(
            dim=self._dim,
            model=self.model,
            cost=self.matrix_cost,
            gain=self.vector_gain,
            gain_function=self.gain_fn,
            lm_damping=self.lm_damping,
            target_com=self.target_com,
            axes=self.__task_axes_idx,
        )

    @final
    @override
    @property
    def empty(self) -> JaxComTask:
        return JaxComTask(
            dim=self.dim,
            model=self.model,  # despite model is not static, I see no point in batching through mjx.model
            gain_function=self.gain_fn,
            lm_damping=self.lm_damping,
            axes=self.__task_axes_idx,
            target_com=None,
            cost=None,
            gain=None,
        )
