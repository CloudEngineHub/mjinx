"""Center of mass task implementation."""

from typing import Callable, Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.configuration import get_joint_zero, joint_difference
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxJointTask(JaxTask):
    """
    A JAX-based implementation of a joint task for inverse kinematics.

    This class represents a task that aims to achieve specific target joint positions
    for the robot model.

    :param target_q: The target joint positions to be achieved.
    """

    target_q: jnp.ndarray

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the error between the current joint positions and the target joint positions.

        :param data: The MuJoCo simulation data.
        :return: The error vector representing the difference between the current and target joint positions.
        """
        return joint_difference(self.model, data.qpos, self.target_q)[self.mask_idxs,]


class JointTask(Task[JaxJointTask]):
    """
    A high-level representation of a joint task for inverse kinematics.

    This class provides an interface for creating and manipulating joint tasks,
    which aim to achieve specific target joint positions for the robot model.

    :param name: The name of the task.
    :param cost: The cost associated with the task.
    :param gain: The gain for the task.
    :param gain_fn: A function to compute the gain dynamically.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param mask: A sequence of integers to mask certain dimensions of the task.
    """

    JaxComponentType: type = JaxJointTask
    __target_q: jnp.ndarray | None

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, cost, gain, gain_fn, lm_damping, mask)
        self.__target_q = None

    def update_model(self, model: mjx.Model):
        """
        Update the MuJoCo model and set the joint dimensions for the task.

        This method is called when the model is updated or when the task
        is first added to the problem.

        :param model: The new MuJoCo model.
        :raises ValueError: If the provided mask is invalid for the model or if the target_q is incompatible.
        """
        super().update_model(model)

        self._dim = model.nq
        if len(self.mask) != self._dim:
            raise ValueError("provided mask in invalid for the model")
        if len(self.mask_idxs) != self._dim:
            self._dim = len(self.mask_idxs)

        # Validate current target_q, if empty -- set via default value
        if self.__target_q is None:
            self.target_q = get_joint_zero(model)[self.mask_idxs,]
        elif self.target_q.shape[-1] != self._dim:
            raise ValueError(
                "provided model is incompatible with target q: "
                f"{len(self.target_q)} is set, model expects {self._dim}."
            )

    @property
    def target_q(self) -> jnp.ndarray:
        """
        Get the current target joint positions for the task.

        :return: The current target joint positions as a numpy array.
        :raises ValueError: If the target value was not provided and the model is missing.
        """
        if self.__target_q is None:
            raise ValueError("target value was neither provided, nor deduced from other arguments (model is missing)")
        return self.__target_q

    @target_q.setter
    def target_q(self, value: Sequence):
        """
        Set the target joint positions for the task.

        :param value: The new target joint positions as a sequence of values.
        """
        self.update_target_q(value)

    def update_target_q(self, target_q: Sequence):
        """
        Update the target joint positions for the task.

        This method allows setting the target joint positions using a sequence of values.

        :param target_q: The new target joint positions as a sequence of values.
        :raises ValueError: If the provided sequence doesn't have the correct length.
        """
        target_q_jnp = jnp.array(target_q)
        if self._dim != -1 and target_q_jnp.shape[-1] != self._dim:
            raise ValueError(
                f"dimension mismatch: expected last dimension to be {self._dim}, got{target_q_jnp.shape[-1]}"
            )
        self.__target_q = target_q_jnp
