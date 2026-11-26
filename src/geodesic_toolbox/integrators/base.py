"""
General integrators for ODEs of the form dx/dt = f(x).
"""

from collections.abc import Callable

import torch
from torch import Tensor

##############################
# General integrators for ODEs of the form dx/dt = f(x)
##############################
# Function that takes a input (B, d) tensor and returns a (B,) tensor.
BatchedFunction = Callable[[Tensor], Tensor]


def _validate_num_states(L: int) -> None:
    if L <= 1:
        raise ValueError("L must be greater than 1.")


def _batched_jacobian(f: BatchedFunction, x: Tensor) -> Tensor:
    single_f = lambda x_i: f(x_i.unsqueeze(0)).squeeze(0)
    return torch.vmap(torch.func.jacrev(single_f))(x)


class Integrator(torch.nn.Module):
    """
    Base class for integrators.

    Parameters
    ----------
    f : BatchedFunction
        Fonction that takes a input (B, d) tensor and returns a (B,) tensor.
    """

    def __init__(self, f: BatchedFunction, *args, **kwargs):
        super().__init__()
        self.f = f

    def forward(
        self, x0: Tensor, L: int, return_traj: bool = False, dirs: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Performs L-1 integration steps starting from x0.

        Parameters
        ----------
        x0 : Tensor (b,d)
            The initial state.
        L : int
            The number of integration steps to perform.
        return_traj : bool
            If True, it returns the trajectory of the samples over the L integration steps.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If None, all samples are integrated forward.

        Returns
        -------
        x_L : Tensor (b,d)
            The new state after L integration steps.
        log_det : Tensor (b,)
            The log-determinant of the Jacobian of the transformation from x0 to x_L.
        or
        traj : Tensor (b,L,d)
            The trajectory of the states over the L integration steps.
        log_det : Tensor (b,)
            The log-determinant of the Jacobian of the transformation from x0 to x_L.
        """
        raise NotImplementedError(
            "The forward method must be implemented by inheriting this class."
        )
