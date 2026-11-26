"""
Integrators for Hamiltonian dynamics dx/dt = J * grad H(x),
which we write using the position q and momentum p as
dq/dt = dH/dp, dp/dt = -dH/dq.
"""

import torch
from torch import Tensor

from .base import _batched_jacobian


class Hamiltonian(torch.nn.Module):
    """
    Hamiltonian function for Riemannian Hamiltonian Monte Carlo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(q,p)

        Parameters
        ----------
        q : Tensor (b,d)
            The position.
        p : Tensor (b,d)
            The momentum.

        Returns
        -------
        Tensor (b,)
        """
        raise NotImplementedError(
            "The Hamiltonian function must be implemented by inheriting this class."
        )

    @torch.enable_grad()
    def dH_dq(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Compute the gradient of the Hamiltonian with respect to the position.

        Parameters
        ----------
        q : Tensor (b,d)
            The position.
        p : Tensor (b,d)
            The momentum.

        Returns
        -------
        Tensor (b,d)
            The gradient of the Hamiltonian with respect to the position.
        """
        return torch.func.grad(lambda q: self.forward(q, p).sum(), argnums=0)(q)

    @torch.enable_grad()
    def dH_dp(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Compute the gradient of the Hamiltonian with respect to the momentum.

        Parameters
        ----------
        q : Tensor (b,d)
            The position.
        p : Tensor (b,d)
            The momentum.

        Returns
        -------
        Tensor (b,d)
            The gradient of the Hamiltonian with respect to the momentum.
        """
        return torch.func.grad(lambda p: self.forward(q, p).sum(), argnums=0)(p)


class HamiltonianToBatchedFunction(torch.nn.Module):
    """
    Class to convert a Hamiltonian function H(q,p) to a batched function f(x) = J * grad H(x) for use in general integrators.
    """

    def __init__(self, H: Hamiltonian):
        super().__init__()
        self.H = H

    @torch.enable_grad()
    def forward(self, x: Tensor) -> Tensor:
        q, p = torch.split(x, x.shape[1] // 2, dim=1)
        dH_dq = self.H.dH_dq(q, p)
        dH_dp = self.H.dH_dp(q, p)
        # Compute f(x) = J * grad H(x) = [dH/dp, -dH/dq]
        f_x = torch.cat([dH_dp, -dH_dq], dim=1)
        return f_x

    @torch.enable_grad()
    def df(self, x):
        """
        Compute the Jacobian of f(x) = J * grad H(x) with respect to x.

        Parameters
        ----------
        x : Tensor (b,d)
            The input state.

        Returns
        -------
        Tensor (b,d,d)
            The Jacobian of f with respect to x.
        """
        jacobian = _batched_jacobian(self.forward, x)
        return jacobian


class HamiltonianIntegrator(torch.nn.Module):
    """
    Base class for Hamiltonian integrators.

    Parameters
    ----------
    H : Hamiltonian
        The Hamiltonian function H(q, p) that takes position q and momentum p and returns the energy.
    """

    def __init__(self, H: Hamiltonian, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.H = H

    def forward(
        self,
        q_0: Tensor,
        p_0: Tensor,
        L: int,
        return_traj: bool = False,
        dirs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Performs L-1 leapfrog steps starting from (q_0, p_0).

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        L : int
            The number of leapfrog steps to perform.
        return_traj : bool
            If True, it returns the trajectory of the samples over the L leapfrog steps.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If None, all samples are integrated forward.

        Returns
        -------
        q_L : Tensor (b,d)
            The new position after L-1 leapfrog steps.
        p_L : Tensor (b,d)
            The new momentum after L-1 leapfrog steps.
        log_det : Tensor (b,)
            The log-determinant of the transformation from (q_0, p_0) to
            (q_L, p_L).
        or
        q_traj : Tensor (b,L,d)
            The trajectory of the positions over the L states.
        p_traj : Tensor (b,L,d)
            The trajectory of the momenta over the L states.
        log_det : Tensor (b,)
            The accumulated log-determinant over the trajectory.
        """
        raise NotImplementedError(
            "The forward method must be implemented by inheriting this class."
        )
