"""
Euler integrators, both for general vector fields and for Hamiltonian dynamics.
"""

import torch
from torch import Tensor
from tqdm import tqdm

from .base import BatchedFunction, Integrator, _batched_jacobian, _validate_num_states
from .hamiltonian import Hamiltonian, HamiltonianIntegrator, HamiltonianToBatchedFunction


class EulerIntegrator(Integrator):
    """
    Euler integrator for Riemannian Hamiltonian Monte Carlo.

    Parameters:
    ----------
    f : BatchedFunction
        Fonction that takes a input (B, d) tensor and returns a (B,) tensor.
    gamma : float
        The step size for the Euler integrator.
    pbar : bool
        If True, display a progress bar during integration. Defaults to False.
    """

    def __init__(
        self, f: BatchedFunction, gamma: float, substeps: int = 1, pbar: bool = False
    ):
        super().__init__(f)
        self.f = f
        self.pbar = pbar
        self.substeps = substeps
        self.gamma = gamma / self.substeps

    def euler_step(self, x0: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform a single Euler step.

        Parameters
        ----------
        x0 : Tensor (b,d)
            The initial state.
        gamma : Tensor (b,1)
            The step size for each batch.

        Returns
        -------
        x_new : Tensor (b,d)
            The new state.
        """
        x_new = x0 + gamma * self.f(x0)
        return x_new

    def forward(
        self, x0: Tensor, L: int, return_traj: bool = False, dirs: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Performs L-1 Euler steps starting from (q_0, p_0).

        Parameters
        ----------
        x0 : Tensor (b,d)
            The initial state.
        L : int
            The number of Euler steps to perform.
        return_traj : bool
            If True, it returns the trajectory of the samples over the L Euler steps.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If None, all samples are integrated forward.

        Returns
        -------
        x_L : Tensor (b,d)
            The new state after L leapfrog steps.
        log_det : Tensor (b,)
            The log-determinant of the Jacobian of the transformation from x0 to x_L.
        or
        traj : Tensor (b,L,d)
            The trajectory of the states over the L leapfrog steps.
        log_det : Tensor (b,)
            The log-determinant of the Jacobian of the transformation from x0 to x_L.
        """
        _validate_num_states(L)
        if dirs is None:
            gamma = torch.full((x0.shape[0], 1), self.gamma, device=x0.device, dtype=x0.dtype)
        else:
            gamma = dirs.reshape(-1, 1).to(device=x0.device, dtype=x0.dtype) * self.gamma

        I = torch.eye(x0.shape[1], device=x0.device, dtype=x0.dtype)
        log_det = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)

        x1 = x0.clone()
        if return_traj:
            traj = [x0.clone().detach()]

        steps = range(L - 1)
        if self.pbar:
            steps = tqdm(steps, desc="Euler integration", unit="steps", leave=False)
        for k in steps:
            for _ in range(self.substeps):
                x0_step = x1
                x1 = self.euler_step(x1, gamma)
                jacobian = _batched_jacobian(self.f, x0_step)
                log_det += torch.linalg.slogdet(I + gamma.unsqueeze(-1) * jacobian)[1]

            if return_traj:
                if k == L - 2:
                    traj.append(x1.clone())
                else:
                    traj.append(x1.clone().detach())

        if return_traj:
            traj = torch.stack(traj, dim=1)
            return traj, log_det
        return x1, log_det


class HamiltonianEulerIntegrator(EulerIntegrator, HamiltonianIntegrator):
    """
    Euler integrator for Riemannian Hamiltonian Monte Carlo.

    Parameters:
    ----------
    H : Hamiltonian
        The Hamiltonian function H(q, p) that takes position q and momentum p and returns the energy.
    gamma : float
        The step size for the Euler integrator.
    substeps : int
        The number of substeps for the Euler integrator.
        This is the number of times the Euler step is applied to the same pair of states (q_0, p_0)
        and (q_1, p_1) before updating the states. This can be used to improve the stability of the integrator.
    pbar : bool
        If True, display a progress bar during integration. Defaults to False.
    """

    def __init__(self, H: Hamiltonian, gamma: float, substeps: int = 1, pbar: bool = False):
        # Initialize explicitly instead of through the cooperative
        # ``super().__init__`` chain: in this class' MRO the chain
        # ``EulerIntegrator -> Integrator -> HamiltonianIntegrator`` ends at
        # ``HamiltonianIntegrator.__init__``, which requires the Hamiltonian
        # and therefore raises ``TypeError`` when the chain is walked.
        torch.nn.Module.__init__(self)
        self.H = H
        self.f = HamiltonianToBatchedFunction(H)
        self.pbar = pbar
        self.substeps = substeps
        self.gamma = gamma / substeps

    def forward(
        self,
        q_0: Tensor,
        p_0: Tensor,
        L: int,
        return_traj: bool = False,
        dirs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Performs L-1 Euler steps starting from (q_0, p_0).

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        L : int
            The number of Euler steps to perform.
        return_traj : bool
            If True, it returns the trajectory of the samples over the L Euler steps.

        Returns
        -------
        q_L : Tensor (b,d)
            The new position after L leapfrog steps.
        p_L : Tensor (b,d)
            The new momentum after L leapfrog steps.
        or
        (Tensor (b,L,d), Tensor (b,L,d))
            The trajectory of the positions and momenta over the L leapfrog steps.
        """
        _validate_num_states(L)
        x_0 = torch.cat([q_0, p_0], dim=1)
        if return_traj:
            traj, logdet = super().forward(x_0, L, return_traj=True, dirs=dirs)
            q_traj, p_traj = torch.split(traj, traj.shape[-1] // 2, dim=-1)
            return q_traj, p_traj, logdet
        else:
            x_1, logdet = super().forward(x_0, L, return_traj=False, dirs=dirs)
            q_1, p_1 = torch.split(x_1, x_1.shape[-1] // 2, dim=-1)
            return q_1, p_1, logdet
