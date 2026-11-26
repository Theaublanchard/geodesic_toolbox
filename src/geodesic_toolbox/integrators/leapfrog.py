"""
Leapfrog integrators for Hamiltonian dynamics.
"""

import torch
from torch import Tensor
from tqdm import tqdm

from .base import _validate_num_states
from .hamiltonian import Hamiltonian, HamiltonianIntegrator


class SeparableLeapfrogIntegrator(HamiltonianIntegrator):
    """
    Leapfrog integrator for separable Hamiltonians.
    This is usefull when the Hamiltonian is separable, i.e. when the kinetic energy does not depend on the position.

    Parameters:
    ----------
    H : Hamiltonian
        The Hamiltonian function H(q, p) that takes position q and momentum p and returns the energy.
    gamma : float
        The step size for the leapfrog integrator.
    substeps : int
        The number of substeps for the leapfrog integrator.
        This is the number of times the leapfrog step is applied to the same pair of states (q_0, p_0)
        and (q_1, p_1) before updating the states. This can be used to improve the stability of the integrator.
    pbar : bool
        If True, display a progress bar during integration. Defaults to False.
    """

    def __init__(
        self,
        H: Hamiltonian,
        gamma: float,
        substeps: int = 1,
        compile_step: bool = False,
        pbar: bool = False,
    ):
        super().__init__(H)
        self.pbar = pbar
        self.gamma = gamma / substeps
        self.substeps = substeps
        self.step = self.leapfrog_step
        if compile_step:
            self.step = torch.compile(self.step, mode="reduce-overhead")

    def leapfrog_step(self, q_0: Tensor, p_0: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
        """
        Leapfrog step for the Hamiltonian H.

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        gamma : Tensor (b,1)
            The step size for each batch.

        Returns
        -------
        q_1 : Tensor (b,d)
            The new position.
        p_1 : Tensor (b,d)
            The new momentum.
        """
        p_half = p_0 - gamma * self.H.dH_dq(q_0, p_0) / 2
        q_1 = q_0 + gamma * self.H.dH_dp(q_0, p_half)
        p_1 = p_half - gamma * self.H.dH_dq(q_1, p_half) / 2
        return q_1, p_1

    def forward(
        self,
        q_0: Tensor,
        p_0: Tensor,
        L: int,
        return_traj: bool = False,
        dirs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
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
            The new position after L leapfrog steps.
        p_L : Tensor (b,d)
            The new momentum after L leapfrog steps.
        or
        (Tensor (b,L,d), Tensor (b,L,d))
            The trajectory of the positions and momenta over the L leapfrog steps.
        """
        _validate_num_states(L)
        q_1, p_1 = q_0.clone(), p_0.clone()
        if return_traj:
            traj_q = [q_0.clone().detach()]
            traj_p = [p_0.clone().detach()]

        if dirs is None:
            gamma = torch.full(
                (q_0.shape[0], 1), self.gamma, device=q_0.device, dtype=q_0.dtype
            )
        else:
            gamma = dirs.reshape(-1, 1).to(device=q_0.device, dtype=q_0.dtype) * self.gamma

        steps = range(L - 1)
        if self.pbar:
            steps = tqdm(steps, desc="Leapfrog integration", unit="steps", leave=False)
        for k in steps:
            q_1, p_1 = self.step(q_1, p_1, gamma)

            if return_traj:
                if k == L - 2:
                    traj_q.append(q_1.clone())
                    traj_p.append(p_1.clone())
                else:
                    traj_q.append(q_1.clone().detach())
                    traj_p.append(p_1.clone().detach())

        log_det = torch.zeros(q_0.shape[0], device=q_0.device, dtype=q_0.dtype)
        if return_traj:
            traj_q = torch.stack(traj_q, dim=1)
            traj_p = torch.stack(traj_p, dim=1)
            return traj_q, traj_p, log_det
        return q_1, p_1, log_det


class ImplicitLeapfrogIntegrator(HamiltonianIntegrator):
    """
    Implicit leapfrog integrator.
    This is usefull when the Hamiltonian is not separable, i.e. when the kinetic energy depends on the position.
    The implicit updates are solved using fixed point iterations.

    Parameters:
    ----------
    f : BatchedFunction
        Fonction that takes a input (B, d) tensor and returns a (B,) tensor.
    gamma : float
        The step size for the leapfrog integrator.
    n_fix_pts : int
        The number of fixed point iterations to perform for the implicit equations.
    substeps : int
        The number of substeps for the leapfrog integrator.
        This is the number of times the leapfrog step is applied to the same pair of states (q_0, p_0)
        and (q_1, p_1) before updating the states. This can be used to improve the stability of the integrator.
    pbar : bool
        If True, display a progress bar during integration. Defaults to False.
    """

    def __init__(
        self,
        H: Hamiltonian,
        gamma: float,
        n_fix_pts: int,
        substeps: int = 1,
        compile_step: bool = False,
        pbar: bool = False,
    ):
        super().__init__(H)
        self.pbar = pbar
        self.n_fix_pts = n_fix_pts
        self.substeps = substeps
        self.gamma = gamma / self.substeps
        self.step = self.leapfrog_step
        if compile_step:
            self.step = torch.compile(self.step, mode="reduce-overhead")

    def get_p_half(self, q_0: Tensor, p_0: Tensor, gamma: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the momentum:
        p_half = p_0 - gamma/2 * dH_dq(q_0, p_half)

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        gamma : Tensor (b,1)
            The step size for each batch.

        Returns
        -------
        p_half : Tensor (b,d)
            The half step momentum.
        """
        p_half = p_0.clone()
        for k in range(self.n_fix_pts):
            p_half_ = p_0 - gamma * self.H.dH_dq(q_0, p_half) / 2
            # if (p_half_ - p_half).abs().max() < 1e-6:
            #     p_half = p_half_
            #     break
            p_half = p_half_
        return p_half

    def get_q_new(self, q_0: Tensor, p_half: Tensor, gamma: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the position:
        q_new = q_0 + gamma/2 * ( dH_dp(q_0, p_half) + dH_dp(q_new,p_half) )

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_half : Tensor (b,d)
            The half step momentum.
        gamma : Tensor (b,1)
            The step size for each batch.

        Returns
        -------
        q_new : Tensor (b,d)
            The new position.
        """
        q_new = q_0.clone()
        dH_dp_0 = self.H.dH_dp(q_0, p_half)
        for k in range(self.n_fix_pts):
            q_new_ = q_0 + gamma * (dH_dp_0 + self.H.dH_dp(q_new, p_half)) / 2
            # if (q_new_ - q_new).abs().max() < 1e-6:
            #     q_new = q_new_
            #     break
            q_new = q_new_
        return q_new

    def leapfrog_step(self, q_0: Tensor, p_0: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
        """
        Leapfrog step for the Hamiltonian H.

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        gamma : Tensor (b,1)
            The step size for each batch.

        Returns
        -------
        q_1 : Tensor (b,d)
            The new position.
        p_1 : Tensor (b,d)
            The new momentum.
        """
        q_1 = q_0.clone()
        p_1 = p_0.clone()
        for _ in range(self.substeps):
            p_half = self.get_p_half(q_1, p_1, gamma)
            q_1 = self.get_q_new(q_1, p_half, gamma)
            p_1 = p_half - gamma * self.H.dH_dq(q_1, p_half) / 2
        return q_1, p_1

    def forward(
        self,
        q_0: Tensor,
        p_0: Tensor,
        L: int,
        return_traj: bool = False,
        dirs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
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
            The new position after L leapfrog steps.
        p_L : Tensor (b,d)
            The new momentum after L leapfrog steps.
        or
        (Tensor (b,L,d), Tensor (b,L,d))
            The trajectory of the positions and momenta over the L leapfrog steps.
        """
        q_1, p_1 = q_0.clone(), p_0.clone()
        if return_traj:
            traj_q = [q_0.clone().detach()]
            traj_p = [p_0.clone().detach()]

        if dirs is None:
            gamma = torch.full(
                (q_0.shape[0], 1), self.gamma, device=q_0.device, dtype=q_0.dtype
            )
        else:
            gamma = dirs.reshape(-1, 1).to(device=q_0.device, dtype=q_0.dtype) * self.gamma

        steps = range(L - 1)
        if self.pbar:
            steps = tqdm(steps, desc="Leapfrog integration", unit="steps", leave=False)
        for k in steps:
            q_1, p_1 = self.step(q_1, p_1, gamma)

            if return_traj:
                if k == L - 2:
                    traj_q.append(q_1.clone())
                    traj_p.append(p_1.clone())
                else:
                    traj_q.append(q_1.clone().detach())
                    traj_p.append(p_1.clone().detach())

        log_det = torch.zeros(q_0.shape[0], device=q_0.device, dtype=q_0.dtype)
        if return_traj:
            traj_q = torch.stack(traj_q, dim=1)
            traj_p = torch.stack(traj_p, dim=1)
            return traj_q, traj_p, log_det
        return q_1, p_1, log_det


class ExplicitLeapfrogIntegrator(HamiltonianIntegrator):
    """
    Explicit leapfrog integrator for Riemannian Hamiltonian Monte Carlo.
    See the paper :
    `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`.

    Parameters:
    ----------
    H : Hamiltonian
        The Hamiltonian function H(q, p) that takes position q and momentum p and returns the energy.
    gamma : float
        The step size for the leapfrog integrator.
    omega : float
        The binding parameter for the leapfrog integrator.
    substeps : int
        The number of substeps for the leapfrog integrator.
        This is the number of times the leapfrog step is applied to the same pair of states (q_0, p_0)
        and (q_1, p_1) before updating the states. This can be used to improve the stability of the integrator.
    pbar : bool
        If True, display a progress bar during integration. Defaults to False.
    """

    def __init__(
        self,
        H: Hamiltonian,
        gamma: float,
        omega: float,
        substeps: int = 1,
        compile_step: bool = False,
        pbar: bool = False,
    ):
        super().__init__(H)
        self.pbar = pbar
        self.substeps = substeps
        self.gamma = gamma / self.substeps
        self.omega = omega
        self.step = self.leapfrog_step
        if compile_step:
            self.step = torch.compile(self.step, mode="reduce-overhead")

        c = torch.Tensor([2 * self.omega * self.gamma]).cos()
        s = torch.Tensor([2 * self.omega * self.gamma]).sin()
        self.register_buffer("c", c, persistent=False)
        self.register_buffer("s", s, persistent=False)

    def binding(self, q_0: Tensor, p_0: Tensor, q_1: Tensor, p_1: Tensor) -> Tensor:
        """
        Compute the binding energy between two states.

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The position of the first state.
        p_0 : Tensor (b,d)
            The momentum of the first state.
        q_1 : Tensor (b,d)
            The position of the second state.
        p_1 : Tensor (b,d)
            The momentum of the second state.

        Returns
        -------
        Tensor (b,)
            The binding energy.
        """
        h = torch.linalg.vector_norm(q_1 - q_0, dim=-1) ** 2 / 2
        h += torch.linalg.vector_norm(p_1 - p_0, dim=-1) ** 2 / 2
        return h

    def H_augmented(self, q_0: Tensor, p_0: Tensor, q_1: Tensor, p_1: Tensor) -> Tensor:
        """
        Compute the augmented Hamiltonian H(q_0, p_0, q_1, p_1) = H(q_0, p_0) + H(q_1, p_1) + omega * binding(q_0, p_0, q_1, p_1)

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The position of the first state.
        p_0 : Tensor (b,d)
            The momentum of the first state.
        q_1 : Tensor (b,d)
            The position of the second state.
        p_1 : Tensor (b,d)
            The momentum of the second state.

        Returns
        -------
        Tensor (b,)
            The augmented Hamiltonian.
        """
        H_0 = self.H(q_0, p_0)
        H_1 = self.H(q_1, p_1)
        H = H_0 + H_1 + self.omega * self.binding(q_0, p_0, q_1, p_1)
        return H

    def leapfrog_step(
        self,
        q_0: Tensor,
        p_0: Tensor,
        q_1: Tensor,
        p_1: Tensor,
        gamma: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Leapfrog step for the augmented Hamiltonian.
        Pseudo code in `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`
        by Cobb et Baydin et al (2019).

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The position of the first state.
        p_0 : Tensor (b,d)
            The momentum of the first state.
        q_1 : Tensor (b,d)
            The position of the second state.
        p_1 : Tensor (b,d)
            The momentum of the second state.
        gamma : Tensor (b,1) | None
            The step size for each batch. If None, it uses self.gamma.
            Used when we want to integrate in different directions for each batch.

        Returns
        -------
        q_0_new : Tensor (b,d)
            The new position of the first state.
        p_0_new : Tensor (b,d)
            The new momentum of the first state.
        q_1_new : Tensor (b,d)
            The new position of the second state.
        p_1_new : Tensor (b,d)
            The new momentum of the second state.
        """
        if gamma is None:
            gamma = torch.full(
                (q_0.shape[0], 1), self.gamma, device=q_0.device, dtype=q_0.dtype
            )
        c = torch.cos(2 * self.omega * gamma)
        s = torch.sin(2 * self.omega * gamma)

        p_0_new = p_0 - gamma / 2 * self.H.dH_dq(q_0, p_1)
        q_1_new = q_1 + gamma / 2 * self.H.dH_dp(q_0, p_1)
        p_1_new = p_1 - gamma / 2 * self.H.dH_dq(q_1_new, p_0)
        q_0_new = q_0 + gamma / 2 * self.H.dH_dp(q_1_new, p_0)

        # Apply the binding map simultaneously from the same pre-rotation state.
        q0_pre, p0_pre = q_0_new, p_0_new
        q1_pre, p1_pre = q_1_new, p_1_new

        q_0_new = (q0_pre + q1_pre + c * (q0_pre - q1_pre) + s * (p0_pre - p1_pre)) / 2
        p_0_new = (p0_pre + p1_pre - s * (q0_pre - q1_pre) + c * (p0_pre - p1_pre)) / 2
        q_1_new = (q0_pre + q1_pre - c * (q0_pre - q1_pre) - s * (p0_pre - p1_pre)) / 2
        p_1_new = (p0_pre + p1_pre + c * (q0_pre - q1_pre) - s * (p0_pre - p1_pre)) / 2

        p_1_new = p_1_new - gamma / 2 * self.H.dH_dq(q_1_new, p_0_new)
        q_0_new = q_0_new + gamma / 2 * self.H.dH_dp(q_1_new, p_0_new)
        p_0_new = p_0_new - gamma / 2 * self.H.dH_dq(q_0_new, p_1_new)
        q_1_new = q_1_new + gamma / 2 * self.H.dH_dp(q_0_new, p_1_new)

        return q_0_new, p_0_new, q_1_new, p_1_new

    @torch.no_grad()
    def forward(
        self,
        q_0: Tensor,
        p_0: Tensor,
        L: int,
        return_traj: bool = False,
        dirs: Tensor | None = None,
    ):
        """Integrate identical initial copies and return the first copy.

        The returned `log_det` is the Jacobian of the full two-copy
        augmented map used by `forward_augmented`, which is zero. It is
        not the Jacobian of the projected first-copy map alone.

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
            Otherwise, it returns the final state after L steps.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If None, all samples are integrated forward.

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
        result = self.forward_augmented(
            q_0, p_0, L, return_traj=return_traj, q_1=q_0, p_1=p_0, dirs=dirs
        )
        if return_traj:
            traj_q_0, traj_p_0, _, _, log_det = result
            return traj_q_0, traj_p_0, log_det
        q_0, p_0, _, _, log_det = result
        return q_0, p_0, log_det

    @torch.no_grad()
    def forward_augmented(
        self,
        q_0: Tensor,
        p_0: Tensor,
        L: int,
        q_1: Tensor,
        p_1: Tensor,
        return_traj: bool = False,
        dirs: Tensor | None = None,
    ):
        """
        Perform L-1 leapfrog steps with the augmented Hamiltonian.

        This explicit Riemannian HMC scheme evolves two coupled copies of the
        Hamiltonian state, ``(q_0, p_0, q_1, p_1)``. The regular ``forward``
        method uses identical copies for callers that only provide one state;
        this method is used when the two copies are maintained separately.

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
        q_1 : Tensor (b,d)
            The initial position of the second copy.
        p_1 : Tensor (b,d)
            The initial momentum of the second copy.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If None, all samples are integrated forward.

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
        q_1 = q_1.clone()
        p_1 = p_1.clone()
        if return_traj:
            traj_q = [q_0.clone().detach()]
            traj_p = [p_0.clone().detach()]
            traj_q_1 = [q_1.clone().detach()]
            traj_p_1 = [p_1.clone().detach()]

        if dirs is None:
            gamma = torch.full(
                (q_0.shape[0], 1), self.gamma, device=q_0.device, dtype=q_0.dtype
            )
        else:
            gamma = dirs.reshape(-1, 1).to(device=q_0.device, dtype=q_0.dtype) * self.gamma

        integration_failed = False
        steps = range(L - 1)
        if self.pbar:
            steps = tqdm(steps, desc="Leapfrog steps", unit="steps", leave=False)
        for k in steps:
            for _ in range(self.substeps):
                q_0, p_0, q_1, p_1 = self.step(q_0, p_0, q_1, p_1, gamma)

                if (
                    not torch.isfinite(q_0).all()
                    or not torch.isfinite(p_0).all()
                    or not torch.isfinite(q_1).all()
                    or not torch.isfinite(p_1).all()
                ):
                    integration_failed = True
                    break

            if integration_failed:
                break

            if return_traj:
                # Keep graph only on the final point.
                if k == L - 2:
                    traj_q.append(q_0.clone())
                    traj_p.append(p_0.clone())
                    traj_q_1.append(q_1.clone())
                    traj_p_1.append(p_1.clone())
                else:
                    traj_q.append(q_0.clone().detach())
                    traj_p.append(p_0.clone().detach())
                    traj_q_1.append(q_1.clone().detach())
                    traj_p_1.append(p_1.clone().detach())

        if return_traj:
            traj_q = torch.stack(traj_q, dim=1)
            traj_p = torch.stack(traj_p, dim=1)
            traj_q_1 = torch.stack(traj_q_1, dim=1)
            traj_p_1 = torch.stack(traj_p_1, dim=1)
            log_det = torch.zeros(q_0.shape[0], device=q_0.device, dtype=q_0.dtype)
            return traj_q, traj_p, traj_q_1, traj_p_1, log_det
        log_det = torch.zeros(q_0.shape[0], device=q_0.device, dtype=q_0.dtype)
        return q_0, p_0, q_1, p_1, log_det
