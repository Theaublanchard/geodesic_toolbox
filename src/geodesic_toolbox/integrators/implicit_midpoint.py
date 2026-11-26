"""
Implicit midpoint integrator, both for general vector fields and for Hamiltonian dynamics.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from tqdm import tqdm

from .base import BatchedFunction, _validate_num_states
from .hamiltonian import (
    Hamiltonian,
    HamiltonianIntegrator,
    HamiltonianToBatchedFunction,
)


class ImplicitMidpointIntegrator(torch.nn.Module):
    """
    Implicit midpoint integrator x1 = x0 + gamma * f((x0 + x1) / 2) for an
    arbitrary vector field f. The step is symmetric (Phi_{-gamma} = Phi_gamma^{-1})
    and the exact log-Jacobian of each discrete step,

        Delta log J = log|det(I + gamma/2 D)| - log|det(I - gamma/2 D)|,
        D = df((x0 + x1) / 2),

    is accumulated so it can be included in a Metropolis ratio.

    Parameters
    ----------
    f : Callable
        Vector field, maps (b, n) to (b, n).
    df : Callable
        Jacobian of the field, maps (b, n) to (b, n, n).
    gamma : float
        Step size.
    N_fx : int
        Number of fixed-point (picard) or Newton iterations per step.
    method : str
        "picard" or "newton".
    jacobian : str
        "estimate" or "exact".
    substeps : int
        Number of substeps for the implicit midpoint step.
        This is the number of times the implicit midpoint step is applied
        to the same pair of states (x0, x1) before updating the states.
    pbar : bool
        If True, display a progress bar during integration. Defaults to False.
    # The following parameters are only used if jacobian="estimate":
    jacobian_mc : int
        Number of Monte Carlo probes for estimating the log-Jacobian using Hutchinson's estimator.
    russian_roulette : float
        Probability of terminating the series for estimating the log-Jacobian.
    """

    def __init__(
        self,
        f: BatchedFunction,
        df: Callable[[Tensor], Tensor],
        gamma: float,
        N_fx: int,
        method: str = "picard",
        jacobian: str = "exact",
        substeps: int = 1,
        jacobian_mc: int = 1,
        russian_roulette: float = 0.5,
        compile_step: bool = False,
        pbar: bool = False,
    ):
        torch.nn.Module.__init__(self)

        self.f = f
        self.df = df
        self.N_fx = N_fx
        self.method = method
        self.jacobian = jacobian
        self.substeps = substeps
        self.gamma = gamma / self.substeps
        self.russian_roulette = russian_roulette
        self.jacobian_mc = jacobian_mc
        self.pbar = pbar

        if jacobian == "estimate":
            self.log_det_jac = self.estimate_log_det_jac
        elif jacobian == "exact":
            self.log_det_jac = self.exact_log_det_jac
        else:
            raise ValueError(
                f"Unknown jacobian method {jacobian}. Choose 'estimate' or 'exact'."
            )

        if method == "picard":
            self.implicit_midpoint_step = self.picard
        elif method == "newton":
            self.implicit_midpoint_step = self.newton
        else:
            raise ValueError(f"Unknown method {method}. Choose 'picard' or 'newton'.")

        self.step = self.implicit_midpoint_step
        if compile_step:
            self.step = torch.compile(self.step, mode="reduce-overhead")

    def picard(self, x0: Tensor, gamma: Tensor, tol: float = 1e-8) -> Tensor:
        """
        Apply Picard fixed-point iteration to solve the implicit midpoint equation
        x1 = x0 + gamma * f((x0 + x1) / 2).
        Iteration stops when the maximum absolute change is below tol or after self.N_fx iterations.

        Parameters
        ----------
        x0 : Tensor (b, n)
            The initial state.
        gamma : Tensor (b, 1)
            The step size for each batch.
        tol : float
            Tolerance for convergence of the fixed-point iteration.
        """
        x1 = x0.clone()
        for _ in range(self.N_fx):
            x_mid = (x1 + x0) / 2
            x1_ = x0 + gamma * self.f(x_mid)
            delta = (x1_ - x1).abs().max()
            x1 = x1_
            if delta < tol:
                break
        return x1

    def newton(self, x0: Tensor, gamma: Tensor) -> Tensor:
        """
        Apply Newton's method to solve the implicit midpoint equation
        x1 = x0 + gamma * f((x0 + x1) / 2). The Jacobian is computed at the midpoint (x0 + x1) / 2.
        Iteration stops after self.N_fx iterations.

        Parameters
        ----------
        x0 : Tensor (b, n)
            The initial state.
        gamma : Tensor (b, 1)
            The step size for each batch.

        Returns
        -------
        x1 : Tensor (b, n)
            The state at the next time step.
        """
        x1 = x0.clone()
        I = torch.eye(x0.shape[-1], device=x0.device, dtype=x0.dtype)
        for _ in range(self.N_fx):
            x_mid = (x1 + x0) / 2
            D = self.df(x_mid)
            residual = x1 - x0 - gamma * self.f(x_mid)
            J = I - 0.5 * gamma.unsqueeze(-1) * D
            x1 = x1 - torch.linalg.solve(J, residual)
        return x1

    def exact_log_det_jac(self, x_mid: Tensor, gamma: Tensor) -> Tensor:
        """
        Compute the exact log-Jacobian of the implicit midpoint step.
        log|det(I + gamma/2 D)| - log|det(I - gamma/2 D)|, where D = df(x_mid).

        Parameters
        ----------
        x_mid : Tensor (b, n)
            The midpoint state (x0 + x1) / 2.
        gamma : Tensor (b, 1)
            The step size for each batch.

        Returns
        -------
        delta : Tensor (b,)
            The log-Jacobian of the implicit midpoint step.
        """
        D = self.df(x_mid)
        I = torch.eye(D.shape[-1], device=D.device, dtype=D.dtype)
        gamma_D = 0.5 * gamma.unsqueeze(-1) * D
        _, logabsdet_plus = torch.linalg.slogdet(I + gamma_D)
        _, logabsdet_minus = torch.linalg.slogdet(I - gamma_D)
        delta = logabsdet_plus - logabsdet_minus
        return delta

    # @TODO: optimize this code
    def estimate_log_det_jac(self, x_mid: Tensor, gamma: Tensor) -> Tensor:
        """
        Unbiased matrix-free estimate of the step log-Jacobian

            Delta log J = 2 sum_{j>=0} tr(A^{2j+1}) / (2j+1),  A = gamma/2 df(x_mid),

        Traces use Hutchinson's estimator with self.jacobian_mc Rademacher
        probes and JVPs of f (the Jacobian is never formed); the series is
        truncated by russian roulette with a Geometric(self.russian_roulette)
        number of terms, each reweighted by its survival probability.
        Converges for spectral radius of A below 1 (small gamma).

        Parameters
        ----------
        x_mid : Tensor (b, n)
            The midpoint state (x0 + x1) / 2.
        gamma : Tensor (b, 1)
            The step size for each batch.

        Returns
        -------
        delta : Tensor (b,)
            The estimated log-Jacobian of the implicit midpoint step.
        """
        b = x_mid.shape[0]
        # Number of odd-order terms, shared across the batch.
        N = int(torch.empty(1).geometric_(self.russian_roulette).item())

        jvp_fn = lambda w: torch.func.jvp(self.f, (x_mid,), (w,))[1]
        half_gamma = 0.5 * gamma  # (b, 1)
        q = 1.0 - self.russian_roulette
        delta = torch.zeros(b, device=x_mid.device, dtype=x_mid.dtype)
        for _ in range(self.jacobian_mc):
            # Rademacher probe (b, n).
            v = (
                torch.randint(0, 2, x_mid.shape, device=x_mid.device)
                .to(dtype=x_mid.dtype)
                .mul_(2)
                .sub_(1)
            )
            w = half_gamma * jvp_fn(v)  # A^1 v
            for j in range(N):
                k = 2 * j + 1
                if j > 0:
                    # Advance from A^{k-2} v to A^k v with two JVPs.
                    w = half_gamma * jvp_fn(half_gamma * jvp_fn(w))
                # Hutchinson estimate of tr(A^k).
                trace_k = (v * w).sum(dim=-1)
                delta = delta + trace_k / (k * q**j)
        return 2.0 * delta / self.jacobian_mc

    def forward(
        self, x0: Tensor, L: int, return_traj: bool = False, dirs: Tensor | None = None
    ):
        """
        Integrate the vector field f using the implicit midpoint method
        starting from initial state x0 for L-1 steps of size self.gamma.

        Parameters
        ----------
        x0 : Tensor (b, n)
            Initial state.
        L : int
            Number of integration steps.
        return_traj : bool
            If True, return the full trajectory (b, L, n) instead of the
            final state.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If
            None, all samples are integrated forward.

        Returns
        -------
        x_L : Tensor (b, n)
            Final state after L-1 integration steps.
        log_det : Tensor (b,)
            Accumulated log-Jacobian of the integration steps.

        or
        traj : Tensor (b, L, n)
            Full trajectory of states over the integration steps.
        log_det : Tensor (b,)
            Accumulated log-Jacobian of the integration steps.
        """
        _validate_num_states(L)
        if dirs is None:
            gamma = torch.full((x0.shape[0], 1), self.gamma, device=x0.device, dtype=x0.dtype)
        else:
            gamma = self.gamma * dirs.reshape(-1, 1).to(device=x0.device, dtype=x0.dtype)

        log_det = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)

        if return_traj:
            traj = [x0.clone()]

        steps = range(L - 1)
        if self.pbar:
            steps = tqdm(
                steps, desc="Implicit midpoint integration", unit="steps", leave=False
            )
        for k in steps:
            for _ in range(self.substeps):
                x1 = self.step(x0, gamma)
                delta = self.log_det_jac((x0 + x1) / 2, gamma)
                x0 = x1
                log_det += delta

            if return_traj:
                if k == L - 2:
                    traj.append(x0.clone())
                else:
                    traj.append(x0.clone().detach())

        if return_traj:
            traj = torch.stack(traj, dim=1)
            return traj, log_det

        return x0, log_det


class HamiltonianImplicitMidpointIntegrator(ImplicitMidpointIntegrator, HamiltonianIntegrator):
    """
    Implicit midpoint integrator for Riemannian Hamiltonian Monte Carlo.

    Parameters:
    ----------
    H : Hamiltonian
        The Hamiltonian function H(q, p) that takes position q and momentum p and returns the energy.
    gamma : float
        The step size for the implicit midpoint integrator.
    N_fx : int
        The number of fixed point iterations to perform for the implicit equations.
    method : str
        The method to use for the implicit midpoint integrator. Can be "picard" or "newton".
    jacobian : str
        The method to use for computing the log-Jacobian. Can be "exact" or "estimate".
    substeps : int
        The number of substeps for the implicit midpoint integrator.
        This is the number of times the implicit midpoint step is applied to the same pair of states (q_0, p_0)
        and (q_1, p_1) before updating the states. This can be used to improve the stability of the integrator.
    jacobian_mc : int
        The number of Monte Carlo samples to use for estimating the log-Jacobian when using the "estimate" method.
    russian_roulette : float
        The probability of terminating the series for estimating the log-Jacobian when using the "estimate" method.
    pbar : bool
        If True, display a progress bar during integration. Defaults to False.
    """

    def __init__(
        self,
        H: Hamiltonian,
        gamma: float,
        N_fx: int,
        method: str = "picard",
        jacobian: str = "exact",
        substeps: int = 1,
        jacobian_mc: int = 1,
        russian_roulette: float = 0.5,
        compile_step: bool = False,
        pbar: bool = False,
    ):
        super().__init__(
            HamiltonianToBatchedFunction(H),
            HamiltonianToBatchedFunction(H).df,
            gamma,
            N_fx,
            method=method,
            jacobian=jacobian,
            substeps=substeps,
            jacobian_mc=jacobian_mc,
            russian_roulette=russian_roulette,
            compile_step=compile_step,
            pbar=pbar,
        )

        self.H = H
        self.log_det_jac = self.zero_log_det_jac

    def zero_log_det_jac(self, x_mid: Tensor, gamma: Tensor) -> Tensor:
        """
        Return a zero log-Jacobian for the implicit midpoint step.
        Because for the Hamiltonian dynamics, the implicit midpoint
        integrator is symplectic and volume-preserving, so the log-Jacobian is zero.

        Parameters
        ----------
        x_mid : Tensor (b, n)
            The midpoint state (x0 + x1) / 2.
        gamma : Tensor (b, 1)
            The step size for each batch.
            Used for API compatibility, not used in the computation.

        Returns
        -------
        delta : Tensor (b,)
            A tensor of zeros with shape (b,).
        """
        return torch.zeros(x_mid.shape[0], device=x_mid.device, dtype=x_mid.dtype)

    def forward(
        self,
        q_0: Tensor,
        p_0: Tensor,
        L: int,
        return_traj: bool = False,
        dirs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform L-1 implicit midpoint steps starting from (q_0, p_0).

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        L : int
            The number of implicit midpoint steps to perform.
        return_traj : bool
            If True, it returns the trajectory of the samples over the L implicit midpoint steps.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If None, all samples are integrated forward.

        Returns
        -------
        q_L : Tensor (b,d)
            The new position after L implicit midpoint steps.
        p_L : Tensor (b,d)
            The new momentum after L implicit midpoint steps.
        or
        (Tensor (b,L,d), Tensor (b,L,d))
            The trajectory of the positions and momenta over the L implicit midpoint steps.
        """
        x_0 = torch.cat([q_0, p_0], dim=1)
        if return_traj:
            traj, logdet = super().forward(x_0, L, return_traj=True, dirs=dirs)
            q_traj, p_traj = torch.split(traj, traj.shape[-1] // 2, dim=-1)
            return q_traj, p_traj, logdet
        else:
            x_1, logdet = super().forward(x_0, L, return_traj=False, dirs=dirs)
            q_1, p_1 = torch.split(x_1, x_1.shape[-1] // 2, dim=-1)
            return q_1, p_1, logdet
