"""
Euler integrators, both for general vector fields and for Hamiltonian dynamics.
"""

from collections.abc import Callable

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


class ImplicitEulerIntegrator(torch.nn.Module):
    """
    Backward (implicit) Euler integrator x1 = x0 + gamma * f(x1) for an
    arbitrary vector field f. The step is A-stable (linearly stable for
    stiff linear fields) and the exact log-Jacobian of each discrete step,

        Delta log J = log|det(I - gamma D)| - log|det(I)|,
        D = df(x1),

    is accumulated so it can be included in a Metropolis ratio.

    The implicit equation is solved with either Picard fixed-point iteration
    (``method="picard"``) or Newton's method (``method="newton"``).

    Parameters
    ----------
    f : BatchedFunction
        Vector field, maps (b, n) to (b, n).
    df : Callable
        Jacobian of the field, maps (b, n) to (b, n, n). Only required when
        ``method="newton"``.
    gamma : float
        Step size.
    N_fx : int
        Number of fixed-point (picard) or Newton iterations per step.
    method : str
        "picard" or "newton".
    jacobian : str
        "exact" or "estimate".
    substeps : int
        Number of substeps for the implicit Euler step.
        This is the number of times the implicit Euler step is applied
        to the same pair of states (x0, x1) before updating the states.
    # The following parameters are only used if jacobian="estimate":
    jacobian_mc : int
        Number of Monte Carlo probes for estimating the log-Jacobian using Hutchinson's estimator.
    russian_roulette : float
        Probability of terminating the series for estimating the log-Jacobian.
    compile_step : bool
        If True, compile the step function with torch.compile.
    pbar : bool
        If True, display a progress bar during integration. Defaults to False.
    """

    def __init__(
        self,
        f: BatchedFunction,
        gamma: float,
        N_fx: int,
        df: Callable[[Tensor], Tensor] | None = None,
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

        if method == "picard":
            self.implicit_euler_step = self.picard
        elif method == "newton":
            if df is None:
                raise ValueError("The 'newton' method requires the field Jacobian 'df'.")
            self.implicit_euler_step = self.newton
        else:
            raise ValueError(f"Unknown method {method}. Choose 'picard' or 'newton'.")

        if jacobian == "estimate":
            if df is None:
                raise ValueError("The 'estimate' jacobian requires the field Jacobian 'df'.")
            self.log_det_jac = self.estimate_log_det_jac
        elif jacobian == "exact":
            if df is None:
                raise ValueError("The 'exact' jacobian requires the field Jacobian 'df'.")
            self.log_det_jac = self.exact_log_det_jac
        else:
            raise ValueError(
                f"Unknown jacobian method {jacobian}. Choose 'estimate' or 'exact'."
            )

        self.step = self.implicit_euler_step
        if compile_step:
            self.step = torch.compile(self.step, mode="reduce-overhead")

    def picard(self, x0: Tensor, gamma: Tensor, tol: float = 1e-8) -> Tensor:
        """
        Apply Picard fixed-point iteration to solve the implicit Euler equation
        x1 = x0 + gamma * f(x1).
        Iteration stops when the maximum absolute change is below tol or after
        self.N_fx iterations.

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
            x1_ = x0 + gamma * self.f(x1)
            delta = (x1_ - x1).abs().max()
            x1 = x1_
            if delta < tol:
                break
        return x1

    def newton(self, x0: Tensor, gamma: Tensor) -> Tensor:
        """
        Apply Newton's method to solve the implicit Euler equation
        x1 = x0 + gamma * f(x1). The Jacobian is evaluated at the current
        iterate x1. Iteration stops after self.N_fx iterations.

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
            D = self.df(x1)
            residual = x1 - x0 - gamma * self.f(x1)
            J = I - gamma.unsqueeze(-1) * D
            x1 = x1 - torch.linalg.solve(J, residual)
        return x1

    def exact_log_det_jac(self, x1: Tensor, gamma: Tensor) -> Tensor:
        """
        Compute the exact log-Jacobian of the implicit Euler step.
        log|det(I - gamma D)|, where D = df(x1). (det(I) = 1, so its log is 0.)

        Parameters
        ----------
        x1 : Tensor (b, n)
            The next state.
        gamma : Tensor (b, 1)
            The step size for each batch.

        Returns
        -------
        delta : Tensor (b,)
            The log-Jacobian of the implicit Euler step.
        """
        D = self.df(x1)
        I = torch.eye(D.shape[-1], device=D.device, dtype=D.dtype)
        gamma_D = gamma.unsqueeze(-1) * D
        _, logabsdet = torch.linalg.slogdet(I - gamma_D)
        return logabsdet

    # @TODO: optimize this code
    def estimate_log_det_jac(self, x1: Tensor, gamma: Tensor) -> Tensor:
        """
        Unbiased matrix-free estimate of the step log-Jacobian

            Delta log J = - sum_{k>=1} tr(A^k) / k,  A = gamma df(x1),

        Traces use Hutchinson's estimator with self.jacobian_mc Rademacher
        probes and JVPs of f (the Jacobian is never formed); the series is
        truncated by russian roulette with a Geometric(self.russian_roulette)
        number of terms, each reweighted by its survival probability.
        Converges for spectral radius of A below 1 (small gamma).

        Parameters
        ----------
        x1 : Tensor (b, n)
            The next state.
        gamma : Tensor (b, 1)
            The step size for each batch.

        Returns
        -------
        delta : Tensor (b,)
            The estimated log-Jacobian of the implicit Euler step.
        """
        b = x1.shape[0]
        # Number of terms, shared across the batch.
        N = int(torch.empty(1).geometric_(self.russian_roulette).item())

        jvp_fn = lambda w: torch.func.jvp(self.f, (x1,), (w,))[1]
        q = 1.0 - self.russian_roulette
        delta = torch.zeros(b, device=x1.device, dtype=x1.dtype)
        for _ in range(self.jacobian_mc):
            # Rademacher probe (b, n).
            v = (
                torch.randint(0, 2, x1.shape, device=x1.device)
                .to(dtype=x1.dtype)
                .mul_(2)
                .sub_(1)
            )
            w = gamma * jvp_fn(v)  # A^1 v
            for j in range(N):
                k = j + 1
                if j > 0:
                    # Advance from A^{k-1} v to A^k v with one JVP.
                    w = gamma * jvp_fn(w)
                # Hutchinson estimate of tr(A^k).
                trace_k = (v * w).sum(dim=-1)
                delta = delta - trace_k / (k * q**j)
        return delta / self.jacobian_mc

    def forward(
        self, x0: Tensor, L: int, return_traj: bool = False, dirs: Tensor | None = None
    ):
        """
        Integrate the vector field f using the implicit (backward) Euler method
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
            steps = tqdm(steps, desc="Implicit Euler integration", unit="steps", leave=False)
        for k in steps:
            for _ in range(self.substeps):
                x1 = self.step(x0, gamma)
                delta = self.log_det_jac(x1, gamma)
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


class HamiltonianImplicitEulerIntegrator(ImplicitEulerIntegrator, HamiltonianIntegrator):
    """
    Backward (implicit) Euler integrator for Riemannian Hamiltonian Monte Carlo.

    Parameters
    ----------
    H : Hamiltonian
        The Hamiltonian function H(q, p) that takes position q and momentum p and returns the energy.
    gamma : float
        The step size for the implicit Euler integrator.
    N_fx : int
        The number of fixed point iterations to perform for the implicit equations.
    method : str
        The method to use for the implicit Euler integrator. Can be "picard" or "newton".
    jacobian : str
        The method to use for computing the log-Jacobian. Can be "exact" or "estimate".
    substeps : int
        The number of substeps for the implicit Euler integrator.
        This is the number of times the implicit Euler step is applied
        to the same pair of states (q_0, p_0) and (q_1, p_1) before updating the states.
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
        f = HamiltonianToBatchedFunction(H)
        super().__init__(
            f,
            gamma,
            N_fx,
            df=f.df,
            method=method,
            jacobian=jacobian,
            substeps=substeps,
            jacobian_mc=jacobian_mc,
            russian_roulette=russian_roulette,
            compile_step=compile_step,
            pbar=pbar,
        )

        self.H = H

    def forward(
        self,
        q_0: Tensor,
        p_0: Tensor,
        L: int,
        return_traj: bool = False,
        dirs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform L-1 implicit Euler steps starting from (q_0, p_0).

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        L : int
            The number of implicit Euler steps to perform.
        return_traj : bool
            If True, it returns the trajectory of the samples over the L implicit Euler steps.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If None, all samples are integrated forward.

        Returns
        -------
        q_L : Tensor (b,d)
            The new position after L implicit Euler steps.
        p_L : Tensor (b,d)
            The new momentum after L implicit Euler steps.
        or
        (Tensor (b,L,d), Tensor (b,L,d))
            The trajectory of the positions and momenta over the L implicit Euler steps.
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
