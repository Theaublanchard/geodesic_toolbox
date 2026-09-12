"""
Script containing integrators.
We first introduce the general operator for ODE of the sort
    dx/dt = f(x)
Then we focus on the case of Hamiltonian dynamics, where
    f(x) = J * grad H(x)
with J the symplectic matrix and H the Hamiltonian function.
"""

import torch
from torch import Tensor
from tqdm import tqdm
from typing import Callable

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


class EulerIntegrator(Integrator):
    """
    Euler integrator for Riemannian Hamiltonian Monte Carlo.

    Parameters:
    ----------
    f : BatchedFunction
        Fonction that takes a input (B, d) tensor and returns a (B,) tensor.
    gamma : float
        The step size for the Euler integrator.
    """

    def __init__(self, f: BatchedFunction, gamma: float, substeps: int = 1):
        super().__init__(f)
        self.f = f
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

        for k in tqdm(range(L - 1), desc="Euler integration", unit="steps", leave=False):
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

        pbar = tqdm(
            range(L - 1), desc="Implicit midpoint integration", unit="steps", leave=False
        )
        for k in pbar:
            for _ in range(self.substeps):
                x1 = self.implicit_midpoint_step(x0, gamma)
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


###############################
# Integrators for Hamiltonian dynamics dx/dt = J * grad H(x)
# Which we write using the position q and momentum p as
# dq/dt = dH/dp, dp/dt = -dH/dq
###############################
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
    """

    def __init__(self, H: Hamiltonian, gamma: float, substeps: int = 1):
        super().__init__(HamiltonianToBatchedFunction(H), gamma, substeps)
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
    """

    def __init__(self, H: Hamiltonian, gamma: float, substeps: int = 1):
        super().__init__(H)
        self.gamma = gamma / substeps
        self.substeps = substeps

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

        for k in tqdm(range(L - 1), desc="Leapfrog integration", unit="steps", leave=False):
            q_1, p_1 = self.leapfrog_step(q_1, p_1, gamma)

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
    """

    def __init__(self, H: Hamiltonian, gamma: float, n_fix_pts: int, substeps: int = 1):
        super().__init__(H)
        self.n_fix_pts = n_fix_pts
        self.substeps = substeps
        self.gamma = gamma / self.substeps

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

        for k in tqdm(range(L - 1), desc="Leapfrog integration", unit="steps", leave=False):
            q_1, p_1 = self.leapfrog_step(q_1, p_1, gamma)

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
    """

    def __init__(self, H: Hamiltonian, gamma: float, omega: float, substeps: int = 1):
        super().__init__(H)
        self.substeps = substeps
        self.gamma = gamma / self.substeps
        self.omega = omega

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
        self, q_0: Tensor, p_0: Tensor, q_1: Tensor, p_1: Tensor
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
        c = self.c.to(q_0.device).to(q_0.dtype)
        s = self.s.to(q_0.device).to(q_0.dtype)

        p_0_new = p_0 - self.gamma / 2 * self.H.dH_dq(q_0, p_1)
        q_1_new = q_1 + self.gamma / 2 * self.H.dH_dp(q_0, p_1)
        p_1_new = p_1 - self.gamma / 2 * self.H.dH_dq(q_1_new, p_0)
        q_0_new = q_0 + self.gamma / 2 * self.H.dH_dp(q_1_new, p_0)

        # Apply the binding map simultaneously from the same pre-rotation state.
        q0_pre, p0_pre = q_0_new, p_0_new
        q1_pre, p1_pre = q_1_new, p_1_new

        q_0_new = (q0_pre + q1_pre + c * (q0_pre - q1_pre) + s * (p0_pre - p1_pre)) / 2
        p_0_new = (p0_pre + p1_pre - s * (q0_pre - q1_pre) + c * (p0_pre - p1_pre)) / 2
        q_1_new = (q0_pre + q1_pre - c * (q0_pre - q1_pre) - s * (p0_pre - p1_pre)) / 2
        p_1_new = (p0_pre + p1_pre + s * (q0_pre - q1_pre) - c * (p0_pre - p1_pre)) / 2

        p_1_new = p_1_new - self.gamma / 2 * self.H.dH_dq(q_1_new, p_0_new)
        q_0_new = q_0_new + self.gamma / 2 * self.H.dH_dp(q_1_new, p_0_new)
        p_0_new = p_0_new - self.gamma / 2 * self.H.dH_dq(q_0_new, p_1_new)
        q_1_new = q_1_new + self.gamma / 2 * self.H.dH_dp(q_0_new, p_1_new)

        return q_0_new, p_0_new, q_1_new, p_1_new

    @torch.no_grad()
    def forward(
        self,
        q_0: Tensor,
        p_0: Tensor,
        L: int,
        return_traj: bool = False,
    ):
        """Integrate identical initial copies and return the first copy.

        The returned `log_det` is the Jacobian of the full two-copy
        augmented map used by `forward_augmented`, which is zero. It is
        not the Jacobian of the projected first-copy map alone.
        """
        result = self.forward_augmented(q_0, p_0, L, return_traj=return_traj, q_1=q_0, p_1=p_0)
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

        is_nan: bool = False
        for k in tqdm(range(L - 1), desc="Leapfrog steps", unit="steps", leave=False):
            for _ in range(self.substeps):
                q_0, p_0, q_1, p_1 = self.leapfrog_step(q_0, p_0, q_1, p_1)

                if (
                    q_0.isnan().any()
                    or p_0.isnan().any()
                    or q_1.isnan().any()
                    or p_1.isnan().any()
                ):
                    # raise ValueError("NaN values encountered in leapfrog step.")
                    print("NaN values encountered in leapfrog step.")
                    is_nan = True
                    break

            if is_nan:
                ...
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
        )

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
