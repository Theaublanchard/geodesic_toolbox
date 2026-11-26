"""
Metropolis-adjusted Langevin algorithm (MALA) sampler.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from tqdm import tqdm

from .base import Sampler


class MALA(Sampler):
    """
    (Preconditioned) Metropolis-adjusted Langevin algorithm.

    With preconditioner ``A`` (symmetric positive definite) and step size ``eps``, the proposal is

        x' = x + eps * A @ score(x) + sqrt(2 * eps) * A^{1/2} @ z,    z ~ N(0, I),

    i.e. ``q(x' | x) = N(x + eps * A @ score(x), 2 * eps * A)``, and it is accepted with probability
    ``min(1, pi(x') q(x | x') / (pi(x) q(x' | x)))``. With ``A = I`` this is standard MALA.
    The chain leaves ``exp(log_target)`` invariant for any valid ``A`` and ``eps``.

    Parameters
    ----------
    log_target : Callable[[Tensor], Tensor]
        Unnormalized log-density, mapping (b, d) positions to (b,) log-densities.
    score_func : Callable[[Tensor], Tensor]
        Gradient of ``log_target``, mapping (b, d) positions to (b, d) gradients.
    step_size : float
        Step size ``eps`` of the Langevin proposal (must be positive).
    preconditioner : Tensor (d, d) | None
        Symmetric positive definite matrix ``A`` (the covariance shape of the proposal noise, i.e. the
        inverse of the "mass matrix" in the HMC sense). ``None`` means identity; no dimension is needed.
    pbar : bool
        If True, display a progress bar while collecting samples.
    bounds : float | None
        If given, proposals with Euclidean norm above ``bounds`` are rejected. This samples the target
        truncated to the ball of that radius.
    """

    def __init__(
        self,
        log_target: Callable[[Tensor], Tensor],
        score_func: Callable[[Tensor], Tensor],
        step_size: float,
        preconditioner: Tensor | None = None,
        pbar: bool = False,
        bounds: float | None = None,
    ):
        super().__init__(pbar)
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if bounds is not None and bounds <= 0:
            raise ValueError("bounds must be positive.")
        self.log_target = log_target
        self.score_func = score_func
        self.step_size = float(step_size)
        self.bounds = bounds

        # Cache factorizations once, in float64 for accuracy; cast to the state's dtype/device per call.
        self.A: Tensor | None = None
        self.A_chol: Tensor | None = None  # lower Cholesky factor L, L @ L.T = A
        self.A_inv: Tensor | None = None
        if preconditioner is not None:
            A = preconditioner.detach().to(torch.float64)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError("preconditioner must be a square (d, d) matrix.")
            if not torch.allclose(A, A.T, rtol=1e-8, atol=1e-10):
                raise ValueError("preconditioner must be symmetric.")
            A = 0.5 * (A + A.T)
            # Cholesky raises if A is not positive definite.
            self.A = A
            self.A_chol = torch.linalg.cholesky(A)
            self.A_inv = torch.cholesky_inverse(self.A_chol)

    def _factors(self, state: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if self.A is None:
            return None, None, None
        if self.A.shape[0] != state.shape[1]:
            raise ValueError(
                f"preconditioner has dimension {self.A.shape[0]} but state has dimension {state.shape[1]}."
            )
        return self.A.to(state), self.A_chol.to(state), self.A_inv.to(state)

    def _log_q(
        self, x: Tensor, x_new: Tensor, grad_x: Tensor, A: Tensor | None, A_inv: Tensor | None
    ) -> Tensor:
        """log q(x_new | x) up to an additive constant that cancels in the MH ratio."""
        drift = grad_x if A is None else grad_x @ A  # A is symmetric
        diff = x_new - x - self.step_size * drift
        quad = (diff**2).sum(-1) if A_inv is None else ((diff @ A_inv) * diff).sum(-1)
        return -quad / (4.0 * self.step_size)

    def _transition(
        self, state: Tensor, log_p: Tensor, grad: Tensor, factors
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One MH transition given the cached log-density and score at ``state``."""
        A, A_chol, A_inv = factors
        eps = self.step_size

        z = torch.randn_like(state)
        noise = z if A_chol is None else z @ A_chol.T  # covariance A
        drift = grad if A is None else grad @ A
        proposal = state + eps * drift + (2.0 * eps) ** 0.5 * noise

        log_p_prop = self.log_target(proposal)
        grad_prop = self.score_func(proposal)

        log_alpha = (
            log_p_prop
            - log_p
            + self._log_q(proposal, state, grad_prop, A, A_inv)
            - self._log_q(state, proposal, grad, A, A_inv)
        )
        # Non-finite proposals (nan/inf log-density or score) are rejected.
        valid = torch.isfinite(log_p_prop) & torch.isfinite(grad_prop).all(dim=-1)
        if self.bounds is not None:
            valid &= torch.linalg.norm(proposal, dim=-1) <= self.bounds
        log_alpha = torch.where(valid & ~torch.isnan(log_alpha), log_alpha, -torch.inf)

        acceptance_probability = torch.exp(torch.clamp(log_alpha, max=0.0))
        accepted = torch.log(torch.rand_like(log_alpha)) < log_alpha

        next_state = torch.where(accepted[:, None], proposal, state)
        next_log_p = torch.where(accepted, log_p_prop, log_p)
        next_grad = torch.where(accepted[:, None], grad_prop, grad)
        return next_state, next_log_p, next_grad, accepted, acceptance_probability

    def step(self, state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Perform a single MALA step.

        Parameters
        ----------
        state : Tensor (num_chains, dimension)
            Current position of each chain.

        Returns
        -------
        next_state : Tensor (num_chains, dimension)
            Next position of each chain.
        diagnostics : dict[str, Tensor]
            ``accepted`` (num_chains,) booleans and ``acceptance_probability`` (num_chains,).
        """
        if state.ndim != 2:
            raise ValueError("state must have shape (num_chains, dimension).")
        with torch.no_grad():
            factors = self._factors(state)
            log_p, grad = self.log_target(state), self.score_func(state)
            next_state, _, _, accepted, prob = self._transition(state, log_p, grad, factors)
        return next_state, {"accepted": accepted, "acceptance_probability": prob}

    def sample(
        self, initial_state: Tensor, num_warmup: int, num_samples: int
    ) -> tuple[Tensor, dict[str, object]]:
        """Run warmup and collect post-warmup states.

        Parameters
        ----------
        initial_state : Tensor (num_chains, dimension)
            Initial position of each chain (must have finite log-density).
        num_warmup : int
            Number of transitions to discard before collecting samples.
        num_samples : int
            Number of post-warmup transitions to retain.

        Returns
        -------
        samples : Tensor (num_chains, num_samples, dimension)
            Collected chain states. Rejected transitions repeat the previous state.
        diagnostics : dict[str, object]
            ``acceptance_rate``, ``acceptance_rate_by_chain``, ``acceptance_probability``
            (num_chains, num_samples) and ``warmup_acceptance_rate``.
        """
        if initial_state.ndim != 2:
            raise ValueError("initial_state must have shape (num_chains, dimension).")
        if num_warmup < 0:
            raise ValueError("num_warmup must be non-negative.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        with torch.no_grad():
            state = initial_state.clone()
            factors = self._factors(state)
            # log-density and score at the current state are carried across steps (no recomputation).
            log_p, grad = self.log_target(state), self.score_func(state)
            if not torch.isfinite(log_p).all():
                raise ValueError("initial_state must have finite log-density for every chain.")

            warmup_accepted = 0
            for _ in range(num_warmup):
                state, log_p, grad, acc, _ = self._transition(state, log_p, grad, factors)
                warmup_accepted += acc.sum().item()

            samples, accepted, probabilities = [], [], []
            iterator = (
                tqdm(range(num_samples), desc="Sampling", unit="draws")
                if self.pbar
                else range(num_samples)
            )
            for _ in iterator:
                state, log_p, grad, acc, prob = self._transition(state, log_p, grad, factors)
                samples.append(state.clone())
                accepted.append(acc)
                probabilities.append(prob)

        accepted_tensor = torch.stack(accepted, dim=1)
        diagnostics: dict[str, object] = {
            "acceptance_rate": accepted_tensor.float().mean().item(),
            "acceptance_rate_by_chain": accepted_tensor.float().mean(dim=1),
            "acceptance_probability": torch.stack(probabilities, dim=1),
            "warmup_acceptance_rate": (
                warmup_accepted / (num_warmup * state.shape[0]) if num_warmup else None
            ),
        }
        return torch.stack(samples, dim=1), diagnostics
