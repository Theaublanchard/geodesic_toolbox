"""
Riemannian HMC samplers, with optional reduced-flip direction persistence.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from tqdm import tqdm

from ..cometric import CoMetric, mat_sqrt
from ..integrators import (
    ExplicitLeapfrogIntegrator,
    Hamiltonian,
    HamiltonianImplicitMidpointIntegrator,
    ImplicitLeapfrogIntegrator,
)
from .base import Sampler, integrate_hamiltonian_isolating_failures
from .hamiltonian import LogTargetRiemannHamiltonian, VolumeRiemannHamiltonian


class _RHMCSamplerBase(Sampler):
    """Shared implementation for the public RHMC samplers.

    Parameters
    ----------
    cometric : CoMetric
        Cometric defining the position-dependent kinetic energy and momentum
        distribution.
    num_integration_steps : int
        Number of integrator steps used for each proposal.
    fixed_point_iterations : int
        Number of fixed-point iterations used by implicit integrators.
    step_size : float
        Integrator step size.
    log_target : Callable[[Tensor], Tensor] | None
        Unnormalized log-density. If omitted, uses the cometric volume density.
    momentum_std : float
        Scale factor for the metric Gaussian momentum.
    bounds : float | None
        Optional radial support bound. Proposals outside it are rejected.
    pbar : bool
        If True, display a progress bar while collecting samples.
    H : Hamiltonian | None
        Optional compatible Hamiltonian override.
    compile_step : bool
        If True, compile the integrator step with ``torch.compile``.
    """

    def __init__(
        self,
        cometric: CoMetric,
        num_integration_steps: int,
        fixed_point_iterations: int,
        step_size: float,
        log_target: Callable[[Tensor], Tensor] | None = None,
        momentum_std: float = 1.0,
        bounds: float | None = None,
        pbar: bool = False,
        H: Hamiltonian | None = None,
        compile_step: bool = True,
    ):
        super().__init__(pbar)
        if num_integration_steps < 2:
            raise ValueError("num_integration_steps must be at least 2.")
        if fixed_point_iterations < 1:
            raise ValueError("fixed_point_iterations must be positive.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if momentum_std <= 0:
            raise ValueError("momentum_std must be positive.")
        self.cometric = cometric
        self.num_integration_steps = num_integration_steps
        self.momentum_std = momentum_std
        self.bounds = bounds
        if H is not None:
            self.H = H
        elif log_target is None:
            self.H = VolumeRiemannHamiltonian(cometric)
        else:
            self.H = LogTargetRiemannHamiltonian(log_target, cometric)
        self.integrator = None

    def sample_momentum(self, state: Tensor) -> Tensor:
        """Draw a metric Gaussian momentum for each chain.

        Parameters
        ----------
        state : Tensor (num_chains, dimension)
            Current position of each chain. The metric at these positions
            determines the momentum covariance.

        Returns
        -------
        Tensor (num_chains, dimension)
            Independently sampled momenta with covariance proportional to the
            metric tensor at each position.
        """
        metric = self.cometric.metric_tensor(state)
        momentum = torch.randn_like(state)
        if self.cometric.is_diag:
            momentum = momentum * metric.sqrt()
        else:
            momentum = torch.einsum("bij,bi->bj", mat_sqrt(metric), momentum)
        return momentum * self.momentum_std

    @torch.no_grad()
    def step(self, state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Perform one RHMC transition using the subclass integrator.

        Parameters
        ----------
        state : Tensor (num_chains, dimension)
            Current position of each chain.

        Returns
        -------
        next_state : Tensor (num_chains, dimension)
            Position after the Metropolis transition.
        diagnostics : dict[str, Tensor]
            Per-chain acceptance indicators and probabilities.
        """
        if state.ndim != 2:
            raise ValueError("state must have shape (num_chains, dimension).")

        momentum = self.sample_momentum(state)
        proposal_q, proposal_p, log_det, valid = integrate_hamiltonian_isolating_failures(
            self.integrator,
            state,
            momentum,
            self.num_integration_steps,
        )
        log_alpha = self.H(state, momentum) - self.H(proposal_q, proposal_p) + log_det

        acceptance_probability = torch.exp(torch.clamp(log_alpha, max=0.0))
        acceptance_probability = torch.nan_to_num(acceptance_probability, nan=0.0)

        if self.bounds is not None:
            acceptance_probability = torch.where(
                torch.linalg.norm(proposal_q, dim=-1) <= self.bounds,
                acceptance_probability,
                torch.zeros_like(acceptance_probability),
            )
        acceptance_probability = torch.where(
            valid, acceptance_probability, torch.zeros_like(acceptance_probability)
        )

        accepted = torch.rand_like(acceptance_probability) < acceptance_probability
        next_state = torch.where(accepted[:, None], proposal_q, state)

        return next_state, {
            "accepted": accepted,
            "acceptance_probability": acceptance_probability,
            "valid": valid,
        }

    @torch.no_grad()
    def sample(
        self, initial_state: Tensor, num_warmup: int, num_samples: int
    ) -> tuple[Tensor, dict[str, object]]:
        """Run warmup and collect post-warmup RHMC states.

        Parameters
        ----------
        initial_state : Tensor (num_chains, dimension)
            Initial position of each chain.
        num_warmup : int
            Number of transitions discarded before collection.
        num_samples : int
            Number of post-warmup transitions to retain.

        Returns
        -------
        samples : Tensor (num_chains, num_samples, dimension)
            Collected chain states.
        diagnostics : dict[str, object]
            Acceptance, validity, and warmup diagnostics.
        """
        if initial_state.ndim != 2:
            raise ValueError("initial_state must have shape (num_chains, dimension).")
        if num_warmup < 0:
            raise ValueError("num_warmup must be non-negative.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        state = initial_state.clone()

        warmup_accepted = 0
        iter_warmup = (
            tqdm(range(num_warmup), desc="Warmup", unit="draws")
            if self.pbar
            else range(num_warmup)
        )
        for i in iter_warmup:
            state, transition = self.step(state)
            warmup_accepted += transition["accepted"].sum().item()

            if self.pbar:
                iter_warmup.set_postfix(
                    {"warmup_acceptance_rate": warmup_accepted / ((1 + i) * state.shape[0])}
                )

        samples, accepted, probabilities, valid = [], [], [], []
        flipped = []
        iterator = (
            tqdm(range(num_samples), desc="Sampling", unit="draws")
            if self.pbar
            else range(num_samples)
        )
        for i in iterator:
            state, transition = self.step(state)

            samples.append(state.clone())
            accepted.append(transition["accepted"])
            probabilities.append(transition["acceptance_probability"])
            valid.append(transition["valid"])

            if "flipped" in transition:
                flipped.append(transition["flipped"])

            if self.pbar:
                iterator.set_postfix(
                    {"acceptance_rate": torch.stack(accepted, dim=1).float().mean().item()}
                )

        accepted_tensor = torch.stack(accepted, dim=1)
        diagnostics: dict[str, object] = {
            "acceptance_rate": accepted_tensor.float().mean().item(),
            "acceptance_rate_by_chain": accepted_tensor.float().mean(dim=1),
            "acceptance_probability": torch.stack(probabilities, dim=1),
            "valid": torch.stack(valid, dim=1),
            "warmup_acceptance_rate": (
                warmup_accepted / (num_warmup * state.shape[0]) if num_warmup else None
            ),
        }

        if flipped:
            diagnostics["flipped"] = torch.stack(flipped, dim=1)

        return torch.stack(samples, dim=1), diagnostics


class ImplicitMidpointRHMCSampler(_RHMCSamplerBase):
    """Riemannian HMC using the implicit midpoint integrator.

    The midpoint integrator is symmetric and volume-preserving for the
    canonical Hamiltonian dynamics.

    ``compile_step`` is forwarded to the integrator and can be used to
    compile its step function with ``torch.compile``.
    """

    def __init__(
        self,
        cometric: CoMetric,
        num_integration_steps: int,
        fixed_point_iterations: int,
        step_size: float,
        log_target: Callable[[Tensor], Tensor] | None = None,
        momentum_std: float = 1.0,
        bounds: float | None = None,
        pbar: bool = False,
        H: Hamiltonian | None = None,
        compile_step: bool = True,
    ):
        super().__init__(
            cometric,
            num_integration_steps,
            fixed_point_iterations,
            step_size,
            log_target,
            momentum_std,
            bounds,
            pbar,
            H,
            compile_step,
        )
        self.integrator = HamiltonianImplicitMidpointIntegrator(
            self.H, step_size, fixed_point_iterations, compile_step=compile_step
        )


class ImplicitLeapfrogRHMCSampler(_RHMCSamplerBase):
    """Riemannian HMC using the implicit leapfrog integrator.

    The position-dependent kinetic energy is handled through implicit
    fixed-point updates.

    ``compile_step`` is forwarded to the integrator and can be used to
    compile its step function with ``torch.compile``.
    """

    def __init__(
        self,
        cometric: CoMetric,
        num_integration_steps: int,
        fixed_point_iterations: int,
        step_size: float,
        log_target: Callable[[Tensor], Tensor] | None = None,
        momentum_std: float = 1.0,
        bounds: float | None = None,
        pbar: bool = False,
        H: Hamiltonian | None = None,
        compile_step: bool = True,
    ):
        super().__init__(
            cometric,
            num_integration_steps,
            fixed_point_iterations,
            step_size,
            log_target,
            momentum_std,
            bounds,
            pbar,
            H,
            compile_step,
        )
        self.integrator = ImplicitLeapfrogIntegrator(
            self.H, step_size, fixed_point_iterations, compile_step=compile_step
        )


class ExplicitRHMCSampler(_RHMCSamplerBase):
    """RHMC using the explicit augmented leapfrog integrator.

    The auxiliary copy is initialized from the current chain state for every
    transition and is not carried between transitions.

    ``compile_step`` is forwarded to the integrator and can be used to
    compile its step function with ``torch.compile``.

    Parameters
    ----------
    omega : float
        Binding frequency for the augmented two-copy dynamics.
    """

    def __init__(
        self,
        cometric: CoMetric,
        num_integration_steps: int,
        step_size: float,
        omega: float,
        fixed_point_iterations: int = 1,  # unused but kept for compatibility
        log_target: Callable[[Tensor], Tensor] | None = None,
        momentum_std: float = 1.0,
        bounds: float | None = None,
        pbar: bool = False,
        H: Hamiltonian | None = None,
        compile_step: bool = True,
    ):
        super().__init__(
            cometric,
            num_integration_steps,
            fixed_point_iterations,
            step_size,
            log_target,
            momentum_std,
            bounds,
            pbar,
            H,
            compile_step,
        )
        self.integrator = ExplicitLeapfrogIntegrator(
            self.H, step_size, omega, compile_step=compile_step
        )


class _ReducedFlipRHMCMixin:
    """
    Base class for RHMC samplers with persistent signed directions and reduced-flip transitions.
    Idea from Sohl-Dickstein (2012): when a proposal is rejected, evaluate the proposal in
    the reverse direction and flip the direction for the next transition with
    probability proportional to the difference in acceptance probabilities.
    The reverse-direction acceptance is used in place of the momentum-flipped proposal
    in the paper. Only computed for rejected samples.
    """

    _directions: Tensor | None = None

    @torch.no_grad()
    def step(self, state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Perform one reduced-flip RHMC transition.

        Parameters
        ----------
        state : Tensor (num_chains, dimension)
            Current position of each chain.

        Returns
        -------
        next_state : Tensor (num_chains, dimension)
            Position after the reduced-flip Metropolis transition.
        diagnostics : dict[str, Tensor]
            Per-chain acceptance, reverse-acceptance, validity, and direction
            flip indicators.
        """
        if state.ndim != 2:
            raise ValueError("state must have shape (num_chains, dimension).")
        if self._directions is None or self._directions.shape[0] != state.shape[0]:
            self._directions = torch.ones(
                state.shape[0], device=state.device, dtype=state.dtype
            )
        directions = self._directions.to(device=state.device, dtype=state.dtype)

        momentum = self.sample_momentum(state)
        initial = torch.cat([state, momentum], dim=-1)

        proposal_q, proposal_p, log_det, valid = integrate_hamiltonian_isolating_failures(
            self.integrator,
            state,
            momentum,
            self.num_integration_steps,
            directions,
        )
        proposal = torch.cat([proposal_q, proposal_p], dim=-1)
        alpha = self._rf_alpha(initial, proposal, log_det, valid, proposal_q)
        uniform = torch.rand_like(alpha)
        accepted = uniform < alpha

        flipped = torch.zeros_like(accepted)
        reverse_alpha = torch.zeros_like(alpha)
        rejected_indices = (~accepted).nonzero(as_tuple=False).squeeze(-1)
        if rejected_indices.numel() > 0:
            reverse_q, reverse_p, reverse_log_det, reverse_valid = (
                integrate_hamiltonian_isolating_failures(
                    self.integrator,
                    state[rejected_indices],
                    momentum[rejected_indices],
                    self.num_integration_steps,
                    -directions[rejected_indices],
                )
            )
            reverse_proposal = torch.cat([reverse_q, reverse_p], dim=-1)
            reverse_alpha[rejected_indices] = self._rf_alpha(
                initial[rejected_indices],
                reverse_proposal,
                reverse_log_det,
                reverse_valid,
                reverse_q,
            )
            flip_probability = (
                reverse_alpha[rejected_indices] - alpha[rejected_indices]
            ).clamp(min=0)
            flipped[rejected_indices] = uniform[rejected_indices] < (
                alpha[rejected_indices] + flip_probability
            )

        self._directions = torch.where(flipped, -directions, directions)

        return torch.where(accepted[:, None], proposal_q, state), {
            "accepted": accepted,
            "acceptance_probability": alpha,
            "reverse_acceptance_probability": reverse_alpha,
            "flipped": flipped,
            "valid": valid,
        }

    def _rf_alpha(
        self,
        initial: Tensor,
        proposal: Tensor,
        log_det: Tensor,
        valid: Tensor,
        proposal_q: Tensor,
    ) -> Tensor:
        """Compute reduced-flip acceptance probabilities.

        Parameters
        ----------
        initial : Tensor (num_chains, 2 * dimension)
            Concatenated initial positions and momenta.
        proposal : Tensor (num_chains, 2 * dimension)
            Concatenated proposed positions and momenta.
        log_det : Tensor (num_chains,)
            Log-Jacobians of the proposal transformation.
        valid : Tensor (num_chains,) bool
            Mask identifying proposals produced by valid integrations.
        proposal_q : Tensor (num_chains, dimension)
            Proposed positions, used for support-bound checks.

        Returns
        -------
        Tensor (num_chains,)
            Per-chain acceptance probabilities in the interval ``[0, 1]``.
        """
        dimension = proposal_q.shape[1]
        log_alpha = (
            self.H(initial[:, :dimension], initial[:, dimension:])
            - self.H(proposal[:, :dimension], proposal[:, dimension:])
            + log_det
        )
        alpha = torch.nan_to_num(torch.exp(torch.clamp(log_alpha, max=0.0)), nan=0.0)

        if self.bounds is not None:
            alpha = torch.where(
                torch.linalg.norm(proposal_q, dim=-1) <= self.bounds,
                alpha,
                torch.zeros_like(alpha),
            )

        return torch.where(valid, alpha, torch.zeros_like(alpha))

    @torch.no_grad()
    def sample(
        self, initial_state: Tensor, num_warmup: int, num_samples: int
    ) -> tuple[Tensor, dict[str, object]]:
        """Run warmup and collect reduced-flip RHMC states.

        Parameters
        ----------
        initial_state : Tensor (num_chains, dimension)
            Initial position of each chain.
        num_warmup : int
            Number of transitions discarded before collection.
        num_samples : int
            Number of post-warmup transitions to retain.

        Returns
        -------
        samples : Tensor (num_chains, num_samples, dimension)
            Collected chain states.
        diagnostics : dict[str, object]
            Standard RHMC diagnostics plus aggregate and per-chain direction
            flip rates.
        """
        self._directions = None
        samples, diagnostics = super().sample(initial_state, num_warmup, num_samples)
        flipped = diagnostics["flipped"]
        assert isinstance(flipped, Tensor)
        diagnostics["flip_rate"] = flipped.float().mean().item()
        diagnostics["flip_rate_by_chain"] = flipped.float().mean(dim=1)
        return samples, diagnostics


class ImplicitLeapfrogRHMCRFSampler(_ReducedFlipRHMCMixin, ImplicitLeapfrogRHMCSampler):
    """Implicit-leapfrog RHMC with reduced-flip direction persistence."""


class ImplicitMidpointRHMCRFSampler(_ReducedFlipRHMCMixin, ImplicitMidpointRHMCSampler):
    """Implicit-midpoint RHMC with reduced-flip direction persistence."""


class ExplicitRHMCRFSampler(_ReducedFlipRHMCMixin, ExplicitRHMCSampler):
    """Explicit augmented RHMC with reduced-flip direction persistence."""
