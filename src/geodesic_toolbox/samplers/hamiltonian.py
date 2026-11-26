"""
Euclidean HMC sampler and Riemannian Hamiltonians for HMC samplers.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from tqdm import tqdm

from ..cometric import CoMetric
from ..integrators import Hamiltonian, SeparableLeapfrogIntegrator
from .base import Sampler


class EuclideanHamiltonian(Hamiltonian):
    """Separable Euclidean Hamiltonian for an unnormalized log-density.

    Parameters
    ----------
    log_target : Callable[[Tensor], Tensor]
        Unnormalized log-density mapping ``(num_chains, dimension)`` positions
        to ``(num_chains,)`` values.
    momentum_std : float
        Standard deviation of the independent Gaussian momentum.
    """

    def __init__(self, log_target: Callable[[Tensor], Tensor], momentum_std: float = 1.0):
        super().__init__()
        if momentum_std <= 0:
            raise ValueError("momentum_std must be positive.")
        self.log_target = log_target
        self.momentum_std = momentum_std

    def U(self, q: Tensor) -> Tensor:
        """Return the potential energy ``-log_target(q)``.

        Parameters
        ----------
        q : Tensor (b, d)
            Batch of positions.

        Returns
        -------
        Tensor (b,)
            Potential energy of each position.
        """
        return -self.log_target(q)

    def K(self, p: Tensor) -> Tensor:
        """Return the kinetic energy of the Gaussian momentum.

        Parameters
        ----------
        p : Tensor (b, d)
            Batch of momenta.

        Returns
        -------
        Tensor (b,)
            Kinetic energy of each momentum.
        """
        return 0.5 * torch.einsum("bi,bi->b", p, p) / self.momentum_std**2

    def forward(self, q: Tensor, p: Tensor) -> Tensor:
        """Evaluate the Hamiltonian for batched positions and momenta.

        Parameters
        ----------
        q : Tensor (b, d)
            Batch of positions.
        p : Tensor (b, d)
            Batch of momenta.

        Returns
        -------
        Tensor (b,)
            Hamiltonian for each position and momentum pair.
        """
        return self.U(q) + self.K(p)


class HamiltonianMonteCarlo(Sampler):
    """Euclidean Hamiltonian Monte Carlo for an unnormalized log-density.

    Parameters are algorithmic settings; the number of warmup and retained
    transitions is supplied to :meth:`sample`.

    Parameters
    ----------
    log_target : Callable[[Tensor], Tensor]
        Unnormalized log-density, maps (b, d) positions to (b,) log-densities.
    num_integration_steps : int
        Number of leapfrog steps per proposal.
    step_size : float
        Leapfrog step size.
    momentum_std : float
        Standard deviation of the Euclidean momentum distribution.
    bounds : float | None
        Optional bounds on the support of the target distribution. If provided,
        proposals outside the bounds are rejected directly.
    pbar : bool
        If True, it shows a progress bar when sampling.
    compile_step : bool
        If True, compile the leapfrog integrator step with ``torch.compile``.
    """

    def __init__(
        self,
        log_target: Callable[[Tensor], Tensor],
        num_integration_steps: int,
        step_size: float,
        momentum_std: float = 1.0,
        bounds: float | None = None,
        pbar: bool = False,
        compile_step: bool = True,
    ):
        super().__init__(pbar)
        if num_integration_steps < 1:
            raise ValueError("num_integration_steps must be positive.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        self.num_integration_steps = num_integration_steps
        self.bounds = bounds
        self.hamiltonian = EuclideanHamiltonian(log_target, momentum_std)
        self.integrator = SeparableLeapfrogIntegrator(
            self.hamiltonian, step_size, compile_step=compile_step
        )

    def sample_momentum(self, state: Tensor) -> Tensor:
        momentum = torch.randn_like(state) * self.hamiltonian.momentum_std
        return momentum

    @torch.no_grad()
    def step(self, state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Perform a single HMC step.

        Parameters
        ----------
        state : Tensor (B, d)
            Current state of the Markov chain

        Returns
        -------
        next_state : Tensor (B, d)
            Next state of the Markov chain
        diagnostics : dict[str, Tensor]
            Diagnostic information about the step.
        """
        if state.ndim != 2:
            raise ValueError("state must have shape (num_chains, dimension).")
        momentum = self.sample_momentum(state)
        # No need to compute the log_det here since the leapfrog integrator is symplectic (log_det = 0)
        proposal, proposal_momentum, _ = self.integrator(
            state,
            momentum,
            self.num_integration_steps + 1,
        )
        log_alpha = self.hamiltonian(state, momentum) - self.hamiltonian(
            proposal, proposal_momentum
        )
        if self.bounds is not None:
            log_alpha = torch.where(
                torch.linalg.norm(proposal, dim=-1) <= self.bounds,
                log_alpha,
                torch.full_like(log_alpha, -torch.inf),
            )
        acceptance_probability = torch.exp(torch.clamp(log_alpha, max=0.0))
        acceptance_probability = torch.nan_to_num(acceptance_probability, nan=0.0)
        accepted = torch.rand_like(acceptance_probability) < acceptance_probability
        next_state = torch.where(accepted[:, None], proposal, state)
        return next_state, {
            "accepted": accepted,
            "acceptance_probability": acceptance_probability,
        }

    @torch.no_grad()
    def sample(
        self, initial_state: Tensor, num_warmup: int, num_samples: int
    ) -> tuple[Tensor, dict[str, object]]:
        """
        Sample from the target distribution using HMC.

        Parameters
        ----------
        initial_state : Tensor (B, d)
            Initial state of the Markov chain.
        num_warmup : int
            Number of warmup transitions discarded before collection.
        num_samples : int
            Number of samples to collect after warmup.

        Returns
        -------
        samples : Tensor (B, num_samples, d)
            Collected samples after warmup.
        diagnostics : dict[str, object]
            Aggregate acceptance rates and per-transition probabilities.
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

        samples = []
        accepted = []
        probabilities = []
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

            if self.pbar:
                iterator.set_postfix(
                    {"acceptance_rate": torch.stack(accepted, dim=1).float().mean().item()}
                )

        accepted_tensor = torch.stack(accepted, dim=1)
        probability_tensor = torch.stack(probabilities, dim=1)
        diagnostics: dict[str, object] = {
            "acceptance_rate": accepted_tensor.float().mean().item(),
            "acceptance_rate_by_chain": accepted_tensor.float().mean(dim=1),
            "acceptance_probability": probability_tensor,
            "warmup_acceptance_rate": (
                warmup_accepted / (num_warmup * state.shape[0]) if num_warmup else None
            ),
        }
        return torch.stack(samples, dim=1), diagnostics


class UniformRiemannHamiltonian(Hamiltonian):
    """Canonical Riemannian Hamiltonian for a density-like target function.

    Parameters
    ----------
    target : Callable[[Tensor], Tensor]
        Positive, unnormalized density mapping batched positions to densities.
    cometric : CoMetric
        Cometric defining the position-dependent kinetic energy.
    """

    def __init__(self, target: Callable[[Tensor], Tensor], cometric: CoMetric):
        super().__init__()
        self.target = target
        self.cometric = cometric
        self.log2pi = torch.log(torch.tensor(2.0 * torch.pi)).item()

    def U(self, z: Tensor) -> Tensor:
        """Return the potential energy induced by the target density.

        Parameters
        ----------
        z : Tensor (b, d)
            Batch of positions.

        Returns
        -------
        Tensor (b,)
            Potential energy ``-log(target(z))`` for each position.
        """
        return -torch.log(self.target(z))

    def K(self, z: Tensor, p: Tensor) -> Tensor:
        """Return the position-dependent kinetic energy.

        Parameters
        ----------
        z : Tensor (b, d)
            Batch of positions.
        p : Tensor (b, d)
            Batch of momenta.

        Returns
        -------
        Tensor (b,)
            Kinetic energy, including the metric-volume normalization term.
        """
        d = z.shape[1]
        p_Ginv_p = self.cometric.cometric(z, p) ** 2
        log_det_G = -self.cometric.inv_logdet(z)
        return 0.5 * p_Ginv_p + 0.5 * log_det_G + 0.5 * d * self.log2pi

    def forward(self, z: Tensor, p: Tensor) -> Tensor:
        """
        Canonical Hamiltonian H(z, p) = U(z) + K(z, p).
        With :
            U(z) = -log target(z)
            K(z, p) = 1/2 p^T G(z)^-1 p + 1/2 log det G(z) + d/2 log 2pi

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        p : Tensor (b,d)
            The momentum.

        Returns
        -------
        Tensor (b,)
            The Hamiltonian.
        """
        return self.U(z) + self.K(z, p)


class VolumeRiemannHamiltonian(UniformRiemannHamiltonian):
    """Canonical Hamiltonian targeting the cometric volume density."""

    def __init__(self, cometric: CoMetric):
        super().__init__(
            target=lambda z: torch.ones(
                z.shape[0], device=z.device
            ),  # Dummy target, not used in this Hamiltonian
            cometric=cometric,
        )

    # We override the U method to return the volume element of the cometric, which is -0.5 * log(det(G(z)^-1)) = 0.5 * log(det(G(z))) = -0.5 * inv_logdet(G(z))
    def U(self, z: Tensor) -> Tensor:
        """Return the potential energy of the cometric volume density.

        Parameters
        ----------
        z : Tensor (b, d)
            Batch of positions.

        Returns
        -------
        Tensor (b,)
            Potential energy induced by the cometric volume element.
        """
        return -0.5 * self.cometric.inv_logdet(z)


class LogTargetRiemannHamiltonian(UniformRiemannHamiltonian):
    """Canonical Riemannian Hamiltonian for an unnormalized log-density.

    Parameters
    ----------
    log_target : Callable[[Tensor], Tensor]
        Unnormalized log-density mapping batched positions to log-densities.
    cometric : CoMetric
        Cometric defining the position-dependent kinetic energy.
    """

    def __init__(self, log_target: Callable[[Tensor], Tensor], cometric: CoMetric):
        super().__init__(
            target=lambda z: torch.ones(z.shape[0], device=z.device),
            cometric=cometric,
        )
        self.log_target = log_target

    def U(self, z: Tensor) -> Tensor:
        """Return the potential energy ``-log_target(z)``.

        Parameters
        ----------
        z : Tensor (b, d)
            Batch of positions.

        Returns
        -------
        Tensor (b,)
            Potential energy of each position.
        """
        return -self.log_target(z)
