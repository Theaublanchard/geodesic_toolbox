import torch
from torch import nn
from tqdm import tqdm
from torch import Tensor
from torch.linalg import LinAlgError as _LinAlgError
from typing import Callable

from geodesic_toolbox.cometric import CoMetric, mat_sqrt, RandersMetrics
from geodesic_toolbox.integrators import (
    ImplicitLeapfrogIntegrator,
    Integrator,
    Hamiltonian,
    SeparableLeapfrogIntegrator,
    HamiltonianIntegrator,
    HamiltonianImplicitMidpointIntegrator,
    ExplicitLeapfrogIntegrator,
)


def integrate_isolating_failures(
    integrator: Integrator, x_0: Tensor, L: int, dirs: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Integrate a batch, isolating the samples whose integration fails.

    ``torch.linalg`` errors are batch-wide: the exception names no sample, so a
    caller that simply catches it has to throw away every proposal in the batch.
    In MCMC that is severe -- one chain which has left the domain rejects the
    proposals of all the others, and with enough chains the acceptance rate
    collapses to 0 while each chain on its own would have sampled fine.

    On failure the batch is retried one sample at a time, which identifies the
    offenders exactly; only they are marked invalid. The retry costs one call
    per sample but is paid only on a failing step. ``cometric.safe_eigh``
    removes the common cause (a non-finite Hessian reaching eigh), so this is
    the backstop for the remaining factorizations -- inverse, Cholesky, slogdet.

    Parameters
    ----------
    integrator : Integrator
        The integrator function to use.
    x_0 : Tensor (b, d)
        Batch of initial states.
    L : int
        Number of integration steps to perform.
    dirs : Tensor (b,) | None
        Per-sample integration direction, forwarded to the integrator.

    Returns
    -------
    x_l : Tensor (b, d)
        Final states.
    log_det : Tensor (b,)
        Log-Jacobians of the transformation.
    valid : Tensor (b,) bool
        Validity mask. Where the mask is False the integration failed, the state is left at ``x_0`` and the caller must reject the sample.
    """
    b = x_0.shape[0]

    def valid_outputs(x_l: Tensor, log_det: Tensor) -> Tensor:
        """Return the samples with finite states and log-Jacobians."""
        return torch.isfinite(x_l).all(dim=-1) & torch.isfinite(log_det)

    try:
        x_l, log_det = integrator(x_0, L, dirs=dirs)
        valid = valid_outputs(x_l, log_det)
        if valid.all():
            return x_l, log_det, valid
    except _LinAlgError:
        x_l = x_0
        log_det = torch.zeros(b, device=x_0.device, dtype=x_0.dtype)

    x_l = x_0.clone()
    log_det = torch.zeros(b, device=x_0.device, dtype=x_0.dtype)
    valid = torch.zeros(b, dtype=torch.bool, device=x_0.device)
    for i in range(b):
        try:
            x_i, log_det_i = integrator(
                x_0[i : i + 1],
                L,
                dirs=None if dirs is None else dirs[i : i + 1],
            )
            valid_i = valid_outputs(x_i, log_det_i)[0]
            if valid_i:
                x_l[i], log_det[i] = x_i[0], log_det_i[0]
                valid[i] = True
        except _LinAlgError:
            continue
    return x_l, log_det, valid


def integrate_hamiltonian_isolating_failures(
    integrator: HamiltonianIntegrator,
    q_0: Tensor,
    p_0: Tensor,
    L: int,
    dirs: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Integrate Hamiltonian states while isolating batch-wide failures.

    Non-finite positions, momenta, and log-Jacobians are treated as
    per-chain integration failures.

    Parameters
    ----------
    integrator : HamiltonianIntegrator
        Hamiltonian integrator used to evolve the states.
    q_0 : Tensor (b, d)
        Batch of initial positions.
    p_0 : Tensor (b, d)
        Batch of initial momenta.
    L : int
        Number of integration steps to perform.
    dirs : Tensor (b,) | None
        Optional per-sample integration directions, forwarded to the
        integrator.

    Returns
    -------
    q_l : Tensor (b, d)
        Final positions. Failed samples retain their initial positions.
    p_l : Tensor (b, d)
        Final momenta. Failed samples retain their initial momenta.
    log_det : Tensor (b,)
        Log-Jacobians of the transformation. Failed samples receive zero.
    valid : Tensor (b,) bool
        Per-sample validity mask. False entries identify failed integrations.
    """
    b = q_0.shape[0]

    def valid_outputs(q_l: Tensor, p_l: Tensor, log_det: Tensor) -> Tensor:
        """Return the samples with finite positions, momenta, and Jacobians."""
        return (
            torch.isfinite(q_l).all(dim=-1)
            & torch.isfinite(p_l).all(dim=-1)
            & torch.isfinite(log_det)
        )

    try:
        q_l, p_l, log_det = integrator.forward(q_0, p_0, L, dirs=dirs)
        valid = valid_outputs(q_l, p_l, log_det)
        if valid.all():
            return q_l, p_l, log_det, valid
    except _LinAlgError:
        q_l = q_0
        p_l = p_0
        log_det = torch.zeros(b, device=q_0.device, dtype=q_0.dtype)

    q_l = q_0.clone()
    p_l = p_0.clone()
    log_det = torch.zeros(b, device=q_0.device, dtype=q_0.dtype)
    valid = torch.zeros(b, dtype=torch.bool, device=q_0.device)
    for i in range(b):
        try:
            q_i, p_i, log_det_i = integrator.forward(
                q_0[i : i + 1],
                p_0[i : i + 1],
                L,
                dirs=None if dirs is None else dirs[i : i + 1],
            )
            valid_i = valid_outputs(q_i, p_i, log_det_i)[0]
            if valid_i:
                q_l[i], p_l[i] = q_i[0], p_i[0]
                log_det[i] = log_det_i[0]
                valid[i] = True
        except _LinAlgError:
            continue
    return q_l, p_l, log_det, valid


class Sampler(nn.Module):
    """Base class for batch MCMC samplers.

    Parameters
    ----------
    pbar : bool
        If True, display a progress bar while collecting samples.
    """

    def __init__(self, pbar: bool = False):
        super().__init__()
        self.pbar = pbar

    def step(self, state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Advance every chain by one MCMC transition.

        Parameters
        ----------
        state : Tensor (num_chains, dimension)
            Current position of each chain.

        Returns
        -------
        next_state : Tensor (num_chains, dimension)
            Position after the transition.
        diagnostics : dict[str, Tensor]
            Per-chain transition diagnostics.
        """
        raise NotImplementedError

    def sample(
        self, initial_state: Tensor, num_warmup: int, num_samples: int
    ) -> tuple[Tensor, dict[str, object]]:
        """Run warmup and collect post-warmup states.

        Parameters
        ----------
        initial_state : Tensor (num_chains, dimension)
            Initial position of each chain.
        num_warmup : int
            Number of transitions to discard before collecting samples.
        num_samples : int
            Number of post-warmup transitions to retain.

        Returns
        -------
        samples : Tensor (num_chains, num_samples, dimension)
            Collected chain states. Rejected transitions remain repeated states.
        diagnostics : dict[str, object]
            Aggregate and per-transition sampling diagnostics.
        """
        raise NotImplementedError


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
        momentum = torch.randn_like(state) * self.hamiltonian.momentum_std
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
        for _ in range(num_warmup):
            state, transition = self.step(state)
            warmup_accepted += transition["accepted"].sum().item()

        samples = []
        accepted = []
        probabilities = []
        iterator = (
            tqdm(range(num_samples), desc="Sampling", unit="draws")
            if self.pbar
            else range(num_samples)
        )
        for _ in iterator:
            state, transition = self.step(state)
            samples.append(state.clone())
            accepted.append(transition["accepted"])
            probabilities.append(transition["acceptance_probability"])

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
        compile_step: bool = False,
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
        for _ in range(num_warmup):
            state, transition = self.step(state)
            warmup_accepted += transition["accepted"].sum().item()

        samples, accepted, probabilities, valid = [], [], [], []
        flipped = []
        iterator = (
            tqdm(range(num_samples), desc="Sampling", unit="draws")
            if self.pbar
            else range(num_samples)
        )
        for _ in iterator:
            state, transition = self.step(state)

            samples.append(state.clone())
            accepted.append(transition["accepted"])
            probabilities.append(transition["acceptance_probability"])
            valid.append(transition["valid"])

            if "flipped" in transition:
                flipped.append(transition["flipped"])

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
        compile_step: bool = False,
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
        compile_step: bool = False,
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
        fixed_point_iterations: int=1,
        log_target: Callable[[Tensor], Tensor] | None = None,
        momentum_std: float = 1.0,
        bounds: float | None = None,
        pbar: bool = False,
        H: Hamiltonian | None = None,
        compile_step: bool = False,
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
