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
    valid = torch.ones(b, dtype=torch.bool, device=x_0.device)
    try:
        x_l, log_det = integrator(x_0, L, dirs=dirs)
        return x_l, log_det, valid
    except _LinAlgError:
        pass

    x_l = x_0.clone()
    log_det = torch.zeros(b, device=x_0.device, dtype=x_0.dtype)
    for i in range(b):
        try:
            x_i, log_det_i = integrator(
                x_0[i : i + 1],
                L,
                dirs=None if dirs is None else dirs[i : i + 1],
            )
            x_l[i], log_det[i] = x_i[0], log_det_i[0]
        except _LinAlgError:
            valid[i] = False
    return x_l, log_det, valid


def integrate_hamiltonian_isolating_failures(
    integrator: HamiltonianIntegrator,
    q_0: Tensor,
    p_0: Tensor,
    L: int,
    dirs: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Integrate Hamiltonian states while isolating batch-wide failures."""
    b = q_0.shape[0]
    valid = torch.ones(b, dtype=torch.bool, device=q_0.device)
    try:
        q_l, p_l, log_det = integrator.forward(q_0, p_0, L, dirs=dirs)
        return q_l, p_l, log_det, valid
    except _LinAlgError:
        pass

    q_l = q_0.clone()
    p_l = p_0.clone()
    log_det = torch.zeros(b, device=q_0.device, dtype=q_0.dtype)
    for i in range(b):
        try:
            q_i, p_i, log_det_i = integrator.forward(
                q_0[i : i + 1],
                p_0[i : i + 1],
                L,
                dirs=None if dirs is None else dirs[i : i + 1],
            )
            q_l[i], p_l[i] = q_i[0], p_i[0]
            log_det[i] = log_det_i[0]
        except _LinAlgError:
            valid[i] = False
    return q_l, p_l, log_det, valid


class Sampler(nn.Module):
    """
    Base class for the MCMC samplers. It defines the interface for the samplers.

    Parameters
    ----------
    pbar : bool
        If True, it shows a progress bar when sampling.
    """

    def __init__(self, pbar: bool = False):
        super().__init__()
        self.pbar = pbar

    def sample(
        self,
        z_0: Tensor,
        return_traj: bool = False,
        return_acceptance: bool = False,
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, return the full sampling trajectory, including ``z_0``.
        return_acceptance : bool
            If True, return the sample or trajectory together with the acceptance rate.

        Returns
        -------
        Tensor (b,d) or Tensor (b,N_run+1,d)
            The new sample, or the trajectory when ``return_traj`` is True.
        or
        (Tensor, float)
            The new sample or trajectory and the acceptance rate when
            ``return_acceptance`` is True.
        or
        The return value does not include the acceptance rate otherwise.
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self, z_0: Tensor, n: int, return_acceptance: bool = False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given initial samples z_0, it returns n new samples for each initial sample.

        Beware that tuning both the batch-size and n is important to avoid using too
        much memory.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial samples.
        n : int
            The number of samples to generate for each initial sample.
        return_acceptance : bool
            If True, it returns the samples aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,n,d)
            The new samples.
        or
        (Tensor (b,n,d), float)
            The new samples and the acceptance rate.
        """
        new_samples = []
        acceptance_rate = []

        # If the batch_size is bigger then the number of samples to generate
        # We process the sampling batch-wise, otherwise we process the sampling
        # sample-wise.
        if z_0.shape[0] > n:
            pbar = tqdm(range(n)) if self.pbar else range(n)
            for k in pbar:
                z_new, acc_rate = self.sample(z_0, return_acceptance=True)
                acceptance_rate.append(acc_rate)
                new_samples.append(z_new)
            new_samples = torch.stack(new_samples, dim=1)

        else:
            pbar = tqdm(range(z_0.shape[0])) if self.pbar else range(z_0.shape[0])
            for k in pbar:
                z_batch = z_0[k].repeat(n, 1)
                z_new, acc_rate = self.sample(z_batch, return_acceptance=True)
                acceptance_rate.append(acc_rate)
                new_samples.append(z_new)
            new_samples = torch.stack(new_samples, dim=0)

        acceptance_rate = torch.Tensor(acceptance_rate).mean().item()

        if return_acceptance:
            return new_samples, acceptance_rate
        else:
            return new_samples


class UniformSeparableRiemannHamiltonian(Hamiltonian):
    def __init__(self, target: Callable[[Tensor], Tensor]):
        super().__init__()
        self.target = target

    def U(self, z: Tensor) -> Tensor:
        """
        Compute the potential energy U(z) = -log(sqrt(det(g_inv(z))))= -1/2 * log(det(g_inv(z)))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        potential energy : Tensor (b,)
        """
        return -0.5 * self.target(z).log()

    def K(self, p: Tensor) -> Tensor:
        """
        Compute the kinetic energy K(p) = 1/2 * p^T p

        Parameters
        ----------
        p : Tensor (b,d)
            The momentum.

        Returns
        -------
        kinetic energy : Tensor (b,)
        """
        return 1 / 2 * torch.einsum("bi,bi->b", p, p)  # p^T @ p

    def forward(self, z: Tensor, p: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(z,p) = U(z) + K(p)

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
        return self.U(z) + self.K(p)


class HMCSampler(Sampler):
    """
    Hamiltonian Monte Carlo sampler with a pdf defined on a manifold.
    It uses the leapfrog integrator to propose new samples from the target distribution.
    The hamiltonian dynamics should be of the form:
    H(p,q) = U(q) + p^T p / 2  (separable Hamiltonian)

    Parameters
    ----------
    cometric : CoMetric
        The cometric that defines the target distribution.
    l : int
        The number of leapfrog steps.
    gamma : float
        The step size.
    N_run : int
        The number of iterations.
    bounds : float
        The bounds of the target distribution. This is because the distribution must be supported on a bounded set.
    beta_0 : float
        The initial temperature for the tempering of the momentum.
    std_0 : float
        The standard deviation of the initial momentum.
    pbar : bool
        If True, it shows a progress bar.
    skip_acceptance : bool
        If True, the acceptance step is skipped. This can be used when differentiabily is needed.
    H : Hamiltonian | None
        Optional Hamiltonian override. It must be compatible with the integrator
        and momentum distribution used by this sampler.
    """

    def __init__(
        self,
        target: Callable[[Tensor], Tensor],
        cometric: CoMetric,
        l: int,
        gamma: float,
        N_run: int,
        bounds: float = 1e3,
        beta_0: float = 1,
        std_0: float = 1,
        pbar: bool = False,
        skip_acceptance: bool = False,
        H: Hamiltonian | None = None,
        compile_step: bool = False,
    ):
        super().__init__(pbar)
        self.cometric = cometric
        self.l = l
        self.gamma = gamma
        self.N_run = N_run
        self.bounds = bounds
        self.beta_0_sqrt = beta_0**0.5
        self.std_0 = std_0
        self.skip_acceptance = skip_acceptance
        self.H = H if H is not None else UniformSeparableRiemannHamiltonian(target)
        self.integrator = SeparableLeapfrogIntegrator(
            self.H, self.gamma, 1, compile_step=compile_step
        )

    def proposal_rate(self, z: Tensor, v: Tensor, z_new: Tensor, v_new: Tensor) -> Tensor:
        """
        Compute the proposal rates based on the value of the Hamiltonian.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = torch.exp(-self.H(z_new, v_new) + self.H(z, v))
        return torch.min(torch.ones_like(alpha), alpha)

    def get_alpha(self, z: Tensor, v: Tensor, z_new: Tensor, v_new: Tensor) -> Tensor:
        """
        Compute the proposal rates by combining the proposal_rate method and the bounds.
        If the new sample is out of bounds, the proposal rate is 0.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = self.proposal_rate(z, v, z_new, v_new)
        z_norm = torch.linalg.norm(z_new, dim=-1)
        if self.bounds is not None:
            out_of_bounds = z_norm > self.bounds
            alpha[out_of_bounds] = 0
        return alpha

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample Euclidean momentum from N(0, I), scaled by ``std_0``.

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        v : Tensor (b,d)
            The sampled momentum.
        """
        return torch.randn_like(z) * self.std_0

    @torch.no_grad()
    def sample(
        self,
        z_0: Tensor,
        return_traj: bool = False,
        progress: bool = False,
        return_acceptance: bool = False,
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, return the trajectory, including the initial sample.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, return the sample or trajectory together with the acceptance rate.

        Returns
        -------
        Tensor (b,d) or Tensor (b,N_run+1,d)
            The new sample, or the trajectory when ``return_traj`` is True.
        or
        (Tensor, float)
            The new sample or trajectory and the acceptance rate when
            ``return_acceptance`` is True.
        or
        The return value does not include the acceptance rate otherwise.
        """
        accepted_samples = 0
        z = z_0.clone()

        if return_traj:
            traj = [z.clone()]

        if progress:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            v_0 = self.sample_momentum(z)

            # A linear-algebra failure is batch-wide and anonymous, so isolate
            # the offending samples instead of rejecting the whole batch.
            z_l, v_l, _, valid = integrate_hamiltonian_isolating_failures(
                self.integrator, z, v_0, self.l
            )
            alpha = self.get_alpha(z, v_0, z_l, v_l)
            alpha = torch.where(valid, alpha, torch.zeros_like(alpha))

            if not self.skip_acceptance:
                u = torch.rand_like(alpha)
                mask = alpha >= u
                z = torch.where(mask[:, None], z_l, z)
                accepted_samples += mask.sum().item()
            else:
                z = z_l
                accepted_samples += z.shape[0]

            if return_traj:
                traj.append(z.clone())

            if progress:
                pbar.set_postfix(
                    {"acceptance_rate": accepted_samples / ((k + 1) * z_0.shape[0])}
                )

        acceptance_rate = accepted_samples / (self.N_run * z_0.shape[0])

        if return_traj:
            traj = torch.stack(traj, dim=1)
            if return_acceptance:
                return traj, acceptance_rate
            else:
                return traj
        if return_acceptance:
            return z, acceptance_rate
        return z


class UniformRiemannHamiltonian(Hamiltonian):
    def __init__(self, target: Callable[[Tensor], Tensor], cometric: CoMetric):
        super().__init__()
        self.target = target
        self.cometric = cometric
        self.log2pi = torch.log(torch.tensor(2.0 * torch.pi)).item()

    def U(self, z: Tensor) -> Tensor:
        return -torch.log(self.target(z))

    def K(self, z: Tensor, p: Tensor) -> Tensor:
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
    def __init__(self, cometric: CoMetric):
        super().__init__(
            target=lambda z: torch.ones(
                z.shape[0], device=z.device
            ),  # Dummy target, not used in this Hamiltonian
            cometric=cometric,
        )

    # We override the U method to return the volume element of the cometric, which is -0.5 * log(det(G(z)^-1)) = -0.5 * log(det(G(z))) = -0.5 * inv_logdet(G(z))
    def U(self, z: Tensor) -> Tensor:
        return -0.5 * self.cometric.inv_logdet(z)


class ImplicitMidpointRHMCSampler(Sampler):
    """
    Riemannian HMC sampler with the canonical dynamics, integrated with the
    implicit midpoint scheme.
    Momentum is drawn from N(0, G(z)) and the trajectory follows the canonical Hamiltonian

        H(z, p) = -log target(z) + 1/2 p^T G(z)^-1 p + 1/2 log det G(z),

    with G(z) an arbitrary position-dependent Riemannian metric (e.g. a
    SoftAbs metric built from the Hessian of -log target, or the identity for
    plain HMC). Implicit midpoint applied to this canonical field is
    symplectic, hence exactly volume preserving (det = 1), so acceptance
    reduces to the plain energy difference of H: no Jacobian correction is
    needed.

    Parameters
    ----------
    target : Callable[[Tensor], Tensor]
        Unnormalized target density, maps (b, d) positions to (b,) densities.
        Must be differentiable with torch.
    cometric : CoMetric
        The Riemannian metric G(z) driving the kinetic energy and the
        momentum distribution.
    l : int
        Number of integrator steps per proposal.
    N_fx : int
        Maximum number of Picard fixed-point iterations per midpoint step.
    gamma : float
        Integrator step size.
    N_run : int
        Number of MCMC iterations.
    pbar : bool
        If True, shows a progress bar when sampling.
    skip_acceptance : bool
        If True, proposals are always accepted (no Metropolis correction).
    reduced_flip : bool
        If True, uses the reduced momentum flip (Sohl-Dickstein 2012) on the
        integration direction upon rejection.
    H : Hamiltonian | None
        Optional Hamiltonian override. It must be compatible with the integrator
        and momentum distribution used by this sampler.
    """

    def __init__(
        self,
        target: Callable[[Tensor], Tensor],
        cometric: CoMetric,
        l: int,
        N_fx: int,
        gamma: float,
        N_run: int,
        pbar: bool = False,
        skip_acceptance=False,
        reduced_flip: bool = True,
        H: Hamiltonian | None = None,
        compile_step: bool = False,
    ):
        super().__init__()
        self.cometric = cometric
        self.target = target
        self.H = H if H is not None else UniformRiemannHamiltonian(target, cometric)
        self.l = l
        self.N_fx = N_fx
        self.gamma = gamma
        self.N_run = N_run
        self.pbar = pbar
        self.skip_acceptance = skip_acceptance
        self.reduced_flip = reduced_flip

        self.integrator = HamiltonianImplicitMidpointIntegrator(
            self.H, gamma, N_fx, compile_step=compile_step
        )

    def sample_momentum(self, z: Tensor) -> Tensor:
        """Draw p ~ N(0, G(z))."""
        G = self.cometric.metric_tensor(z)
        p = torch.randn_like(z)
        if self.cometric.is_diag:
            p = p * G.sqrt()
        else:
            p = torch.einsum("bij,bi->bj", mat_sqrt(G), p)
        return p

    def proposal_rate(self, x_0: Tensor, x_l: Tensor, log_det: Tensor) -> Tensor:
        """
        Metropolis-Hastings acceptance probability of the proposal x_l obtained
        from x_0 by the implicit midpoint map with log-Jacobian log_det (= 0
        here, the map is symplectic):

            alpha = min(1, exp(H(x_0) - H(x_l) + log_det)).

        Shapes: x_0, x_l (b, 2d); log_det (b,); output (b,).
        """
        d = x_0.shape[-1] // 2
        log_alpha = self.H(x_0[:, :d], x_0[:, d:]) - self.H(x_l[:, :d], x_l[:, d:]) + log_det
        alpha = torch.exp(torch.clamp(log_alpha, max=0.0))
        return torch.nan_to_num(alpha, nan=0.0)

    @torch.no_grad()
    def sample(
        self,
        z_0: Tensor,
        return_traj: bool = False,
        progress: bool = False,
        return_acceptance: bool = False,
        return_flip: bool = False,
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target
        distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, return the trajectory, including the initial sample.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, return the sample or trajectory together with the acceptance rate.
        return_flip : bool
            If True, it returns the proportion of direction flips over all steps.

        Returns
        -------
        Tensor (b,d) or Tensor (b,N_run+1,d)
            The new sample, or the trajectory when ``return_traj`` is True.
        or
        (Tensor, float)
            The new sample or trajectory and the acceptance rate when
            ``return_acceptance`` is True.
        or
        The return value does not include the acceptance rate otherwise.
        """
        accepted_samples = 0
        flipped_samples = 0
        z = z_0.clone()
        d = z.shape[1]
        dirs = torch.ones(z.shape[0], device=z_0.device, dtype=z_0.dtype)

        if return_traj:
            traj = [z.clone()]

        if progress or self.pbar:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            p_0 = self.sample_momentum(z)
            x_0 = torch.cat([z, p_0], dim=-1)
            # A linear-algebra failure is batch-wide and anonymous, so isolate
            # the offending samples instead of rejecting the whole batch: one
            # chain that has left the domain must not veto the others.
            z_l, p_l, log_det, valid = integrate_hamiltonian_isolating_failures(
                self.integrator, z, p_0, self.l, dirs
            )
            x_l = torch.cat([z_l, p_l], dim=-1)
            alpha = self.proposal_rate(x_0, x_l, log_det)
            alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
            z_l = x_l[:, :d]

            if not self.skip_acceptance:
                u = torch.rand_like(alpha)
                accept_mask = u < alpha
                if self.reduced_flip:
                    rej_idx = (~accept_mask).nonzero(as_tuple=False).squeeze(-1)
                    if rej_idx.numel() > 0:
                        # Reduced flip from Sohl-Dickstein (2012)
                        # applied to the auxiliary integration direction.
                        # The reverse-direction acceptance is used in place
                        # of the momentum-flipped proposal in the paper.
                        # Only computed for rejected samples.
                        z_l_flip, p_l_flip, log_det_flip, valid_flip = (
                            integrate_hamiltonian_isolating_failures(
                                self.integrator,
                                z[rej_idx],
                                p_0[rej_idx],
                                self.l,
                                -dirs[rej_idx],
                            )
                        )
                        x_l_flip = torch.cat([z_l_flip, p_l_flip], dim=-1)
                        alpha_flip_rej = self.proposal_rate(
                            x_0[rej_idx], x_l_flip, log_det_flip
                        )
                        alpha_flip_rej = torch.where(
                            valid_flip, alpha_flip_rej, torch.zeros_like(alpha_flip_rej)
                        )
                        alpha_flip = torch.zeros_like(alpha)
                        alpha_flip[rej_idx] = alpha_flip_rej
                        p_flip = (alpha_flip - alpha).clamp(min=0)
                        flip_mask = ~accept_mask & (u < alpha + p_flip)
                    else:
                        flip_mask = ~accept_mask  # all False, no rejections
                else:
                    flip_mask = ~accept_mask
                z = torch.where(accept_mask[:, None], z_l, z)
                dirs = torch.where(flip_mask, -dirs, dirs)
                accepted_samples += accept_mask.sum().item()
                flipped_samples += flip_mask.sum().item()
            else:
                # Even without the Metropolis correction, never adopt an
                # invalid state (integration blow-up, or a finite state
                # outside the region where the metric is defined): the
                # momentum sampler could not be evaluated there. Such states
                # are exactly those with alpha = 0 (NaN energies are mapped
                # to alpha = 0 by proposal_rate).
                valid_mask = torch.isfinite(z_l).all(dim=-1) & (alpha > 0)
                z = torch.where(valid_mask[:, None], z_l, z)
                accepted_samples += z.shape[0]

            if return_traj:
                traj.append(z.clone())

            if progress or self.pbar:
                pbar.set_postfix(
                    {"acceptance_rate": accepted_samples / ((k + 1) * z_0.shape[0])}
                )

        acceptance_rate = accepted_samples / (self.N_run * z_0.shape[0])
        flip_rate = flipped_samples / (self.N_run * z_0.shape[0])

        if return_traj:
            traj = torch.stack(traj, dim=1)
            if return_acceptance:
                return (
                    (traj, acceptance_rate, flip_rate)
                    if return_flip
                    else (traj, acceptance_rate)
                )
            return (traj, flip_rate) if return_flip else traj
        if return_acceptance:
            return (z, acceptance_rate, flip_rate) if return_flip else (z, acceptance_rate)
        return (z, flip_rate) if return_flip else z


class ImplicitRHMCSampler(Sampler):
    """
    Riemannian Hamiltonian Monte Carlo sampler with a pdf defined on a manifold.
    It uses the leapfrog integrator to propose new samples from the target distribution.
    The leapfrog integrator is solved implicitly.
    It uses a tempering scheme on the momentum.
    Here the target distribution is defined by the volume element of the cometric.

    Parameters
    ----------
    cometric : CoMetric
        The cometric that defines the target distribution.
    l : int
        The number of leapfrog steps.
    N_fx : int
        The number of fixed point iterations.
    gamma : float
        The step size.
    N_run : int
        The number of iterations.
    std_0 : float
        The standard deviation of the initial momentum.
    bounds : float
        The bounds of the target distribution. This is because the distribution must be supported on a bounded set.
    beta_0 : float
        The initial temperature for the tempering of the momentum.
    pbar : bool
        If True, it shows a progress bar.
    skip_acceptance : bool
        If True, the acceptance step is skipped. This can be used when differentiabily is needed.
    threshold_fx : float
        The threshold for the fixed point iterations. If the maximum change in the fixed point iterations is less than this threshold, the iterations are stopped.
    H : Hamiltonian | None
        Optional Hamiltonian override. It must be compatible with the integrator
        and momentum distribution used by this sampler.
    """

    def __init__(
        self,
        cometric: CoMetric,
        l: int,
        N_fx: int,
        gamma: float,
        N_run: int,
        std_0: float = 1.0,
        bounds: float = 1e3,
        beta_0: float = 1,
        pbar: bool = False,
        skip_acceptance: bool = False,
        threshold_fx: float = 1e-5,
        H: Hamiltonian | None = None,
        compile_step: bool = False,
    ):
        super().__init__(pbar)
        self.cometric = cometric
        self.l = l
        self.N_fx = N_fx
        self.gamma = gamma
        self.N_run = N_run
        self.std_0 = std_0
        self.bounds = bounds
        self.beta_0_sqrt = beta_0**0.5
        self.skip_acceptance = skip_acceptance
        self.threshold_fx = threshold_fx
        self.H = H if H is not None else VolumeRiemannHamiltonian(cometric)
        self.integrator = ImplicitLeapfrogIntegrator(
            self.H, gamma, N_fx, compile_step=compile_step
        )

    def tempering(self, k) -> float:
        """
        Compute the tempering coefficient at step k.

        Parameters
        ----------
        k : int
            The current step.

        Returns
        -------
        beta_k : float
            The tempering coefficient at step k.
        """
        beta_k = ((1 - 1 / self.beta_0_sqrt) * (k / self.N_run) ** 2) + 1 / self.beta_0_sqrt
        return beta_k

    def proposal_rate(self, z: Tensor, v: Tensor, z_new: Tensor, v_new: Tensor) -> Tensor:
        """
        Compute the proposal rates based on the value of the Hamiltonian.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = torch.exp(-self.H(z_new, v_new) + self.H(z, v))
        return torch.min(torch.ones_like(alpha), alpha)

    def get_alpha(self, z: Tensor, v: Tensor, z_new: Tensor, v_new: Tensor) -> Tensor:
        """
        Compute the proposal rates by combining the proposal_rate method and the bounds.
        If the new sample is out of bounds, the proposal rate is 0.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = self.proposal_rate(z, v, z_new, v_new)
        z_norm = torch.linalg.norm(z_new, dim=-1)
        out_of_bounds = z_norm > self.bounds
        alpha[out_of_bounds] = 0
        return alpha

    def leapfrog(self, z: Tensor, v: Tensor, return_traj: bool = False) -> Tensor:
        """
        Perform l leapfrog steps with tempering of the momentum.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        return_traj : bool
            If True, it returns the trajectory of the samples over the l leapfrog steps.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.
        or
        (Tensor (b,l+1,d), Tensor (b,l+1,d))
            The trajectory of the positions and velocities over the l leapfrog steps.
        """
        z_new, v_new = z.clone(), v.clone()
        if return_traj:
            traj_q = [z_new.clone()]
            traj_p = [v_new.clone()]
        beta_k_minus_1_sqrt = self.beta_0_sqrt
        if self.l <= 1:
            raise ValueError("l must be greater than 1.")
        for k in range(self.l - 1):
            z_new, v_new, _ = self.integrator(z_new, v_new, 2)
            beta_k_sqrt = self.tempering(k)
            v_new = (beta_k_minus_1_sqrt / beta_k_sqrt) * v_new
            beta_k_minus_1_sqrt = beta_k_sqrt

            if return_traj:
                traj_q.append(z_new.clone())
                traj_p.append(v_new.clone())

        if return_traj:
            traj_q = torch.stack(traj_q, dim=1)
            traj_p = torch.stack(traj_p, dim=1)
            return traj_q, traj_p

        return z_new, v_new

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample the momentum from the Gaussian distribution N(0, g(z))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        v : Tensor (b,d)
            The sampled momentum.
        """
        g = self.cometric.metric_tensor(z)
        v = torch.randn_like(z)
        if self.cometric.is_diag:
            v = v * g.sqrt() * self.std_0
        else:
            v = torch.einsum("bij,bi->bj", mat_sqrt(g), v) * self.std_0
        return v

    @torch.no_grad()
    def sample(
        self,
        z_0: Tensor,
        return_traj: bool = False,
        progress: bool = False,
        return_acceptance: bool = False,
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, return the trajectory, including the initial sample.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, return the sample or trajectory together with the acceptance rate.

        Returns
        -------
        Tensor (b,d) or Tensor (b,N_run+1,d)
            The new sample, or the trajectory when ``return_traj`` is True.
        or
        (Tensor, float)
            The new sample or trajectory and the acceptance rate when
            ``return_acceptance`` is True.
        or
        The return value does not include the acceptance rate otherwise.
        """
        accepted_samples = 0
        z = z_0.clone()

        if return_traj:
            traj = [z.clone()]

        if progress:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            v_0 = self.sample_momentum(z)
            try:
                z_l, v_l = self.leapfrog(z, v_0)
                alpha = self.get_alpha(z, v_0, z_l, v_l)
            except _LinAlgError:
                # @TODO: Handle this error properly.
                # Not the best way to handle this error.
                # Because a single LinAlgError for a given sample
                # will stop the whole process even for other valid samples.
                alpha = torch.zeros(z.shape[0], device=z.device)
                z_l = z.clone()

            if not self.skip_acceptance:
                u = torch.rand_like(alpha)
                mask = alpha >= u
                z = torch.where(mask[:, None], z_l, z)
                accepted_samples += mask.sum().item()
            else:
                z = z_l
                accepted_samples += z.shape[0]

            if return_traj:
                traj.append(z.clone())

            if progress:
                pbar.set_postfix(
                    {"acceptance_rate": accepted_samples / ((k + 1) * z_0.shape[0])}
                )

        acceptance_rate = accepted_samples / (self.N_run * z_0.shape[0])

        if return_traj:
            traj = torch.stack(traj, dim=1)
            if return_acceptance:
                return traj, acceptance_rate
            else:
                return traj
        if return_acceptance:
            return z, acceptance_rate
        return z


class ExplicitRHMCSampler(Sampler):
    """
    Explicit Riemannian Hamiltonian Monte Carlo sampler with a pdf defined on a manifold.
    It uses the augmented leapfrog integrator to propose new samples from the target distribution.
    It uses a tempering scheme on the momentum.
    Here the target distribution is defined by the volume element of the cometric.
    But this class is easily heritable to define other target distributions. Just redefine
    the p_target method.

    `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`
    by Cobb et Baydin et al (2019).

    Parameters
    ----------
    cometric : CoMetric
        The cometric that defines the target distribution.
    l : int
        The number of leapfrog steps.
    gamma : float
        The step size.
    omega : float
        The binding parameter
    N_run : int
        The number of iterations.
    std_0 : float
        The standard deviation of the initial momentum.
    bounds : float
        The bounds of the target distribution. This is because the distribution must be supported on a bounded set.
    beta_0 : float
        The initial temperature for the tempering of the momentum.
    pbar : bool
        If True, it shows a progress bar.
    skip_acceptance : bool
        If True, the acceptance step is skipped. This can be used when differentiabily is needed.
    H : Hamiltonian | None
        Optional Hamiltonian override. It must be compatible with the integrator
        and momentum distribution used by this sampler.
    """

    def __init__(
        self,
        cometric: CoMetric,
        l: int,
        gamma: float,
        omega: float,
        N_run: int,
        bounds: float = 1e3,
        std_0: float = 1.0,
        beta_0: float = 1,
        pbar: bool = False,
        skip_acceptance: bool = False,
        H: Hamiltonian | None = None,
        compile_step: bool = False,
    ):
        super().__init__(pbar)
        self.cometric = cometric
        self.l = l
        self.gamma = gamma
        self.omega = omega
        self.N_run = N_run
        self.std_0 = std_0
        self.bounds = bounds
        self.beta_0_sqrt = beta_0**0.5
        self.skip_acceptance = skip_acceptance

        c = torch.Tensor([2 * self.omega * self.gamma]).cos()
        s = torch.Tensor([2 * self.omega * self.gamma]).sin()
        self.register_buffer("c", c, persistent=False)
        self.register_buffer("s", s, persistent=False)

        self.H = H if H is not None else VolumeRiemannHamiltonian(cometric)
        self.integrator = ExplicitLeapfrogIntegrator(
            self.H, gamma, omega, compile_step=compile_step
        )

    def tempering(self, k) -> float:
        """
        Compute the tempering coefficient at step k.

        Parameters
        ----------
        k : int
            The current step.

        Returns
        -------
        beta_k : float
            The tempering coefficient at step k.
        """
        beta_k = ((1 - 1 / self.beta_0_sqrt) * (k / self.N_run) ** 2) + 1 / self.beta_0_sqrt
        return beta_k

    def proposal_rate(
        self,
        z_l_0: Tensor,
        v_l_0: Tensor,
        z_0: Tensor,
        v0: Tensor,
    ) -> Tensor:
        """
        Compute the proposal rates based on the value of the Hamiltonian.

        Parameters
        ----------
        z_l_0 : Tensor (b,d)
            The new position of the first state.
        v_l_0 : Tensor (b,d)
            The new velocity of the first state.
        z_0 : Tensor (b,d)
            The initial position of the first state.
        v0 : Tensor (b,d)
            The initial velocity of the first state.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        H_new = self.H(z_l_0, v_l_0)
        H_old = self.H(z_0, v0)
        alpha = torch.exp(-H_new + H_old)
        return torch.min(torch.ones_like(alpha), alpha)

    def get_alpha(
        self,
        z_l_0: Tensor,
        v_l_0: Tensor,
        z_l_1: Tensor,
        z_0: Tensor,
        v0: Tensor,
    ) -> Tensor:
        """
        Compute the proposal rates by combining the proposal_rate method and the bounds.
        If the new sample is out of bounds, the proposal rate is 0.

        Parameters
        ----------
        z_l_0 : Tensor (b,d)
            The new position of the first state.
        v_l_0 : Tensor (b,d)
            The new velocity of the first state.
        z_l_1 : Tensor (b,d)
            The new position of the second state.
        z_0 : Tensor (b,d)
            The initial position of the first state.
        v0 : Tensor (b,d)
            The initial velocity of the first state.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = self.proposal_rate(z_l_0, v_l_0, z_0, v0)
        if self.bounds is not None:
            z_0_norm = torch.linalg.norm(z_l_0, dim=-1)
            z_1_norm = torch.linalg.norm(z_l_1, dim=-1)
            z_norm = torch.max(z_0_norm, z_1_norm)
            out_of_bounds = z_norm > self.bounds
            alpha[out_of_bounds] = 0
        return alpha

    def leapfrog(
        self, z_0: Tensor, v0: Tensor, z_1: Tensor, v1: Tensor, return_traj: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform l leapfrog steps with tempering of the momentum.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial position of the first state.
        v0 : Tensor (b,d)
            The initial velocity of the first state.
        z_1 : Tensor (b,d)
            The initial position of the second state.
        v1 : Tensor (b,d)
            The initial velocity of the second state.
        return_traj : bool
            If True, it returns the trajectory of the samples over the l leapfrog steps.

        Returns
        -------
        z_l_0 : Tensor (b,d)
            The new position of the first state.
        v_l_0 : Tensor (b,d)
            The new velocity of the first state.
        z_l_1 : Tensor (b,d)
            The new position of the second state.
        v_l_1 : Tensor (b,d)
            The new velocity of the second state.
        or
        (Tensor (b,l+1,d), Tensor (b,l+1,d), Tensor (b,l+1,d), Tensor (b,l+1,d))
            The trajectory of the positions and velocities over the l leapfrog steps.
        """
        z_l_0, v_l_0, z_l_1, v_l_1 = z_0.clone(), v0.clone(), z_1.clone(), v1.clone()
        if return_traj:
            traj_q_0 = [z_l_0.clone()]
            traj_p_0 = [v_l_0.clone()]
            traj_q_1 = [z_l_1.clone()]
            traj_p_1 = [v_l_1.clone()]
        beta_k_minus_1_sqrt = self.beta_0_sqrt
        if self.l <= 1:
            raise ValueError("l must be greater than 1.")
        for k in range(self.l - 1):
            z_l_0, v_l_0, z_l_1, v_l_1, _ = self.integrator.forward_augmented(
                z_l_0,
                v_l_0,
                2,
                q_1=z_l_1,
                p_1=v_l_1,
            )
            beta_k_sqrt = self.tempering(k)
            v_l_0 = (beta_k_minus_1_sqrt / beta_k_sqrt) * v_l_0
            v_l_1 = (beta_k_minus_1_sqrt / beta_k_sqrt) * v_l_1
            beta_k_minus_1_sqrt = beta_k_sqrt

            if return_traj:
                traj_q_0.append(z_l_0.clone())
                traj_p_0.append(v_l_0.clone())
                traj_q_1.append(z_l_1.clone())
                traj_p_1.append(v_l_1.clone())

        if return_traj:
            traj_q_0 = torch.stack(traj_q_0, dim=1)
            traj_p_0 = torch.stack(traj_p_0, dim=1)
            traj_q_1 = torch.stack(traj_q_1, dim=1)
            traj_p_1 = torch.stack(traj_p_1, dim=1)
            return traj_q_0, traj_p_0, traj_q_1, traj_p_1

        return z_l_0, v_l_0, z_l_1, v_l_1

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample the momentum from the Gaussian distribution N(0, g(z))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        v : Tensor (b,d)
            The sampled momentum.
        """
        g = self.cometric.metric_tensor(z)
        v = torch.randn_like(z)
        if self.cometric.is_diag:
            v = v * g.sqrt() * self.std_0
        else:
            v = torch.einsum("bij,bi->bj", mat_sqrt(g), v) * self.std_0
        return v

    def sample(
        self,
        z_0: Tensor,
        return_traj: bool = False,
        progress: bool = False,
        return_acceptance: bool = False,
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, return the trajectory, including the initial sample.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, return the sample or trajectory together with the acceptance rate.

        Returns
        -------
        Tensor (b,d) or Tensor (b,N_run+1,d)
            The new sample, or the trajectory when ``return_traj`` is True.
        or
        (Tensor, float)
            The new sample or trajectory and the acceptance rate when
            ``return_acceptance`` is True.
        or
        The return value does not include the acceptance rate otherwise.
        """
        accepted_samples = 0
        z_0 = z_0.clone()
        z_1 = z_0.clone()

        if return_traj:
            traj = [z_0.clone()]

        if progress:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            v_0 = self.sample_momentum(z_0)
            v_1 = v_0.clone()

            z_l_0, v_l_0, z_l_1, v_l_1 = self.leapfrog(z_0, v_0, z_1, v_1)

            if not self.skip_acceptance:
                alpha = self.get_alpha(z_l_0, v_l_0, z_l_1, z_0, v_0)

                u = torch.rand_like(alpha)
                mask = alpha >= u
                z_0 = torch.where(mask[:, None], z_l_0, z_0)
                z_1 = torch.where(mask[:, None], z_l_1, z_1)
                accepted_samples += mask.sum().item()
            else:
                z_0 = z_l_0
                z_1 = z_l_1
                accepted_samples += z_0.shape[0]

            if return_traj:
                traj.append(z_0.clone())
            if progress:
                pbar.set_postfix(
                    {"acceptance_rate": accepted_samples / ((k + 1) * z_0.shape[0])}
                )

        acceptance_rate = accepted_samples / (self.N_run * z_0.shape[0])

        if return_traj:
            traj = torch.stack(traj, dim=1)
            if return_acceptance:
                return traj, acceptance_rate
            else:
                return traj
        if return_acceptance:
            return z_0, acceptance_rate
        return z_0
