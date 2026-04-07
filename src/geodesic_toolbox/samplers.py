import torch
from torch import nn
from tqdm import tqdm
from torch import Tensor
from torch.linalg import LinAlgError as _LinAlgError

from .cometric import CoMetric, mat_sqrt, RandersMetrics


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

    def sample(self, z_0: Tensor, return_acceptance: bool) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,d), float)
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


class ConditionnalSampler(Sampler):
    """
    Base class for the conditionnal samplers.
    These samplers generate samples from a target distribution conditioned on a class.

    Parameters
    ----------
    pbar : bool
        If True, it shows a progress bar when sampling.
    """

    def __init__(self, pbar: bool = False):
        super().__init__(pbar)

    def sample(
        self, z_0: Tensor, return_acceptance: bool = False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution with the associated class.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        x : Tensor (b,d)
            The new samples.
        y : Tensor (b,)
            The class of the samples.
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self, z_0: Tensor, n: int, return_acceptance: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, float]:
        """
        Given initial samples z_0, it returns n new samples for each initial sample
        with the associated class.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial samples.
        n : int
            The number of samples to generate for each initial sample.
        return_acceptance : bool
            If True, it returns the samples and the class aswell as the acceptance rate.

        Returns
        -------
        x : Tensor (b,n,d)
            The new samples.
        y : Tensor (b,n,)
            The class of the samples.
        """
        new_samples = []
        new_classes = []
        acceptance_rate = []

        # If the batch_size is bigger then the number of samples to generate
        # We process the sampling batch-wise, otherwise we process the sampling
        # sample-wise.
        if z_0.shape[0] > n:
            pbar = tqdm(range(n)) if self.pbar else range(n)
            for k in pbar:
                x, y, acc_rate = self.sample(z_0, return_acceptance=True)
                new_samples.append(x)
                new_classes.append(y)
                acceptance_rate.append(acc_rate)
            new_classes = torch.stack(new_classes, dim=1)
            new_samples = torch.stack(new_samples, dim=1)

        else:
            pbar = tqdm(range(z_0.shape[0])) if self.pbar else range(z_0.shape[0])
            for k in pbar:
                z_batch = z_0[k].repeat(n, 1)
                x, y, acc_rate = self.sample(z_batch, return_acceptance=True)
                new_samples.append(x)
                new_classes.append(y)
                acceptance_rate.append(acc_rate)
            new_classes = torch.stack(new_classes, dim=0)
            new_samples = torch.stack(new_samples, dim=0)

        acceptance_rate = torch.mean(torch.Tensor(acceptance_rate)).item()

        if return_acceptance:
            return new_samples, new_classes, acceptance_rate
        else:
            return new_samples, new_classes


class ConstantClassSampler(ConditionnalSampler):
    """
    Conditionnal sampler that generates samples from a target distribution with a constant class.

    Parameters
    ----------
    sampler : Sampler
        The sampler to use.
    y : int
        The class of the samples.
    pbar : bool
        If True, it shows a progress bar when sampling.
    """

    def __init__(self, sampler: Sampler, y: int, pbar: bool = False):
        super().__init__(pbar)
        self.sampler = sampler
        self.y = y

    def sample(
        self, z_0: Tensor, return_acceptance: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, float]:
        y = torch.full((z_0.shape[0],), self.y, dtype=torch.long, device=z_0.device)
        if return_acceptance:
            x, acc_rate = self.sampler.sample(z_0, return_acceptance=True)
            return x, y, acc_rate
        else:
            x = self.sampler.sample(z_0)
            return x, y


class MixtureOfSamplers(nn.Module):
    """
    Mixture of samplers. It generates samples from a mixture of samplers.

    Parameters
    ----------
    samplers : list[Sampler]
        The samplers to use
    """

    def __init__(self, samplers: list[Sampler]):
        super().__init__()
        self.samplers = samplers
        self.n_samplers = len(samplers)

    def __iter__(self):
        yield from self.samplers

    def forward(
        self, z_0: Tensor, n: int, return_acceptance: bool = False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Samples n new samples from the mixture of samplers.
        Each sampler is used n//n_samplers times.
        If n is not divisible by n_samplers, the remaining samples are sampled from a random sampler.

        Parameters
        ----------
        n : int
            Number of samples to generate
        z_0 : Tensor (b, d)
            The initial samples.

        Returns
        -------
        x : Tensor (b, n, ...)
            The samples generated
        """

        x = []
        acceptance_rate = []

        for i in range(self.n_samplers):
            n_i = n // self.n_samplers
            if n_i == 0:
                continue
            x_i, acc_rate = self.samplers[i](z_0, n_i, return_acceptance=True)
            acceptance_rate.append(acc_rate)
            x.append(x_i)

        # If n%self.n_samplers != 0, we need to sample the remaining samples
        # Just take a random sampler and sample the remaining samples
        if n % self.n_samplers != 0:
            i = torch.randint(0, self.n_samplers, (1,)).item()
            n_i = n % self.n_samplers
            x_i, acc_rate = self.samplers[i](z_0, n_i, return_acceptance=True)
            acceptance_rate.append(acc_rate)
            x_i = self.samplers[i](z_0, n_i)
            x.append(x_i)

        x = torch.cat(x, dim=1)
        acceptance_rate = torch.mean(torch.stack(acceptance_rate)).item()

        if return_acceptance:
            return x, acceptance_rate
        else:
            return x


class MixtureOfCondtionnalSamplers(nn.Module):
    """
    Mixture of conditionnal samplers. It generates samples from a mixture of conditionnal samplers.

    Parameters
    ----------
    samplers : list[ConditionnalSampler]
        The conditionnal samplers to use
    """

    def __init__(self, samplers: list[ConditionnalSampler]):
        super().__init__()
        self.samplers = samplers
        self.n_samplers = len(samplers)

    def forward(
        self, z_0: Tensor, n: int, return_acceptance: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, float]:
        """
        Samples n new samples from the mixture of samplers.
        Each sampler is used n//n_samplers times.
        If n is not divisible by n_samplers, the remaining samples are sampled from a random sampler

        Parameters
        ----------
        n : int
            Number of samples to generate
        z_0 : Tensor (b, d)
            The initial samples.

        Returns
        -------
        x : Tensor (b, n, ...)
            The samples generated
        y : Tensor (b, n,)
            The class of the samples
        """

        x, y = [], []
        acceptance_rate = []

        for i in range(self.n_samplers):
            n_i = n // self.n_samplers
            x_i, y_i, acc_rate = self.samplers[i](z_0, n_i, return_acceptance=True)
            x.append(x_i)
            y.append(y_i)
            acceptance_rate.append(acc_rate)

        # If n%self.n_samplers != 0, we need to sample the remaining samples
        # Just take a random sampler and sample the remaining samples
        if n % self.n_samplers != 0:
            i = torch.randint(0, self.n_samplers, (1,)).item()
            n_i = n % self.n_samplers
            x_i, y_i, acc_rate = self.samplers[i](z_0, n_i, return_acceptance=True)
            x.append(x_i)
            y.append(y_i)
            acceptance_rate.append(acc_rate)

        x = torch.cat(x, dim=1)
        y = torch.cat(y, dim=1)
        acceptance_rate = torch.mean(torch.stack(acceptance_rate)).item()

        if return_acceptance:
            return x, y, acceptance_rate
        else:
            return x, y


class HMCSampler(Sampler):
    """
    Hamiltonian Monte Carlo sampler with a pdf defined on a manifold.
    It uses the leapfrog integrator to propose new samples from the target distribution.
    The hamiltonian dynamics is :
    H(p,q) = U(q) + p^T p / 2  (separable Hamiltonian)
    It uses a tempering scheme on the momentum.
    Here the target distribution is defined by the volume element of the cometric.

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
    """

    def __init__(
        self,
        cometric: CoMetric,
        l: int,
        gamma: float,
        N_run: int,
        bounds: float = 1e3,
        beta_0: float = 1,
        std_0: float = 1,
        pbar: bool = False,
        skip_acceptance: bool = False,
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

        # @TODO : make this faster
        no_batch_forward = lambda x: self.U(x.unsqueeze(0)).squeeze(0)
        self._grad_U = torch.vmap(torch.func.jacrev(no_batch_forward))
        self.grad_U = lambda z: self._grad_U(z).squeeze(1)

    def p_target(self, z: Tensor) -> Tensor:
        """
        Compute the target distribution p(z) = sqrt(det(g_inv(z)))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        p(z) : Tensor (b,)
            The target distribution.
        """
        g_inv = self.cometric(z)
        return g_inv.det().abs().sqrt()

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
        g_inv = self.cometric(z)
        return -0.5 * torch.logdet(g_inv)

    def K(self, v: Tensor) -> Tensor:
        """
        Compute the kinetic energy K(v) = 1/2 * v^T v

        Parameters
        ----------
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        kinetic energy : Tensor (b,)
        """
        return 1 / 2 * torch.einsum("bi,bi->b", v, v)  # v^T @ v

    def H(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(z,v) = U(z) + K(v)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        Tensor (b,)
        """
        return self.U(z) + self.K(v)

    def leapfrog_step(self, z: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform a single leapfrog step assuming the Hamiltonian is separable and K(v) = 1/2 * v^T v.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.
        """
        v_half = v - self.gamma / 2 * self.grad_U(z)
        z_new = z + self.gamma * v_half
        v_new = v_half - self.gamma / 2 * self.grad_U(z_new)
        return z_new, v_new

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
        if self.bounds is not None:
            out_of_bounds = z_norm > self.bounds
            alpha[out_of_bounds] = 0
        return alpha

    def leapfrog(
        self, z: Tensor, v: Tensor, return_traj: bool = False
    ) -> tuple[Tensor, Tensor]:
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
        for k in range(self.l):
            z_new, v_new = self.leapfrog_step(z_new, v_new)
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
        v = torch.einsum("bij,bi->bj", mat_sqrt(g), v) * self.std_0
        return v

    @torch.no_grad()
    def sample(
        self, z_0: Tensor, return_traj=False, progress=False, return_acceptance=False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, it returns the trajectory of the samples aswell as the acceptance rate.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,N_run,d) , float)
            The trajectory of the samples (the initial sample is the first element) and the acceptance rate.
        or
        (Tensor (b,d), float)
            The new samples and the acceptance rate.
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


# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# @TODO : Finish to adapt the other samplers to the new interface
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================


class MMALA(Sampler):
    """
    Manifold Metropolis-Adjusted Langevin Algorithm Sampler from

    `Riemann manifold Langevin and Hamiltonian Monte Carlo methods` by Girolami and Calderhead 2011.

    @TODO : I am pretty sure the implementation is plain wrong

    Parameters
    ----------
    cometric : CoMetric
        The cometric to use.
    l : int
        The number of integration steps between each proposal.
    gamma : float
        The step size.
    N_run : int
        The number of proposal steps to run.
    bounds : float
        The bounds of the manifold.
    pbar : bool
        Whether to display a progress bar or not.
    skip_acceptance : bool
        If True, the acceptance step is skipped. This can be used when differentiabily is needed.
    """

    def __init__(
        self,
        cometric: CoMetric,
        l: int,
        gamma: float,
        N_run: int,
        bounds: float,
        pbar: bool = False,
        skip_acceptance: bool = False,
    ) -> None:
        super().__init__(pbar)
        self.cometric = cometric
        self.l = l
        self.gamma = gamma
        self.N_run = N_run
        self.bounds = bounds
        self.skip_acceptance = skip_acceptance

        self._grad_U = torch.func.jacrev(self.U)
        self.grad_U = lambda z: self._grad_U(z).sum(1)

    def p_target(self, z: Tensor) -> Tensor:
        p = self.cometric(z).det().sqrt()
        return p

    def U(self, z: Tensor) -> Tensor:
        return -torch.log(self.p_target(z))

    def K(self, v: Tensor) -> Tensor:
        g_inv = self.cometric(v)  # This is weird, no position involved ?
        det_g = 1 / g_inv.det()
        velocity = torch.einsum("bj,bij,bi->b", v, g_inv, v)
        return 0.5 * velocity + 0.5 * torch.log(det_g)

    def H(self, z: Tensor) -> Tensor:
        return self.U(z) + self.K(z)

    def sqrtmh(self, A):
        """
        Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices

        See : https://github.com/pytorch/pytorch/issues/25481#issuecomment-1109537907

        Parameters
        ----------
        A : Tensor (..., n, n)
            The matrix to compute the square root of.

        Returns
        -------
        Tensor (..., n, n)
            The square root of the matrix.
        """
        L, Q = torch.linalg.eigh(A)
        zero = torch.zeros((), device=L.device, dtype=L.dtype)
        threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
        L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
        return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

    def proposal(self, x: Tensor) -> Tensor:
        """Proposal of `Riemann manifold Langevin and Hamiltonian Monte Carlo methods` p8.
        in the case of constant curvature manifolds."""
        g_inv = self.cometric(x)
        dU = self.grad_U(x)[:, :, None]
        z = torch.randn_like(x)[:, :, None]
        z = self.gamma * torch.bmm(self.sqrtmh(g_inv), z).squeeze(2)
        x_new = x + z + self.gamma**2 / 2 * torch.bmm(g_inv, dU).squeeze(2)
        return x_new

    def proposal_rate(self, z: Tensor, z_new: Tensor) -> Tensor:
        alpha = torch.exp(-self.H(z_new) + self.H(z))
        return torch.min(torch.ones_like(alpha), alpha)

    def get_alpha(self, z: Tensor, z_new: Tensor) -> Tensor:
        """Compute the proposal rates. If the new sample is out of bounds, the proposal rate is 0."""
        alpha = self.proposal_rate(z, z_new)
        z_norm = torch.linalg.norm(z_new, dim=-1)
        out_of_bounds = z_norm > self.bounds
        alpha[out_of_bounds] = 0
        return alpha

    def run(self, z: Tensor) -> Tensor:
        z_new = z.clone()
        for k in range(self.l):
            z_new = self.proposal(z_new)
        return z_new

    def sample(self, z_0: Tensor, return_traj=False) -> Tensor:
        z = z_0.clone()

        if return_traj:
            traj = [z.clone()]

        for k in range(self.N_run):
            z_l = self.run(z)

            if not self.skip_acceptance:
                alpha = self.get_alpha(z, z_l)
                u = torch.rand_like(alpha)
                mask = alpha >= u
                z = torch.where(mask[:, None], z_l, z)
            else:
                z = z_l

            if return_traj:
                traj.append(z.clone())

        if return_traj:
            return torch.stack(traj, dim=1)
        else:
            return z


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

        no_batch_U = lambda x: self.U(x.unsqueeze(0)).squeeze(0)
        self._grad_U = torch.vmap(torch.func.jacrev(no_batch_U))
        self.grad_U = lambda z: self._grad_U(z).squeeze(1)

        no_batch_H = lambda x, y: self.H(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
        self._dH_dz = torch.vmap(torch.func.jacrev(no_batch_H, argnums=0))
        self._dH_dv = torch.vmap(torch.func.jacrev(no_batch_H, argnums=1))
        self.dH_dz = lambda z, v: self._dH_dz(z, v).squeeze(1)
        self.dH_dv = lambda z, v: self._dH_dv(z, v).squeeze(1)

        self.log2pi = torch.log(torch.tensor(2 * 3.1415927410125732))

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
        return -0.5 * self.cometric.inv_logdet(z)

    def K(self, v: Tensor, q: Tensor) -> Tensor:
        """
        Compute the kinetic energy K(v) = - N(v ;0, g(z))
        ie K(v) = 1/2 * v^T g_inv(z) v - 1/2 * log(det(g_inv(z)))

        Parameters
        ----------
        v : Tensor (b,d)
            The velocity.
        z : Tensor (b,d)
            The position.

        Returns
        -------
        kinetic energy : Tensor (b,)
        """
        logdet_ginv = self.cometric.inv_logdet(q)
        velocity = self.cometric.cometric(q, v)
        return 0.5 * velocity - 0.5 * logdet_ginv + 0.5 * v.shape[1] * self.log2pi

    def H(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(z,v) = U(z) + K(v)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        Tensor (b,)
        """
        return self.U(z) + self.K(v, z)

    def get_v_half(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the velocity.
        v_half = v - gamma/2 * dH_dz(z, v_half)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        v_half : Tensor (b,d)
            The half step velocity.
        """
        v_half = v.clone()
        for k in range(self.N_fx):
            v_half_ = v_half - self.gamma * self.dH_dz(z, v_half) / 2
            if (v_half_ - v_half).abs().max() < self.threshold_fx:
                v_half = v_half_
                break
            v_half = v_half_
        return v_half

    def get_z_new(self, z: Tensor, v_half: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the position.
        z_new = z + gamma/2 * ( dH_dv(z, v_half) + dH_dv(z_new,v_half) )

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v_half : Tensor (b,d)
            The half step velocity.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        """
        z_new = z.clone()
        for k in range(self.N_fx):
            z_new_ = (
                z_new + self.gamma * (self.dH_dv(z, v_half) + self.dH_dv(z_new, v_half)) / 2
            )
            if (z_new_ - z_new).abs().max() < self.threshold_fx:
                z_new = z_new_
                break
            z_new = z_new_
        return z_new

    def leapfrog_step(self, z: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform a single leapfrog step.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.
        """
        v_half = self.get_v_half(z, v)
        z_new = self.get_z_new(z, v_half)
        v_new = v_half - self.gamma * self.dH_dz(z_new, v_half) / 2
        return z_new, v_new

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
        for k in range(self.l):
            z_new, v_new = self.leapfrog_step(z_new, v_new)
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
        self, z_0: Tensor, return_traj=False, progress=False, return_acceptance=False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, it returns the trajectory of the samples aswell as the acceptance rate.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,N_run,d) , float)
            The trajectory of the samples (the initial sample is the first element) and the acceptance rate.
        or
        (Tensor (b,d), float)
            The new samples and the acceptance rate.
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


class Hamiltonian(torch.nn.Module):
    """
    Hamiltonian function for Riemannian Hamiltonian Monte Carlo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(z,v)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        Tensor (b,)
        """
        raise NotImplementedError(
            "The Hamiltonian function must be implemented by inheriting this class."
        )


class ExplicitLeapfrogIntegrator(torch.nn.Module):
    """
    Explicit leapfrog integrator for Riemannian Hamiltonian Monte Carlo.

    Parameters:
    ----------
    H : Hamiltonian
        The Hamiltonian function H(z, v) that takes position q and momentum p and returns the energy.
    gamma : float
        The step size for the leapfrog integrator.
    omega : float
        The binding parameter for the leapfrog integrator.
    """

    def __init__(self, H: Hamiltonian, gamma: float, omega: float):
        super().__init__()
        self.H_base = H
        self.gamma = gamma
        self.omega = omega

        c = torch.Tensor([2 * self.omega * self.gamma]).cos()
        s = torch.Tensor([2 * self.omega * self.gamma]).sin()
        self.register_buffer("c", c, persistent=False)
        self.register_buffer("s", s, persistent=False)

        # Compute per-sample gradients to avoid materializing a full (B, B, D) Jacobian.
        no_batch_H = lambda z, v: self.H_base(z.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
        self._dH_dz = torch.vmap(torch.func.jacrev(no_batch_H, argnums=0))
        self._dH_dv = torch.vmap(torch.func.jacrev(no_batch_H, argnums=1))
        self.dH_dz = lambda z, v: self._dH_dz(z, v)
        self.dH_dv = lambda z, v: self._dH_dv(z, v)

    def binding(self, z_0: Tensor, v_0: Tensor, z_1: Tensor, v_1: Tensor) -> Tensor:
        """
        Compute the binding energy between two states.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The position of the first state.
        v_0 : Tensor (b,d)
            The velocity of the first state.
        z_1 : Tensor (b,d)
            The position of the second state.
        v_1 : Tensor (b,d)
            The velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The binding energy.
        """
        h = torch.linalg.vector_norm(z_1 - z_0, dim=-1) ** 2 / 2
        h += torch.linalg.vector_norm(v_1 - v_0, dim=-1) ** 2 / 2
        return h

    def H(self, z_0: Tensor, v_0: Tensor, z_1: Tensor, v_1: Tensor) -> Tensor:
        """
        Compute the augmented Hamiltonian H(z_0, v_0, z_1, v_1) = H(z_0, v_0) + H(z_1, v_1) + omega * binding(z_0, v_0, z_1, v_1)

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The position of the first state.
        v_0 : Tensor (b,d)
            The velocity of the first state.
        z_1 : Tensor (b,d)
            The position of the second state.
        v_1 : Tensor (b,d)
            The velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The augmented Hamiltonian.
        """
        H_0 = self.H_base(z_0, v_0)
        H_1 = self.H_base(z_1, v_1)
        H = H_0 + H_1 + self.omega * self.binding(z_0, v_0, z_1, v_1)
        return H

    def leapfrog_step(
        self, z_0: Tensor, v_0: Tensor, z_1: Tensor, v_1: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Leapfrog step for the augmented Hamiltonian.
        Pseudo code in `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`
        by Cobb et Baydin et al (2019).

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The position of the first state.
        v_0 : Tensor (b,d)
            The velocity of the first state.
        z_1 : Tensor (b,d)
            The position of the second state.
        v_1 : Tensor (b,d)
            The velocity of the second state.

        Returns
        -------
        z_0_new : Tensor (b,d)
            The new position of the first state.
        v_0_new : Tensor (b,d)
            The new velocity of the first state.
        z_1_new : Tensor (b,d)
            The new position of the second state.
        v_1_new : Tensor (b,d)
            The new velocity of the second state.
        """
        c = self.c.to(z_0.device).to(z_0.dtype)
        s = self.s.to(z_0.device).to(z_0.dtype)

        v_0_new = v_0 - self.gamma / 2 * self.dH_dz(z_0, v_1)
        z_1_new = z_1 + self.gamma / 2 * self.dH_dv(z_0, v_1)
        v_1_new = v_1 - self.gamma / 2 * self.dH_dz(z_1_new, v_0)
        z_0_new = z_0 + self.gamma / 2 * self.dH_dv(z_1_new, v_0)

        z_0_new = (z_0_new + z_1_new + c * (z_0_new - z_1_new) + s * (v_0_new - v_1_new)) / 2
        v_0_new = (v_0_new + v_1_new - s * (z_0_new - z_1_new) + c * (v_0_new - v_1_new)) / 2
        z_1_new = (z_0_new + z_1_new - c * (z_0_new - z_1_new) - s * (v_0_new - v_1_new)) / 2
        v_1_new = (v_0_new + v_1_new + s * (z_0_new - z_1_new) - c * (v_0_new - v_1_new)) / 2

        v_1_new = v_1_new - self.gamma / 2 * self.dH_dz(z_1_new, v_0_new)
        z_0_new = z_0_new + self.gamma / 2 * self.dH_dv(z_1_new, v_0_new)
        v_0_new = v_0_new - self.gamma / 2 * self.dH_dz(z_0_new, v_1_new)
        z_1_new = z_1_new + self.gamma / 2 * self.dH_dv(z_0_new, v_1_new)

        return z_0_new, v_0_new, z_1_new, v_1_new

    @torch.no_grad()
    def forward(self, z_0: Tensor, v_0: Tensor, L: int, return_traj: bool = False):
        z_1, v_1 = z_0.clone(), v_0.clone()
        if return_traj:
            traj_z = [z_0.clone().detach()]
            traj_v = [v_0.clone().detach()]

        for k in tqdm(range(L)):
            z_0, v_0, z_1, v_1 = self.leapfrog_step(z_0, v_0, z_1, v_1)

            if return_traj:
                # Keep graph only on the final point.
                if k == L - 1:
                    traj_z.append(z_0.clone())
                    traj_v.append(v_0.clone())
                else:
                    traj_z.append(z_0.clone().detach())
                    traj_v.append(v_0.clone().detach())

        if return_traj:
            traj_z = torch.stack(traj_z, dim=1)
            traj_v = torch.stack(traj_v, dim=1)
            return traj_z, traj_v
        return z_0, v_0


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

        self._dH_dz_ = torch.func.jacrev(self.H_base, argnums=0)
        self._dH_dv = torch.func.jacrev(self.H_base, argnums=1)
        self.dH_dz = lambda z, v: self._dH_dz_(z, v).sum(1)
        self.dH_dv = lambda z, v: self._dH_dv(z, v).sum(1)

        self.log2pi = torch.log(torch.tensor(2 * 3.1415927410125732))

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
        return -0.5 * self.cometric.inv_logdet(z)

    def K(self, v: Tensor, z: Tensor) -> Tensor:
        """
        Compute the kinetic energy K(v) = - N(v ;0, g(z))
        ie K(v) = 1/2 * v^T g_inv(z) v - 1/2 * log(det(g_inv(z)))

        Parameters
        ----------
        v : Tensor (b,d)
            The velocity.
        z : Tensor (b,d)
            The position.

        Returns
        -------
        kinetic energy : Tensor (b,)
        """
        logdet_ginv = self.cometric.inv_logdet(z)
        velocity = self.cometric.cometric(z, v)
        return 0.5 * velocity - 0.5 * logdet_ginv + 0.5 * v.shape[1] * self.log2pi

    def H_base(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(z,v) = U(z) + K(v)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        Tensor (b,)
        """
        return self.U(z) + self.K(v, z)

    def binding(self, z_0: Tensor, v_0: Tensor, z_1: Tensor, v_1: Tensor) -> Tensor:
        """
        Compute the binding energy between two states.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The position of the first state.
        v_0 : Tensor (b,d)
            The velocity of the first state.
        z_1 : Tensor (b,d)
            The position of the second state.
        v_1 : Tensor (b,d)
            The velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The binding energy.
        """
        h = torch.linalg.vector_norm(z_1 - z_0, dim=-1) ** 2 / 2
        h += torch.linalg.vector_norm(v_1 - v_0, dim=-1) ** 2 / 2
        return h

    def H(self, z_0: Tensor, v_0: Tensor, z_1: Tensor, v_1: Tensor) -> Tensor:
        """
        Compute the augmented Hamiltonian H(z_0, v_0, z_1, v_1) = H(z_0, v_0) + H(z_1, v_1) + omega * binding(z_0, v_0, z_1, v_1)

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The position of the first state.
        v_0 : Tensor (b,d)
            The velocity of the first state.
        z_1 : Tensor (b,d)
            The position of the second state.
        v_1 : Tensor (b,d)
            The velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The augmented Hamiltonian.
        """
        H_0 = self.H_base(z_0, v_0)
        H_1 = self.H_base(z_1, v_1)
        H = H_0 + H_1 + self.omega * self.binding(z_0, v_0, z_1, v_1)
        return H

    def leapfrog_step(
        self, z_0: Tensor, v_0: Tensor, z_1: Tensor, v_1: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Leapfrog step for the augmented Hamiltonian.
        Pseudo code in `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`
        by Cobb et Baydin et al (2019).

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The position of the first state.
        v_0 : Tensor (b,d)
            The velocity of the first state.
        z_1 : Tensor (b,d)
            The position of the second state.
        v_1 : Tensor (b,d)
            The velocity of the second state.

        Returns
        -------
        z_0_new : Tensor (b,d)
            The new position of the first state.
        v_0_new : Tensor (b,d)
            The new velocity of the first state.
        z_1_new : Tensor (b,d)
            The new position of the second state.
        v_1_new : Tensor (b,d)
            The new velocity of the second state.
        """
        v_0_new = v_0 - self.gamma / 2 * self.dH_dz(z_0, v_1)
        z_1_new = z_1 + self.gamma / 2 * self.dH_dv(z_0, v_1)
        v_1_new = v_1 - self.gamma / 2 * self.dH_dz(z_1_new, v_0)
        z_0_new = z_0 + self.gamma / 2 * self.dH_dv(z_1_new, v_0)

        z_0_new = (
            z_0_new + z_1_new + self.c * (z_0_new - z_1_new) + self.s * (v_0_new - v_1_new)
        ) / 2
        v_0_new = (
            v_0_new + v_1_new - self.s * (z_0_new - z_1_new) + self.c * (v_0_new - v_1_new)
        ) / 2
        z_1_new = (
            z_0_new + z_1_new - self.c * (z_0_new - z_1_new) - self.s * (v_0_new - v_1_new)
        ) / 2
        v_1_new = (
            v_0_new + v_1_new + self.s * (z_0_new - z_1_new) - self.c * (v_0_new - v_1_new)
        ) / 2

        v_1_new = v_1_new - self.gamma / 2 * self.dH_dz(z_1_new, v_0_new)
        z_0_new = z_0_new + self.gamma / 2 * self.dH_dv(z_1_new, v_0_new)
        v_0_new = v_0_new - self.gamma / 2 * self.dH_dz(z_0_new, v_1_new)
        z_1_new = z_1_new + self.gamma / 2 * self.dH_dv(z_0_new, v_1_new)

        return z_0_new, v_0_new, z_1_new, v_1_new

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
        z_l_1: Tensor,
        v_l_1: Tensor,
        z_0: Tensor,
        v0: Tensor,
        z_1: Tensor,
        v1: Tensor,
    ) -> Tensor:
        """
        Compute the proposal rates based on the value of the Hamiltonian.

        Parameters
        ----------
        z_l_0 : Tensor (b,d)
            The new position of the first state.
        v_l_0 : Tensor (b,d)
            The new velocity of the first state.
        z_l_1 : Tensor (b,d)
            The new position of the second state.
        v_l_1 : Tensor (b,d)
            The new velocity of the second state.
        z_0 : Tensor (b,d)
            The initial position of the first state.
        v0 : Tensor (b,d)
            The initial velocity of the first state.
        z_1 : Tensor (b,d)
            The initial position of the second state.
        v1 : Tensor (b,d)
            The initial velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        H_new = self.H_base(z_l_0, v_l_0)
        H_old = self.H_base(z_0, v0)
        alpha = torch.exp(-H_new + H_old)
        return torch.min(torch.ones_like(alpha), alpha)

    def get_alpha(
        self,
        z_l_0: Tensor,
        v_l_0: Tensor,
        z_l_1: Tensor,
        v_l_1: Tensor,
        z_0: Tensor,
        v0: Tensor,
        z_1: Tensor,
        v1: Tensor,
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
        v_l_1 : Tensor (b,d)
            The new velocity of the second state.
        z_0 : Tensor (b,d)
            The initial position of the first state.
        v0 : Tensor (b,d)
            The initial velocity of the first state.
        z_1 : Tensor (b,d)
            The initial position of the second state.
        v1 : Tensor (b,d)
            The initial velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = self.proposal_rate(z_l_0, v_l_0, z_l_1, v_l_1, z_0, v0, z_1, v1)
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
        for k in range(self.l):
            z_l_0, v_l_0, z_l_1, v_l_1 = self.leapfrog_step(z_l_0, v_l_0, z_l_1, v_l_1)
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

    def sample(self, z_0: Tensor, return_traj=False, progress=False, return_acceptance=False):
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, it returns the trajectory of the samples aswell as the acceptance rate.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,N_run,d) , float)
            The trajectory of the samples (the initial sample is the first element) and the acceptance rate.
        or
        (Tensor (b,d), float)
            The new samples and the acceptance rate.
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
                alpha = self.get_alpha(z_l_0, v_l_0, z_l_1, v_l_1, z_0, v_0, z_1, v_1)

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


class ExplicitRFHMCSampler(ExplicitRHMCSampler):
    """
    Explicit Riemannian Hamiltonian Monte Carlo sampler with a pdf defined on a manifold.
    It uses a Randers metric to propose new samples hence providing
    time consistent trajectories.
    It uses the augmented leapfrog integrator to propose new samples from the target distribution.
    It uses a tempering scheme on the momentum.
    Here the target distribution is defined by the volume element of the cometric.
    But this class is easily heritable to define other target distributions. Just redefine
    the p_target method.

    `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`
    by Cobb et Baydin et al (2019).

    Parameters
    ----------
    randers : RandersMetrics
        The Randers metric that defines the target distribution.
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
    """

    def __init__(
        self,
        randers: RandersMetrics,
        l: int,
        gamma: float,
        omega: float,
        N_run: int,
        bounds: float,
        std_0: float = 1.0,
        beta_0: float = 1,
        pbar: bool = False,
        skip_acceptance: bool = False,
    ):
        super().__init__(
            randers.base_cometric,
            l,
            gamma,
            omega,
            N_run,
            bounds,
            std_0,
            beta_0,
            pbar,
            skip_acceptance,
        )
        self.randers = randers

    # # Override the kinetic energy function to use the Randers cometric
    # def K(self, v: Tensor, z: Tensor) -> Tensor:
    #     """
    #     Compute the kinetic energy K(v) = - N(v ;0, g(z))
    #     ie K(v) = 1/2 * v^T g_inv(z) v - 1/2 * log(det(g_inv(z)))
    #     where g is fundamental tensor of the Randers metric.

    #     Parameters
    #     ----------
    #     v : Tensor (b,d)
    #         The velocity.
    #     z : Tensor (b,d)
    #         The position.

    #     Returns
    #     -------
    #     kinetic energy : Tensor (b,)
    #     """
    #     g_fund = self.randers.fundamental_tensor(z, v)
    #     g_fund_inv = torch.linalg.inv(g_fund)
    #     logdet_ginv = torch.logdet(g_fund_inv)
    #     velocity = torch.einsum("bj,bij,bi->b", v, g_fund_inv, v)
    #     return 0.5 * velocity - 0.5 * logdet_ginv + 0.5 * v.shape[1] * self.log2pi

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample the momentum from N(0, g(z,omega(z)))

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
        omega = self.randers.omega(z)
        dot_prod = torch.einsum("bi,bi->b", v, omega)
        v = torch.sign(dot_prod)[:, None] * v
        v = -torch.einsum("bij,bi->bj", mat_sqrt(g), v) * self.std_0
        return v
