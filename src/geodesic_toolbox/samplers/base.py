"""
Base class for batch MCMC samplers and helpers to isolate batch-wide
integration failures.
"""

import torch
from torch import Tensor, nn
from torch.linalg import LinAlgError as _LinAlgError

from ..integrators import HamiltonianIntegrator, Integrator


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
