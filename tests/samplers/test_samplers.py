import pytest
import torch

from geodesic_toolbox import samplers
from geodesic_toolbox.cometric import IdentityCoMetric


def test_integrate_isolating_failures_retries_nonfinite_batch_outputs():
    class NonFiniteIntegrator:
        def __call__(self, x, L, dirs=None):
            if x.shape[0] == 1:
                return x + L, torch.ones(1, dtype=x.dtype)
            result = x + 1
            result[1, 0] = torch.nan
            return result, torch.tensor([0.5, torch.inf], dtype=x.dtype)

    initial = torch.zeros(2, 2)
    result, log_det, valid = samplers.integrate_isolating_failures(
        NonFiniteIntegrator(), initial, L=2
    )

    assert torch.equal(valid, torch.tensor([True, True]))
    assert torch.equal(result, initial + 2)
    assert torch.equal(log_det, torch.ones(2))


def test_integrate_isolating_failures_restores_nonfinite_single_chain():
    class NonFiniteIntegrator:
        def __call__(self, x, L, dirs=None):
            if x.shape[0] > 1:
                result = x + 1
                result[1, 0] = torch.nan
                return result, torch.tensor([0.5, torch.inf], dtype=x.dtype)
            if x[0, 0] == 1:
                return torch.full_like(x, torch.nan), torch.tensor([torch.inf])
            return x + L, torch.ones(1, dtype=x.dtype)

    initial = torch.zeros(2, 2)
    initial[1, 0] = 1
    result, log_det, valid = samplers.integrate_isolating_failures(
        NonFiniteIntegrator(), initial, L=2
    )

    assert torch.equal(valid, torch.tensor([True, False]))
    assert torch.equal(result[0], torch.full((2,), 2.0))
    assert torch.equal(result[1], initial[1])
    assert torch.equal(log_det, torch.tensor([1.0, 0.0]))


def test_integrate_hamiltonian_isolating_failures_retries_and_isolates_errors():
    class PartiallyFailingIntegrator:
        def forward(self, q, p, L, dirs=None):
            if q.shape[0] > 1:
                raise torch.linalg.LinAlgError("batch failure")
            if q[0, 0] == 1:
                raise torch.linalg.LinAlgError("single-chain failure")
            q_result = q + 1
            p_result = p + 1
            log_det = torch.tensor([torch.inf], dtype=q.dtype)
            if q[0, 0] == 0:
                log_det.zero_()
            return q_result, p_result, log_det

    q_initial = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    p_initial = torch.zeros_like(q_initial)
    q_result, p_result, log_det, valid = samplers.integrate_hamiltonian_isolating_failures(
        PartiallyFailingIntegrator(), q_initial, p_initial, L=2
    )

    assert torch.equal(valid, torch.tensor([True, False]))
    assert torch.equal(q_result[0], torch.ones(2))
    assert torch.equal(p_result[0], torch.ones(2))
    assert torch.equal(q_result[1], q_initial[1])
    assert torch.equal(p_result[1], p_initial[1])
    assert torch.equal(log_det, torch.zeros(2))


def test_hmc_returns_post_warmup_samples_and_diagnostics():
    sampler = samplers.HamiltonianMonteCarlo(
        log_target=lambda x: -0.5 * (x**2).sum(dim=-1),
        num_integration_steps=2,
        step_size=0.05,
    )

    samples, diagnostics = sampler.sample(torch.zeros(3, 2), num_warmup=4, num_samples=5)

    assert samples.shape == (3, 5, 2)
    assert diagnostics["acceptance_rate"] >= 0
    assert diagnostics["acceptance_rate"] <= 1
    assert diagnostics["acceptance_rate_by_chain"].shape == (3,)
    assert diagnostics["acceptance_probability"].shape == (3, 5)
    assert diagnostics["warmup_acceptance_rate"] is not None


def test_hmc_rejects_invalid_sampling_counts():
    sampler = samplers.HamiltonianMonteCarlo(
        log_target=lambda x: -0.5 * (x**2).sum(dim=-1),
        num_integration_steps=1,
        step_size=0.05,
    )

    with pytest.raises(ValueError, match="num_warmup"):
        sampler.sample(torch.zeros(1, 2), num_warmup=-1, num_samples=1)
    with pytest.raises(ValueError, match="num_samples"):
        sampler.sample(torch.zeros(1, 2), num_warmup=0, num_samples=0)


def test_rhmc_returns_post_warmup_samples_and_diagnostics():
    sampler = samplers.ImplicitMidpointRHMCSampler(
        cometric=IdentityCoMetric(),
        num_integration_steps=2,
        fixed_point_iterations=2,
        step_size=0.01,
        log_target=lambda x: -0.5 * (x**2).sum(dim=-1),
    )

    samples, diagnostics = sampler.sample(torch.zeros(2, 2), num_warmup=1, num_samples=3)

    assert samples.shape == (2, 3, 2)
    assert diagnostics["acceptance_probability"].shape == (2, 3)
    assert diagnostics["valid"].shape == (2, 3)
    assert 0 <= diagnostics["acceptance_rate"] <= 1


def test_rhmc_accepts_implicit_leapfrog_integrator():
    sampler = samplers.ImplicitLeapfrogRHMCSampler(
        cometric=IdentityCoMetric(),
        num_integration_steps=2,
        fixed_point_iterations=2,
        step_size=0.01,
    )

    samples, _ = sampler.sample(torch.zeros(1, 2), num_warmup=0, num_samples=1)

    assert samples.shape == (1, 1, 2)


def test_explicit_rhmc_refreshes_auxiliary_state_each_transition():
    sampler = samplers.ExplicitRHMCSampler(
        cometric=IdentityCoMetric(),
        num_integration_steps=2,
        fixed_point_iterations=2,
        step_size=0.01,
        omega=1.0,
    )

    samples, diagnostics = sampler.sample(torch.zeros(2, 2), num_warmup=1, num_samples=2)

    assert samples.shape == (2, 2, 2)
    assert diagnostics["valid"].shape == (2, 2)


def test_reduced_flip_midpoint_rhmc_reports_direction_flips():
    sampler = samplers.ImplicitMidpointRHMCRFSampler(
        cometric=IdentityCoMetric(),
        num_integration_steps=2,
        fixed_point_iterations=2,
        step_size=0.01,
    )

    samples, diagnostics = sampler.sample(torch.zeros(2, 2), num_warmup=1, num_samples=2)

    assert samples.shape == (2, 2, 2)
    assert 0 <= diagnostics["flip_rate"] <= 1
    assert diagnostics["flip_rate_by_chain"].shape == (2,)


@pytest.mark.parametrize(
    "sampler_factory",
    [
        lambda: samplers.ImplicitLeapfrogRHMCRFSampler(IdentityCoMetric(), 2, 2, 0.01),
        lambda: samplers.ExplicitRHMCRFSampler(IdentityCoMetric(), 2, 0.01, omega=1.0),
    ],
)
def test_reduced_flip_variants_cover_all_integrators(sampler_factory):
    sampler = sampler_factory()
    samples, diagnostics = sampler.sample(torch.zeros(1, 2), num_warmup=0, num_samples=1)

    assert samples.shape == (1, 1, 2)
    assert 0 <= diagnostics["flip_rate"] <= 1


def test_reduced_flip_step_initializes_directions_lazily():
    sampler = samplers.ImplicitMidpointRHMCRFSampler(
        cometric=IdentityCoMetric(),
        num_integration_steps=2,
        fixed_point_iterations=2,
        step_size=0.01,
    )

    next_state, diagnostics = sampler.step(torch.zeros(2, 2))

    assert next_state.shape == (2, 2)
    assert diagnostics["flipped"].shape == (2,)
