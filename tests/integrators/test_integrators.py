import math

import pytest
import torch

from geodesic_toolbox import integrators


def _patch_tqdm(monkeypatch, fake_tqdm):
    # Each integrator submodule binds tqdm in its own namespace.
    for module in (integrators.euler, integrators.implicit_midpoint, integrators.leapfrog):
        monkeypatch.setattr(module, "tqdm", fake_tqdm)


class QuadraticHamiltonian(integrators.Hamiltonian):
    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return (q.square() + p.square()).sum(dim=-1) / 2


def _integrator_cases(pbar: bool):
    pbar_kwargs = {"pbar": True} if pbar else {}
    x = torch.zeros(1, 2)
    q = torch.zeros(1, 1)
    p = torch.zeros(1, 1)
    f = lambda state: torch.zeros_like(state)
    df = lambda state: torch.zeros(
        state.shape[0], state.shape[1], state.shape[1], device=state.device
    )
    hamiltonian = QuadraticHamiltonian()

    return [
        (
            integrators.EulerIntegrator(f, 0.1, **pbar_kwargs),
            lambda method: method(x, 2),
        ),
        (
            integrators.HamiltonianEulerIntegrator(hamiltonian, 0.1, **pbar_kwargs),
            lambda method: method(q, p, 2),
        ),
        (
            integrators.ImplicitMidpointIntegrator(f, df, 0.1, N_fx=2, **pbar_kwargs),
            lambda method: method(x, 2),
        ),
        (
            integrators.SeparableLeapfrogIntegrator(hamiltonian, 0.1, **pbar_kwargs),
            lambda method: method(q, p, 2),
        ),
        (
            integrators.ImplicitLeapfrogIntegrator(
                hamiltonian, 0.1, n_fix_pts=2, **pbar_kwargs
            ),
            lambda method: method(q, p, 2),
        ),
        (
            integrators.ExplicitLeapfrogIntegrator(hamiltonian, 0.1, omega=0.1, **pbar_kwargs),
            lambda method: method(q, p, 2),
        ),
        (
            integrators.HamiltonianImplicitMidpointIntegrator(
                hamiltonian, 0.1, N_fx=2, **pbar_kwargs
            ),
            lambda method: method(q, p, 2),
        ),
    ]


def test_public_names_exported_from_package():
    expected = [
        "BatchedFunction",
        "EulerIntegrator",
        "ExplicitLeapfrogIntegrator",
        "Hamiltonian",
        "HamiltonianEulerIntegrator",
        "HamiltonianImplicitMidpointIntegrator",
        "HamiltonianIntegrator",
        "HamiltonianToBatchedFunction",
        "ImplicitLeapfrogIntegrator",
        "ImplicitMidpointIntegrator",
        "Integrator",
        "SeparableLeapfrogIntegrator",
    ]
    for name in expected:
        assert hasattr(integrators, name), f"geodesic_toolbox.integrators.{name} is missing"


def test_integrators_disable_progress_bars_by_default(monkeypatch):
    progress_calls = []

    def unexpected_progress_bar(*args, **kwargs):
        progress_calls.append(kwargs)
        raise AssertionError("tqdm should not be called when pbar is False")

    _patch_tqdm(monkeypatch, unexpected_progress_bar)

    for method, run in _integrator_cases(pbar=False):
        assert method.pbar is False
        run(method)

    assert not progress_calls


def test_integrators_enable_progress_bars_when_requested(monkeypatch):
    progress_calls = []

    def fake_progress_bar(iterable, **kwargs):
        progress_calls.append(kwargs["desc"])
        return iterable

    _patch_tqdm(monkeypatch, fake_progress_bar)

    for method, run in _integrator_cases(pbar=True):
        run(method)

    assert progress_calls == [
        "Euler integration",
        "Euler integration",
        "Implicit midpoint integration",
        "Leapfrog integration",
        "Leapfrog integration",
        "Leapfrog steps",
        "Implicit midpoint integration",
    ]


##############################
# Behavioural tests: error paths, dynamics, reversibility, trajectories
##############################


def _quadratic_state():
    q0 = torch.tensor([[0.6, -0.4]])
    p0 = torch.tensor([[0.2, 0.3]])
    return q0, p0


def _hamiltonian_integrators():
    H = QuadraticHamiltonian()
    return {
        "separable": integrators.SeparableLeapfrogIntegrator(H, 0.1),
        "implicit": integrators.ImplicitLeapfrogIntegrator(H, 0.1, n_fix_pts=5),
        "explicit": integrators.ExplicitLeapfrogIntegrator(H, 0.1, omega=1.0),
        "midpoint": integrators.HamiltonianImplicitMidpointIntegrator(H, 0.1, N_fx=5),
        "euler": integrators.HamiltonianEulerIntegrator(H, 0.1),
    }


@pytest.mark.parametrize("L", [0, 1])
def test_invalid_num_states_raises(L):
    integrator = integrators.EulerIntegrator(lambda x: x, 0.1)
    with pytest.raises(ValueError, match="L must be greater than 1"):
        integrator(torch.zeros(1, 2), L)


def test_abstract_base_classes_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        integrators.Integrator(lambda x: x)(torch.zeros(1, 2), 2)
    with pytest.raises(NotImplementedError):
        integrators.Hamiltonian()(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(NotImplementedError):
        integrators.HamiltonianIntegrator(QuadraticHamiltonian())(
            torch.zeros(1, 2), torch.zeros(1, 2), 2
        )


def test_implicit_midpoint_rejects_unknown_options():
    f = lambda x: x
    df = lambda x: torch.eye(2).repeat(x.shape[0], 1, 1)
    with pytest.raises(ValueError, match="Unknown jacobian method"):
        integrators.ImplicitMidpointIntegrator(f, df, 0.1, N_fx=2, jacobian="bogus")
    with pytest.raises(ValueError, match="Unknown method"):
        integrators.ImplicitMidpointIntegrator(f, df, 0.1, N_fx=2, method="bogus")


def test_hamiltonian_gradients_quadratic():
    H = QuadraticHamiltonian()
    q0, p0 = _quadratic_state()
    assert torch.allclose(H.dH_dq(q0, p0), q0, atol=1e-6)
    assert torch.allclose(H.dH_dp(q0, p0), p0, atol=1e-6)


def test_hamiltonian_to_batched_function_and_jacobian():
    H = QuadraticHamiltonian()
    f = integrators.HamiltonianToBatchedFunction(H)
    q0, p0 = _quadratic_state()
    x = torch.cat([q0, p0], dim=1)

    # f(x) = [dH/dp, -dH/dq] = [p, -q] for the quadratic Hamiltonian.
    assert torch.allclose(f(x), torch.cat([p0, -q0], dim=1), atol=1e-6)

    jac = f.df(x)
    # d/d[q,p] [p, -q] = [[0, 0, 1, 0], [0, 0, 0, 1], [0, -1, 0, 0], [0, 0, -1, 0]]
    I2 = torch.eye(2)
    Z2 = torch.zeros(2, 2)
    expected = torch.cat([torch.cat([Z2, I2], dim=1), torch.cat([-I2, Z2], dim=1)], dim=0)[
        None
    ]
    assert torch.allclose(jac, expected, atol=1e-6)


@pytest.mark.parametrize("name", ["separable", "implicit", "midpoint"])
def test_symmetric_integrators_are_reversible(name):
    """Integrating L steps forward then L steps backward returns to the start.

    Only time-*symmetric* methods are reversible this way; the explicit
    two-copy scheme is symplectic but not time-reversible (symplecticity does
    not imply reversibility). Its dynamics are pinned instead by
    ``test_hamiltonian_integrators_are_second_order``.
    """
    integrator = _hamiltonian_integrators()[name]
    q0, p0 = _quadratic_state()
    dirs = torch.full((1,), -1.0)
    q1, p1, _ = integrator(q0, p0, 10)
    q2, p2, _ = integrator(q1, p1, 10, dirs=dirs)
    assert torch.allclose(q2, q0, atol=1e-6)
    assert torch.allclose(p2, p0, atol=1e-6)


def test_implicit_midpoint_is_reversible():
    A = torch.tensor([[0.2, -0.1], [0.15, 0.05]])
    f = lambda x: x @ A.T
    df = lambda x: A.repeat(x.shape[0], 1, 1)
    integrator = integrators.ImplicitMidpointIntegrator(f, df, 0.2, N_fx=3)
    x0 = torch.tensor([[0.3, -0.4]])
    x1, _ = integrator(x0, 8)
    x2, _ = integrator(x1, 8, dirs=torch.full((1,), -1.0))
    assert torch.allclose(x2, x0, atol=1e-6)


@pytest.mark.parametrize("name", ["separable", "implicit", "explicit", "midpoint"])
def test_hamiltonian_integrators_are_second_order(name):
    """Quadratic H gives the harmonic oscillator; these 2nd-order methods must
    track the exact rotation to O(gamma^2)."""
    gamma = 0.05
    L = 21  # t = 1.0
    H = QuadraticHamiltonian()
    if name == "separable":
        integrator = integrators.SeparableLeapfrogIntegrator(H, gamma)
    elif name == "implicit":
        integrator = integrators.ImplicitLeapfrogIntegrator(H, gamma, n_fix_pts=5)
    elif name == "explicit":
        integrator = integrators.ExplicitLeapfrogIntegrator(H, gamma, omega=1.0)
    else:
        integrator = integrators.HamiltonianImplicitMidpointIntegrator(H, gamma, N_fx=5)
    q0, p0 = _quadratic_state()
    q1, p1, _ = integrator(q0, p0, L)

    t = (L - 1) * gamma
    q_exact = q0 * math.cos(t) + p0 * math.sin(t)
    p_exact = p0 * math.cos(t) - q0 * math.sin(t)
    error = torch.cat([q1 - q_exact, p1 - p_exact]).abs().max().item()
    assert error < 5 * gamma**2


def test_hamiltonian_integrators_return_zero_log_det():
    q0, p0 = _quadratic_state()
    for name, integrator in _hamiltonian_integrators().items():
        if name == "euler":
            continue  # Euler is not volume-preserving
        _, _, log_det = integrator(q0, p0, 5)
        assert torch.allclose(log_det, torch.zeros(1)), f"{name} log_det not zero"


def test_euler_matches_exact_discrete_map_and_log_det():
    gamma = 0.1
    L = 5
    integrator = integrators.HamiltonianEulerIntegrator(QuadraticHamiltonian(), gamma)
    q0, p0 = _quadratic_state()
    q1, p1, log_det = integrator(q0, p0, L)

    # f([q,p]) = [p, -q], so the discrete Euler map is x <- (I + gamma J4) x.
    I2 = torch.eye(2)
    Z2 = torch.zeros(2, 2)
    J4 = torch.cat([torch.cat([Z2, I2], dim=1), torch.cat([-I2, Z2], dim=1)], dim=0)
    M = torch.eye(4) + gamma * J4
    x_expected = torch.cat([q0, p0], dim=1) @ torch.linalg.matrix_power(
        M.transpose(0, 1), L - 1
    )
    q_expected, p_expected = torch.split(x_expected, 2, dim=1)
    assert torch.allclose(q1, q_expected, atol=1e-6)
    assert torch.allclose(p1, p_expected, atol=1e-6)

    # Per step: det(I4 + gamma J4) = (1 + gamma^2)^2 (two symplectic 2x2 blocks).
    expected_log_det = (L - 1) * 2 * math.log(1 + gamma**2)
    assert torch.allclose(log_det, torch.full((1,), expected_log_det), atol=1e-6)


def test_euler_dirs_are_equivalent_to_negative_gamma():
    f = lambda x: -x
    x0 = torch.tensor([[0.5, -0.2]])
    with_dirs, _ = integrators.EulerIntegrator(f, 0.1)(x0, 4, dirs=torch.tensor([-1.0]))
    reference, _ = integrators.EulerIntegrator(f, -0.1)(x0, 4)
    assert torch.allclose(with_dirs, reference, atol=1e-7)


def test_generic_implicit_midpoint_matches_exact_implicit_step():
    A = torch.tensor([[0.2, -0.1], [0.15, 0.05]])
    f = lambda x: x @ A.T
    df = lambda x: A.repeat(x.shape[0], 1, 1)
    gamma = 0.2
    L = 4
    x0 = torch.tensor([[0.3, -0.4]])

    integrator = integrators.ImplicitMidpointIntegrator(f, df, gamma, N_fx=3)
    x1, log_det = integrator(x0, L)

    # Exact implicit midpoint for a linear field: x1 = (I - gA/2)^{-1}(I + gA/2) x0.
    x_expected_col = x0.T
    for _ in range(L - 1):
        x_expected_col = torch.linalg.solve(
            torch.eye(2) - gamma / 2 * A,
            (torch.eye(2) + gamma / 2 * A) @ x_expected_col,
        )
    assert torch.allclose(x1, x_expected_col.T, atol=1e-6)

    expected_log_det = (L - 1) * (
        torch.log(torch.det(torch.eye(2) + gamma / 2 * A))
        - torch.log(torch.det(torch.eye(2) - gamma / 2 * A))
    )
    assert torch.allclose(log_det, expected_log_det, atol=1e-6)


def test_implicit_midpoint_newton_matches_picard():
    A = torch.tensor([[0.2, -0.1], [0.15, 0.05]])
    f = lambda x: x @ A.T
    df = lambda x: A.repeat(x.shape[0], 1, 1)
    x0 = torch.tensor([[0.3, -0.4]])
    picard, _ = integrators.ImplicitMidpointIntegrator(f, df, 0.2, N_fx=3, method="picard")(
        x0, 4
    )
    newton, _ = integrators.ImplicitMidpointIntegrator(f, df, 0.2, N_fx=3, method="newton")(
        x0, 4
    )
    assert torch.allclose(picard, newton, atol=1e-6)


def test_estimate_log_det_jac_is_zero_for_zero_field():
    f = lambda x: torch.zeros_like(x)
    df = lambda x: torch.zeros(x.shape[0], x.shape[1], x.shape[1])
    integrator = integrators.ImplicitMidpointIntegrator(
        f, df, 0.2, N_fx=2, jacobian="estimate", jacobian_mc=8
    )
    _, log_det = integrator(torch.zeros(1, 2), 4)
    assert torch.allclose(log_det, torch.zeros(1))


def test_estimate_log_det_jac_is_close_to_exact():
    torch.manual_seed(0)
    A = 0.05 * torch.eye(2)
    f = lambda x: x @ A.T
    df = lambda x: A.repeat(x.shape[0], 1, 1)
    gamma = 0.2
    L = 4
    integrator = integrators.ImplicitMidpointIntegrator(
        f, df, gamma, N_fx=2, jacobian="estimate", jacobian_mc=256
    )
    _, log_det = integrator(torch.tensor([[0.3, -0.4]]), L)
    exact = (L - 1) * (
        torch.log(torch.det(torch.eye(2) + gamma / 2 * A))
        - torch.log(torch.det(torch.eye(2) - gamma / 2 * A))
    )
    assert abs(log_det.item() - exact.item()) < 0.05 * abs(exact.item())


@pytest.mark.parametrize("name", ["separable", "implicit", "explicit", "midpoint", "euler"])
def test_hamiltonian_return_traj_shapes_and_endpoints(name):
    L = 5
    integrator = _hamiltonian_integrators()[name]
    q0, p0 = _quadratic_state()
    q_traj, p_traj, _ = integrator(q0, p0, L, return_traj=True)
    assert q_traj.shape == (1, L, 2)
    assert p_traj.shape == (1, L, 2)
    assert torch.allclose(q_traj[:, 0], q0)
    assert torch.allclose(p_traj[:, 0], p0)
    q_end, p_end, _ = integrator(q0, p0, L)
    assert torch.allclose(q_traj[:, -1], q_end, atol=1e-6)
    assert torch.allclose(p_traj[:, -1], p_end, atol=1e-6)


def test_generic_return_traj_shapes_and_endpoints():
    f = lambda x: -x
    df = lambda x: -torch.eye(2).repeat(x.shape[0], 1, 1)
    integrator = integrators.ImplicitMidpointIntegrator(f, df, 0.1, N_fx=3)
    x0 = torch.tensor([[0.5, -0.2]])
    traj, _ = integrator(x0, 4, return_traj=True)
    assert traj.shape == (1, 4, 2)
    assert torch.allclose(traj[:, 0], x0)
    x_end, _ = integrator(x0, 4)
    assert torch.allclose(traj[:, -1], x_end, atol=1e-6)


def test_hamiltonian_implicit_midpoint_exposes_H():
    H = QuadraticHamiltonian()
    integrator = integrators.HamiltonianImplicitMidpointIntegrator(H, 0.1, N_fx=2)
    assert integrator.H is H


def test_separable_leapfrog_substeps_scale_step_size():
    H = QuadraticHamiltonian()
    q0, p0 = _quadratic_state()
    # SeparableLeapfrog applies one leapfrog step per outer iteration, so
    # substeps only scales the step size: gamma/substeps, not extra steps.
    substepped, p_substepped, _ = integrators.SeparableLeapfrogIntegrator(H, 0.2, substeps=2)(
        q0, p0, 5
    )
    halved, p_halved, _ = integrators.SeparableLeapfrogIntegrator(H, 0.1)(q0, p0, 5)
    assert torch.allclose(substepped, halved, atol=1e-6)
    assert torch.allclose(p_substepped, p_halved, atol=1e-6)


@pytest.mark.parametrize(
    "make_substepped, make_reference",
    [
        (
            lambda: integrators.HamiltonianEulerIntegrator(
                QuadraticHamiltonian(), 0.2, substeps=2
            ),
            lambda: integrators.HamiltonianEulerIntegrator(QuadraticHamiltonian(), 0.1),
        ),
        (
            lambda: integrators.ImplicitLeapfrogIntegrator(
                QuadraticHamiltonian(), 0.2, 5, substeps=2
            ),
            lambda: integrators.ImplicitLeapfrogIntegrator(QuadraticHamiltonian(), 0.1, 5),
        ),
        (
            lambda: integrators.ExplicitLeapfrogIntegrator(
                QuadraticHamiltonian(), 0.2, 1.0, substeps=2
            ),
            lambda: integrators.ExplicitLeapfrogIntegrator(QuadraticHamiltonian(), 0.1, 1.0),
        ),
        (
            lambda: integrators.HamiltonianImplicitMidpointIntegrator(
                QuadraticHamiltonian(), 0.2, N_fx=5, substeps=2
            ),
            lambda: integrators.HamiltonianImplicitMidpointIntegrator(
                QuadraticHamiltonian(), 0.1, N_fx=5
            ),
        ),
    ],
)
def test_substeps_repeat_the_step(make_substepped, make_reference):
    """These integrators loop `substeps` times per outer iteration, so
    (gamma, substeps=2, L) equals (gamma/2, substeps=1, 2L-1)."""
    q0, p0 = _quadratic_state()
    q_sub, p_sub, _ = make_substepped()(q0, p0, 3)
    q_ref, p_ref, _ = make_reference()(q0, p0, 5)
    assert torch.allclose(q_sub, q_ref, atol=1e-6)
    assert torch.allclose(p_sub, p_ref, atol=1e-6)


def test_explicit_leapfrog_binding_and_augmented_hamiltonian():
    H = QuadraticHamiltonian()
    integrator = integrators.ExplicitLeapfrogIntegrator(H, 0.1, omega=1.0)
    q0, p0 = _quadratic_state()
    q1 = q0 + 0.3
    p1 = p0 - 0.2
    binding = 0.5 * torch.sum((q1 - q0) ** 2) + 0.5 * torch.sum((p1 - p0) ** 2)
    expected = H(q0, p0) + H(q1, p1) + 1.0 * binding
    assert torch.allclose(integrator.H_augmented(q0, p0, q1, p1), expected, atol=1e-6)


def test_explicit_leapfrog_step_without_gamma_uses_default():
    H = QuadraticHamiltonian()
    integrator = integrators.ExplicitLeapfrogIntegrator(H, 0.1, omega=1.0)
    q0, p0 = _quadratic_state()
    implicit_gamma = integrator.leapfrog_step(q0, p0, q0, p0)
    explicit_gamma = integrator.leapfrog_step(q0, p0, q0, p0, gamma=torch.full((1, 1), 0.1))
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(implicit_gamma, explicit_gamma))


def test_explicit_forward_augmented_distinct_copies():
    integrator = integrators.ExplicitLeapfrogIntegrator(QuadraticHamiltonian(), 0.1, omega=1.0)
    q0, p0 = _quadratic_state()
    q1 = q0 + 0.3
    p1 = p0 - 0.2
    qL, _, _, pL1, log_det = integrator.forward_augmented(q0, p0, 4, q1, p1)
    assert qL.shape == (1, 2) and pL1.shape == (1, 2)
    assert torch.isfinite(qL).all() and torch.isfinite(pL1).all()
    assert torch.allclose(log_det, torch.zeros(1))

    traj = integrator.forward_augmented(q0, p0, 4, q1, p1, return_traj=True)
    traj_q, _, traj_q1, traj_p1, _ = traj
    assert traj_q.shape == (1, 4, 2) and traj_p1.shape == (1, 4, 2)
    assert torch.allclose(traj_q[:, 0], q0)
    assert torch.allclose(traj_q1[:, 0], q1)


def test_explicit_forward_stops_on_nonfinite_states():
    integrator = integrators.ExplicitLeapfrogIntegrator(QuadraticHamiltonian(), 0.1, omega=1.0)
    q0 = torch.tensor([[float("inf"), 0.0]])
    p0 = torch.zeros(1, 2)
    q1, _, log_det = integrator(q0, p0, 5)
    # No exception; the non-finite state is propagated as-is.
    assert not torch.isfinite(q1).all()
    assert torch.allclose(log_det, torch.zeros(1))
