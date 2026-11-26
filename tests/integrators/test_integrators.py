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
