"""Tests for pullback-style cometrics (PullBack, PBIG, Lifted, FisherRao, height maps)."""

import torch

from geodesic_toolbox import cometric


def log_likelihood(x, mu):
    distr = torch.distributions.MultivariateNormal(
        loc=mu, covariance_matrix=torch.eye(mu.shape[-1])
    )
    return distr.log_prob(x)


def test_fisher_rao_cometric():
    cm = cometric.FisherRaoCometric(
        log_likelihood=log_likelihood,
        reg_coef=1e-3,
        softabs_alpha=1e5,
        N_pts=100,
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_fisher_rao_cometric_softabs():
    fim = cometric.FisherRaoCometric(
        log_likelihood=log_likelihood,
        N_pts=100,
    )
    theta = torch.randn(4, 3)
    out = fim.cometric_tensor(theta)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_pullback_cometric_identity_base():
    class DummyDiffeoMorphism(torch.nn.Module):
        def forward(self, q):
            return q**3

    cm = cometric.PullBackCometric(
        diffeo=DummyDiffeoMorphism(),
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_pullback_cometric_non_identity_base():
    class DummyDiffeoMorphism(torch.nn.Module):
        def forward(self, q):
            return q**3

    cm = cometric.PullBackCometric(
        diffeo=DummyDiffeoMorphism(),
        base_cometric=cometric.PointCarreCoMetric(),
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_pullback_cometric_autograd():

    class DummyDiffeoMorphism(torch.nn.Module):
        def forward(self, q):
            return q**3

    cm = cometric.PullBackCometric(
        diffeo=DummyDiffeoMorphism(),
        method="autograd",
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_pullback_cometric_jacobian_method():
    class DummyDiffeoMorphism(torch.nn.Module):
        def forward(self, q):
            return q**3

        def jacobian(self, x):
            batch_size, dim = x.shape
            J = torch.zeros(batch_size, dim, dim)
            for i in range(dim):
                J[:, i, i] = 3 * x[:, i] ** 2
            return J

    cm = cometric.PullBackCometric(
        diffeo=DummyDiffeoMorphism(),
        method="jacobian_method",
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_lifted_cometric():
    base_cm = cometric.IdentityCoMetric()

    class DummyClassifier(torch.nn.Module):
        def forward(self, q):
            # (B,1) output as required by LiftedCometric
            return (1 / (q[:, 0] + 1e-3))[:, None]

    h = DummyClassifier()
    lifted_cm = cometric.LiftedCometric(
        base_cometric=base_cm,
        h=h,
        beta=1.0,
    )
    x = torch.randn(4, 3)
    out = lifted_cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_funcheightmap_cometric():
    def f(x, y):
        return torch.sin(x) * torch.cos(y)

    cm = cometric.FunctionalHeightMapCometric(func=f)
    x = torch.randn(4, 2)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 2, 2)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)
