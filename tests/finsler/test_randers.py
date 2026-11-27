"""Tests for RandersMetrics."""

import pytest
import torch

from geodesic_toolbox import cometric, finsler

from .dummies import DummyAccuracyPredictor, DummyDirectionPredictor, Omega


def test_forward_randers():
    """Test the forward pass of the Omega class."""
    batch_size = 10
    dim = 5
    X = torch.randn(batch_size, dim)
    v = torch.randn(batch_size, dim)

    direction = DummyDirectionPredictor(dim)
    amplitude = DummyAccuracyPredictor()
    cometric_model = cometric.IdentityCoMetric()

    omega_model = Omega(direction, amplitude, cometric_model)
    randers = finsler.RandersMetrics(
        base_cometric=cometric_model,
        omega=omega_model,
        beta=1.0,
    )
    metric_val = randers(X, v)
    assert metric_val.shape == (batch_size,)
    assert (metric_val > 0).all()


def test_fund_tensor_analytic_randers():
    x = torch.randn(10, 3)
    v = torch.randn(10, 3)
    cometric_model = cometric.IdentityCoMetric()
    direction = DummyDirectionPredictor(3)
    amplitude = DummyAccuracyPredictor()
    omega_model = Omega(direction, amplitude, cometric_model)
    randers = finsler.RandersMetrics(
        base_cometric=cometric_model,
        omega=omega_model,
        beta=1.0,
    )
    fund_tensor = randers.fundamental_tensor(x, v)
    assert fund_tensor.shape == (10, 3, 3)
    assert torch.allclose(fund_tensor, fund_tensor.transpose(1, 2))
    for i in range(10):
        eigvals = torch.linalg.eigvalsh(fund_tensor[i])
        assert (eigvals > 0).all()


def test_fund_tensor_grad_randers():
    x = torch.randn(10, 3)
    v = torch.randn(10, 3)
    cometric_model = cometric.IdentityCoMetric()
    direction = DummyDirectionPredictor(3)
    amplitude = DummyAccuracyPredictor()
    omega_model = Omega(direction, amplitude, cometric_model)
    randers = finsler.RandersMetrics(
        base_cometric=cometric_model,
        omega=omega_model,
        beta=1.0,
    )
    fund_tensor = randers.fundamental_tensor(x, v)
    assert fund_tensor.shape == (10, 3, 3)
    assert torch.allclose(fund_tensor, fund_tensor.transpose(1, 2))
    for i in range(10):
        eigvals = torch.linalg.eigvalsh(fund_tensor[i])
        assert (eigvals > 0).all()


def test_fund_tensor_exact_randers():
    x = torch.randn(10, 3)
    v = torch.randn(10, 3)
    cometric_model = cometric.IdentityCoMetric(is_diag=False)
    direction = DummyDirectionPredictor(3)
    amplitude = DummyAccuracyPredictor()
    omega_model = Omega(direction, amplitude, cometric_model)
    randers = finsler.RandersMetrics(
        base_cometric=cometric_model,
        omega=omega_model,
        beta=0.0,
    )
    fund_tensor = randers.fundamental_tensor(x, v)
    assert torch.allclose(fund_tensor, cometric_model(x))


def test_invfund_tensor_analytic():
    x, v = torch.randn(10, 3), torch.randn(10, 3)
    cometric_model = cometric.IdentityCoMetric()
    direction = DummyDirectionPredictor(3)
    amplitude = DummyAccuracyPredictor()
    omega_model = Omega(direction, amplitude, cometric_model)
    randers = finsler.RandersMetrics(
        base_cometric=cometric_model,
        omega=omega_model,
        beta=1.0,
    )
    fund = randers.fundamental_tensor(x, v)
    inv_fund = randers.inv_fund_tensor_analytic_(x, v)
    assert torch.allclose(fund.inverse(), inv_fund, atol=1e-5)


def test_invfund_tensor():
    x, v = torch.randn(10, 3), torch.randn(10, 3)
    cometric_model = cometric.IdentityCoMetric()
    direction = DummyDirectionPredictor(3)
    amplitude = DummyAccuracyPredictor()
    omega_model = Omega(direction, amplitude, cometric_model)
    randers = finsler.RandersMetrics(
        base_cometric=cometric_model,
        omega=omega_model,
        beta=1.0,
    )
    fund = randers.fundamental_tensor(x, v)
    inv_fund = randers.inverse_fundamental_tensor(x, v)
    assert torch.allclose(fund.inverse(), inv_fund, atol=1e-5)


def test_invfund_tensor_non_diag():
    x, v = torch.randn(10, 3), torch.randn(10, 3)
    cometric_model = cometric.IdentityCoMetric(is_diag=False)
    direction = DummyDirectionPredictor(3)
    amplitude = DummyAccuracyPredictor()
    omega_model = Omega(direction, amplitude, cometric_model)
    randers = finsler.RandersMetrics(
        base_cometric=cometric_model,
        omega=omega_model,
        beta=.9,
    )
    fund = randers.fundamental_tensor(x, v)
    inv_fund = randers.inv_fund_tensor_analytic_(x, v)
    assert torch.allclose(fund.inverse(), inv_fund, atol=1e-5)


def test_beta_form():
    x, v = torch.randn(10, 3), torch.randn(10, 3)
    cometric_model = cometric.IdentityCoMetric()
    omega_model = Omega(DummyDirectionPredictor(3), DummyAccuracyPredictor(), cometric_model)
    beta = 0.7
    randers = finsler.RandersMetrics(base_cometric=cometric_model, omega=omega_model, beta=beta)
    out = randers.beta_form(x, v)
    expected = beta * torch.einsum("bi,bi->b", omega_model(x), v)
    assert out.shape == (10,)
    assert torch.allclose(out, expected)


def test_legendre_transform_diag():
    x, v = torch.randn(10, 3), torch.randn(10, 3)
    cometric_model = cometric.IdentityCoMetric()
    omega_model = Omega(DummyDirectionPredictor(3), DummyAccuracyPredictor(), cometric_model)
    beta = 0.8
    randers = finsler.RandersMetrics(base_cometric=cometric_model, omega=omega_model, beta=beta)
    p = randers.legendre_transform(x, v)
    assert p.shape == (10, 3)
    # With the identity cometric, G v = v and the closed form is
    # p = F * (v / |v| + beta * omega)
    b = beta * omega_model(x)
    v_norm = v.norm(dim=-1)
    F = v_norm + torch.einsum("bi,bi->b", b, v)
    expected = F[:, None] * (v / v_norm[:, None] + b)
    assert torch.allclose(p, expected, atol=1e-6)


def test_legendre_transform_non_diag():
    x, v = torch.randn(10, 3), torch.randn(10, 3)
    cometric_model = cometric.IdentityCoMetric(is_diag=False)
    omega_model = Omega(DummyDirectionPredictor(3), DummyAccuracyPredictor(), cometric_model)
    beta = 0.8
    randers = finsler.RandersMetrics(base_cometric=cometric_model, omega=omega_model, beta=beta)
    p = randers.legendre_transform(x, v)
    assert p.shape == (10, 3)
    G = cometric_model.metric_tensor(x)
    Gv = torch.einsum("bij,bj->bi", G, v)
    b = beta * omega_model(x)
    v_norm = torch.einsum("bij,bi,bj->b", G, v, v).sqrt()
    F = v_norm + torch.einsum("bi,bi->b", b, v)
    expected = F[:, None] * (Gv / v_norm[:, None] + b)
    assert torch.allclose(p, expected, atol=1e-6)


def test_legendre_transform_zero_v():
    # A zero tangent vector must not produce NaNs (norm is clamped to eps).
    x = torch.randn(5, 3)
    v = torch.zeros(5, 3)
    cometric_model = cometric.IdentityCoMetric()
    omega_model = Omega(DummyDirectionPredictor(3), DummyAccuracyPredictor(), cometric_model)
    randers = finsler.RandersMetrics(base_cometric=cometric_model, omega=omega_model, beta=0.5)
    p = randers.legendre_transform(x, v)
    assert torch.isfinite(p).all()
    # v = 0: p = F * omega with F = beta * omega . 0 = 0 -> p = 0
    assert torch.allclose(p, torch.zeros_like(p), atol=1e-6)


def _make_primal_randers(is_diag: bool, beta: float = 1.0) -> finsler.RandersMetrics:
    cometric_model = cometric.IdentityCoMetric(is_diag=is_diag)
    omega_model = Omega(DummyDirectionPredictor(3), DummyAccuracyPredictor(), cometric_model)
    return finsler.RandersMetrics(base_cometric=cometric_model, omega=omega_model, beta=beta)


def test_dual_randers_diag():
    x, v = torch.randn(10, 3), torch.randn(10, 3)
    primal = _make_primal_randers(is_diag=True)
    dual = finsler.DualRandersMetrics(primal)
    # Forward pass
    out = dual(x, v)
    assert out.shape == (10,)
    assert (out >= dual.epsilon).all()
    # omega_star
    omega_star = dual.omega_star(x)
    assert omega_star.shape == (10, 3)
    # For the identity cometric, omega* = -omega / (1 - ||omega||^2)
    omega = primal.beta * primal.omega(x)
    alpha = 1 - (omega**2).sum(dim=-1)
    assert torch.allclose(omega_star, -omega / alpha[:, None], atol=1e-6)
    # G_star
    G_star = dual.G_star(x)
    assert G_star.shape == (10, 3, 3)
    for i in range(10):
        eigvals = torch.linalg.eigvalsh(G_star[i])
        assert (eigvals > 0).all()
    # Wrappers: dual.omega and the dual cometric must route through omega_star / G_star
    assert torch.allclose(dual.omega(x), dual.omega_star(x))
    G_dual = dual.base_cometric.metric_tensor(x)
    C_dual = dual.base_cometric.cometric_tensor(x)
    assert torch.allclose(G_dual, G_star)
    eye = torch.eye(3).expand(10, 3, 3)
    assert torch.allclose(C_dual @ G_dual, eye, atol=1e-5)


def test_dual_randers_non_diag():
    x, v = torch.randn(10, 3), torch.randn(10, 3)
    primal = _make_primal_randers(is_diag=False)
    dual = finsler.DualRandersMetrics(primal)
    out = dual(x, v)
    assert out.shape == (10,)
    assert (out >= dual.epsilon).all()
    omega_star = dual.omega_star(x)
    assert omega_star.shape == (10, 3)
    G_star = dual.G_star(x)
    assert G_star.shape == (10, 3, 3)
    for i in range(10):
        eigvals = torch.linalg.eigvalsh(G_star[i])
        assert (eigvals > 0).all()
    # F*(x, v) = |v|_{G*} + omega* . v
    omega = primal.beta * primal.omega(x)
    G_inv = primal.base_cometric.cometric_tensor(x)
    G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)
    alpha = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)
    omega_star_expected = -1 / alpha[:, None] * G_inv_w
    assert torch.allclose(omega_star, omega_star_expected, atol=1e-6)


def test_dual_randers_epsilon():
    x, v = torch.randn(5, 3), torch.randn(5, 3)
    primal = _make_primal_randers(is_diag=True)
    dual = finsler.DualRandersMetrics(primal, epsilon=1e-3)
    out = dual(x, v)
    assert (out > 0).all()


def test_binet_legendre_diag():
    x = torch.randn(10, 3)
    randers = _make_primal_randers(is_diag=True)
    bl = finsler.BinetLegendreRanders(randers)
    out = bl(x)
    assert out.shape == (10, 3, 3)
    assert torch.allclose(out, bl.cometric_tensor(x))
    assert torch.allclose(out, out.transpose(1, 2), atol=1e-6)
    for i in range(10):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert (eigvals > 0).all()


def test_binet_legendre_non_diag():
    x = torch.randn(10, 3)
    randers = _make_primal_randers(is_diag=False)
    bl = finsler.BinetLegendreRanders(randers)
    out = bl(x)
    assert out.shape == (10, 3, 3)
    assert torch.allclose(out, bl.cometric_tensor(x))
    assert torch.allclose(out, out.transpose(1, 2), atol=1e-6)
    for i in range(10):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert (eigvals > 0).all()


def test_beta_out_of_range():
    cometric_model = cometric.IdentityCoMetric()
    omega_model = Omega(DummyDirectionPredictor(3), DummyAccuracyPredictor(), cometric_model)
    with pytest.raises(AssertionError):
        finsler.RandersMetrics(base_cometric=cometric_model, omega=omega_model, beta=1.5)
