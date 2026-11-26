"""Tests for RandersMetrics."""

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
