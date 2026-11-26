"""Tests for Matsumoto and Slope metrics."""

import torch

from geodesic_toolbox import cometric, finsler

from .dummies import DummyAccuracyPredictor, DummyDirectionPredictor, Omega


def height_map(x: torch.Tensor) -> torch.Tensor:
    """A simple height map function."""
    return torch.sin(x[:, 0]) * torch.cos(x[:, 1])


def test_SlopeMetrics():
    model = finsler.SlopeMetrics(f=height_map)
    # df_ differentiates the height map w.r.t. x, so x must require grad
    x = torch.randn(10, 2, requires_grad=True)
    v = torch.randn(10, 2)
    metric_val = model(x, v)
    assert metric_val.shape == (10,)
    assert (metric_val > 0).all()


class OneForm(torch.nn.Module):
    def __init__(self, omega: Omega):
        super().__init__()
        self.omega = omega

    def forward(self, x, v):
        omega_x = self.omega(x)
        omega_x_v = torch.einsum("bi,bi->b", omega_x, v)
        return omega_x_v


def test_MatsumotoMetrics():
    cometric_model = cometric.IdentityCoMetric()
    direction = DummyDirectionPredictor(3)
    amplitude = DummyAccuracyPredictor()
    omega_model = Omega(direction, amplitude, cometric_model)
    beta = OneForm(omega_model)
    model = finsler.MatsumotoMetrics(alpha_inv=cometric_model, beta=beta)
    x = torch.randn(10, 3)
    v = torch.randn(10, 3)
    metric_val = model(x, v)
    assert metric_val.shape == (10,)
    assert (metric_val > 0).all()
