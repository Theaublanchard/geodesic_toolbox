"""Tests for the Finsler base classes."""

import torch

from geodesic_toolbox import finsler


def test_ToyFinslerMetric():
    model = finsler.ToyFinslerMetric()
    x = torch.randn(10, 3)
    v = torch.randn(10, 3)
    metric_val = model(x, v)
    assert metric_val.shape == (10,)
    assert (metric_val > 0).all()
