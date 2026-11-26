"""Tests for SoftAbsCometric."""

import pytest
import torch

from geodesic_toolbox import cometric


def test_fail_instantiate_SoftAbsCometric():
    with pytest.raises(NotImplementedError):
        cometric.SoftAbsCometric(cometric.IdentityCoMetric())


def test_forward_pass_SoftAbsCometric():
    base_cm = cometric.PointCarreCoMetric()
    softabs_cm = cometric.SoftAbsCometric(base_cm, alpha=10)
    x = torch.randn(4, 2)
    out = softabs_cm.cometric_tensor(x)
    assert out.shape == (4, 2, 2)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)
    inv_out = softabs_cm.metric_tensor(x)
    assert inv_out.shape == (4, 2, 2)
    for i in range(inv_out.shape[0]):
        eigvals = torch.linalg.eigvalsh(inv_out[i])
        assert torch.all(eigvals > 0)
