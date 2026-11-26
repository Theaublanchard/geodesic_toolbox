"""Tests for PointCarreCoMetric."""

import torch

from geodesic_toolbox import cometric


def test_pointcarre_instantiate():
    print(cometric.PointCarreCoMetric())


def test_pointcarre():
    x = 2 * torch.rand(3, 2) - 1
    cm = cometric.PointCarreCoMetric()
    cm(x)
    out = cm(torch.zeros(1, 2))
    assert out.shape == (1, 2, 2)
    assert (out[0] == 1 / 4 * torch.eye(2)).all()


def test_logdet():
    x = 2 * torch.rand(3, 2) - 1
    cm = cometric.PointCarreCoMetric()
    logdet = cm.logdet(x)
    norm_q_sqr = torch.linalg.vector_norm(x, dim=1) ** 2
    scalar = 4 / (1 - norm_q_sqr) ** 2
    true_log_det = 2 * torch.log(scalar)
    assert logdet.shape == (3,)
    assert torch.allclose(logdet, true_log_det)


def test_invlogdet():
    x = 2 * torch.rand(3, 2) - 1
    cm = cometric.PointCarreCoMetric()
    logdet = cm.inv_logdet(x)
    norm_q_sqr = torch.linalg.vector_norm(x, dim=1) ** 2
    scalar = 1 / 4 * (1 - norm_q_sqr) ** 2
    true_log_det = 2 * torch.log(scalar)
    assert logdet.shape == (3,)
    assert torch.allclose(logdet, true_log_det)


def test_dot_not_diag():
    x = 2 * torch.rand(3, 2) - 1
    v = torch.randn(3, 2)
    cm = cometric.PointCarreCoMetric()
    out = cm.inv_dot(x, v, v)
    norm_q_sqr = torch.linalg.vector_norm(x, dim=1) ** 2
    scalar = 1 / 4 * (1 - norm_q_sqr) ** 2
    true_out = scalar * torch.sum(v**2, dim=1)
    assert torch.allclose(out, true_out)


def test_invdot_not_diag():
    x = 2 * torch.rand(3, 2) - 1
    p = torch.randn(3, 2)
    cm = cometric.PointCarreCoMetric()
    out = cm.dot(x, p, p)
    norm_q_sqr = torch.linalg.vector_norm(x, dim=1) ** 2
    scalar = 4 / (1 - norm_q_sqr) ** 2
    true_out = scalar * torch.sum(p**2, dim=1)
    assert torch.allclose(out, true_out)
