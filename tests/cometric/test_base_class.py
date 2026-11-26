"""Tests for the cometric base classes (CoMetric, IdentityCoMetric, composition)."""

import torch

from geodesic_toolbox import cometric


def test_identity_instantiate():
    print(cometric.IdentityCoMetric())


def test_identity_cometric_basic():  # Test that IdentityCoMetric returns correct values
    dim = 3
    batch = 5
    x = torch.randn(batch, dim)
    coscale = 2.0
    cm = cometric.IdentityCoMetric(coscale=coscale)
    out = cm.cometric_tensor(x)
    assert torch.allclose(out, coscale * torch.ones_like(x))
    metric = cm.metric_tensor(x)
    assert torch.allclose(metric, 1 / coscale * torch.ones_like(x))
    p = torch.randn(batch, dim)
    metric_val = cm.dot(x, p, p)
    assert torch.allclose(metric_val, 1 / coscale * torch.sum(p**2, dim=1))
    v = torch.randn(batch, dim)
    cometric_val = cm.inv_dot(x, v, v)
    assert torch.allclose(cometric_val, coscale * torch.sum(v**2, dim=1))
    print(cometric)


def test_lodet_diagonal():
    cm = cometric.IdentityCoMetric()
    x = torch.randn(4, 3)
    logdet = cm.logdet(x)
    true_log_det = 3 * torch.log(torch.tensor(1.0))
    assert logdet.shape == (4,)
    assert torch.allclose(logdet, true_log_det)


def test_invlogdet_diagonal():
    cm = cometric.IdentityCoMetric()
    x = torch.randn(4, 3)
    logdet = cm.inv_logdet(x)
    true_log_det = -3 * torch.log(torch.tensor(1.0))
    assert logdet.shape == (4,)
    assert torch.allclose(logdet, true_log_det)


def test_dot_diagonal():
    cm = cometric.IdentityCoMetric()
    x = torch.randn(4, 3)
    v = torch.randn(4, 3)
    out = cm.inv_dot(x, v, v)
    true_out = torch.sum(v**2, dim=1)
    assert torch.allclose(out, true_out)


def test_invdot_diagonal():
    cm = cometric.IdentityCoMetric()
    x = torch.randn(4, 3)
    p = torch.randn(4, 3)
    out = cm.dot(x, p, p)
    true_out = torch.sum(p**2, dim=1)
    assert torch.allclose(out, true_out)


def test_angle():
    cm = cometric.IdentityCoMetric()
    x = torch.randn(1, 2)
    u = torch.randn(1, 2)
    v = torch.randn(1, 2)
    angle = cm.angle(x, u, v)
    true_angle = torch.acos(torch.cosine_similarity(u, v))
    assert torch.allclose(angle, true_angle)

    ############################################################
    # Sum and product cometric
    ############################################################


def test_sum_cometric_all_diag():  # Sum is done at the metric tensor level: G = G_1 + G_2 = 2 I
    cm_1 = cometric.IdentityCoMetric()
    cm_2 = cometric.IdentityCoMetric()
    x = torch.randn(4, 3)
    sum_cm = cm_1 + cm_2
    out = sum_cm.cometric_tensor(x)
    out_inv = sum_cm.metric_tensor(x)
    assert torch.allclose(out_inv, 2 * torch.ones_like(x))
    assert torch.allclose(out, 0.5 * torch.ones_like(x))


def test_sum_cometric_mixed_1():  # At the origin: G_pointcarre = 4 I, G_identity = I, so G = 5 I
    cm_2 = cometric.IdentityCoMetric()
    cm_1 = cometric.PointCarreCoMetric()
    x = torch.zeros(1, 2)
    sum_cm = cm_1 + cm_2
    out = sum_cm.cometric_tensor(x)
    out_inv = sum_cm.metric_tensor(x)
    assert torch.allclose(out_inv, 5 * torch.eye(2).unsqueeze(0))
    assert torch.allclose(out, 1 / 5 * torch.eye(2).unsqueeze(0))


def test_sum_cometric_mixed_2():  # At the origin: G_pointcarre = 4 I, G_identity = I, so G = 5 I
    cm_1 = cometric.IdentityCoMetric()
    cm_2 = cometric.PointCarreCoMetric()
    x = torch.zeros(1, 2)
    sum_cm = cm_1 + cm_2
    out = sum_cm.cometric_tensor(x)
    out_inv = sum_cm.metric_tensor(x)
    assert torch.allclose(out_inv, 5 * torch.eye(2).unsqueeze(0))
    assert torch.allclose(out, 1 / 5 * torch.eye(2).unsqueeze(0))


def test_sum_cometric_none_diag():  # At the origin: G = 4 I + 4 I = 8 I
    cm_1 = cometric.PointCarreCoMetric()
    cm_2 = cometric.PointCarreCoMetric()
    x = torch.zeros(1, 2)
    sum_cm = cm_1 + cm_2
    out = sum_cm.cometric_tensor(x)
    out_inv = sum_cm.metric_tensor(x)
    assert torch.allclose(out_inv, 8 * torch.eye(2).unsqueeze(0))
    assert torch.allclose(out, 1 / 8 * torch.eye(2).unsqueeze(0))


def test_product_cometric():
    cm_1 = cometric.IdentityCoMetric()
    scale = 2
    prod_cm = scale * cm_1
    x = torch.randn(4, 3)
    out = prod_cm.cometric_tensor(x)
    out_inv = prod_cm.metric_tensor(x)
    assert torch.allclose(out_inv, 1 / scale * torch.ones_like(x))
    assert torch.allclose(out, scale * torch.ones_like(x))


############################################################
# Pointcarré cometric
############################################################
