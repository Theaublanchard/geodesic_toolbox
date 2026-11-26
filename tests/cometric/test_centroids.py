"""Tests for CentroidsCometric."""

import pytest
import torch

from geodesic_toolbox import cometric


def test_centroids_cometric_instantiante():
    centroids = torch.randn(4, 2)
    cometric_centroids = torch.stack([torch.eye(2) for _ in range(4)])
    print(
        cometric.CentroidsCometric(centroids=centroids, cometric_centroids=cometric_centroids)
    )


def test_centroids_cometric_basic_not_diag():  # Test CentroidsCometric interpolation
    dim = 2
    K = 4
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.stack([torch.eye(dim) for _ in range(K)])
    reg_coef = 1e-3
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        reg_coef=reg_coef,
        metric_weight=False,
        K=1,
    )
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        reg_coef=reg_coef,
        metric_weight=False,
        K=10000000,
    )
    # Test output shape
    x = torch.randn(3, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (3, dim, dim)
    # Should be positive definite (diagonal dominant)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_centroids_cometric_basic_diag():  # Test CentroidsCometric interpolation
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.ones_like(centroids)
    reg_coef = 1e-3
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        reg_coef=reg_coef,
        metric_weight=False,
    )
    # Test output shape
    x = torch.randn(3, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (3, dim)
    # Should be positive definite (diagonal dominant)
    assert torch.all(out > 0)


def test_centroids_cometric_metric_weight_true_not_diag():  # Test CentroidsCometric with metric_weight=True
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.stack([torch.eye(dim) for _ in range(K)])
    reg_coef = 1e-3
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        reg_coef=reg_coef,
        metric_weight=True,
    )
    x = torch.randn(2, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (2, dim, dim)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_centroids_cometric_metric_weight_true_diag():  # Test CentroidsCometric with metric_weight=True
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.ones_like(centroids)
    reg_coef = 1e-3
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        reg_coef=reg_coef,
        metric_weight=True,
    )
    x = torch.randn(2, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (2, dim)
    assert torch.all(out > 0)


def test_centroids_not_symetric():
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.eye(dim).unsqueeze(0).repeat(K, 1, 1)
    cometric_centroids[0, 0, 1] = 10.0  # Not symmetric
    cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        reg_coef=1e-3,
        metric_weight=True,
    )


def test_centroids_diff_shape_diag():
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.randn(K, dim + 1)  # Wrong shape
    with pytest.raises(AssertionError):
        cometric.CentroidsCometric(
            centroids=centroids,
            cometric_centroids=cometric_centroids,
            reg_coef=1e-3,
            metric_weight=True,
        )


def test_centroids_not_square():
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.randn(K, dim, dim + 1)  # Wrong shape
    with pytest.raises(AssertionError):
        cometric.CentroidsCometric(
            centroids=centroids,
            cometric_centroids=cometric_centroids,
            reg_coef=1e-3,
            metric_weight=True,
            K=K,
        )


def _test_load_centroids(diag=True):  # Test loading centroids from file
    import os

    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    if diag:
        cometric_centroids = torch.ones_like(centroids)
    else:
        cometric_centroids = torch.stack([torch.eye(dim) for _ in range(K)])
    reg_coef = 1e-3
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        reg_coef=reg_coef,
        metric_weight=False,
        K=K,
    )
    filepath = "test_centroids.pth"
    torch.save(cm.state_dict(), filepath)
    cm_loaded = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        reg_coef=reg_coef,
        metric_weight=False,
        K=K,
    )
    cm_loaded.load_state_dict(torch.load(filepath))
    os.remove(filepath)

    # The loaded cometric must behave like the original
    x = torch.randn(3, dim)
    assert torch.allclose(cm(x), cm_loaded(x))


def test_load_centroids():
    _test_load_centroids(diag=True)
    _test_load_centroids(diag=False)


############################################################
# DiagonalCometricModel cometric
############################################################
