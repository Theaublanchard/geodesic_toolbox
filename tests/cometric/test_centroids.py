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


def test_compute_global_temperature_square():
    centroids = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    # 2nd-nearest neighbor (neighbor_k=1) distances:
    #   (0,0) -> (0,2) : 2
    #   (1,0) -> (0,2) : sqrt(5)
    #   (0,2) -> (1,0) : sqrt(5)
    # max squared distance = 5
    temp = cometric.compute_global_temperature_square(centroids)
    assert temp == pytest.approx(5.0)
    # The temperature does not depend on the batch: same for every centroid
    assert cometric.compute_global_temperature_square(centroids, neighbor_k=0) == pytest.approx(
        4.0
    )  # 1st neighbor of (1,0)/(0,2) is 1 and sqrt(2)... max = 2 -> squared 4


def test_centroids_cometric_use_global_temperature():
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.stack([torch.eye(dim) for _ in range(K)])
    kappa = 2.0
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        use_global_temperature=True,
        kappa=kappa,
    )
    # All bandwidths must share the same global temperature
    assert cm.tau_squared.shape == (K,)
    assert (cm.tau_squared == cm.tau_squared[0]).all()
    temp = cometric.compute_global_temperature_square(centroids)
    assert torch.allclose(cm.tau_squared, torch.full((K,), kappa * temp))
    x = torch.randn(3, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (3, dim, dim)
    for i in range(3):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert (eigvals > 0).all()


def test_centroid_scales_all_non_finite():
    # A non-finite kappa makes every bandwidth non-finite: the fallback fills them with 1.0.
    centroids = torch.randn(5, 2)
    data = torch.randn(20, 2)
    tau2 = cometric.compute_centroid_scales(
        centroids, data, torch.tensor(float("nan"))
    )
    assert tau2.shape == (5,)
    assert (tau2 == 1.0).all()


def test_centroid_scales_partial_non_finite():
    # A non-finite kappa for a single centroid makes exactly one bandwidth non-finite
    # while the others stay finite: the non-finite entry must be replaced by the
    # median of the finite ones.
    centroids = torch.randn(7, 2)
    data = torch.randn(20, 2)
    kappa = torch.ones(7)
    kappa[3] = float("nan")
    tau2 = cometric.compute_centroid_scales(centroids, data, kappa)
    assert tau2.shape == (7,)
    assert torch.isfinite(tau2).all()
    assert (tau2 > 0).all()
    # The non-finite entry is filled with the median of the finite ones
    finite = tau2[torch.tensor([0, 1, 2, 4, 5, 6])]
    assert tau2[3] == torch.median(finite)


############################################################
# LANDCometric
############################################################


def test_land_cometric_basic():
    K = 6
    dim = 3
    centroids = torch.rand(K, dim)
    cm = cometric.LANDCometric(centroids=centroids, alpha=1)
    assert cm.is_diag
    assert cm.K == K
    x = torch.rand(4, dim)
    h = cm.h(x)
    assert h.shape == (4, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, dim)
    # h(x) >= 0 and reg_coef > 0
    assert (out >= 1e-3 - 1e-12).all()
    # extra_repr must not raise
    assert "K=6" in cm.extra_repr()


def test_land_cometric_alpha2_global_temp():
    K = 8
    dim = 2
    centroids = torch.rand(K, dim)
    kappa = 2.0
    cm = cometric.LANDCometric(
        centroids=centroids, alpha=2, kappa=kappa, use_global_temp=True
    )
    assert cm.K == K
    # All bandwidths share the global temperature
    assert (cm.tau_squared == cm.tau_squared[0]).all()
    x = torch.rand(3, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (3, dim)
    assert (out > 0).all()


def test_land_cometric_kmedoids():
    K = 8
    dim = 2
    centroids = torch.rand(K, dim)
    cm = cometric.LANDCometric(centroids=centroids, K=4)
    assert cm.K == 4
    # K larger than the number of centroids falls back to using all of them
    cm_big = cometric.LANDCometric(centroids=centroids, K=1000)
    assert cm_big.K == K
    x = torch.rand(3, dim)
    assert cm.cometric_tensor(x).shape == (3, dim)
    assert cm_big.cometric_tensor(x).shape == (3, dim)


def test_land_cometric_invalid_params():
    centroids = torch.rand(4, 2)
    with pytest.raises(AssertionError):
        cometric.LANDCometric(centroids=centroids, alpha=0)
    with pytest.raises(AssertionError):
        cometric.LANDCometric(centroids=centroids, alpha=2.5)
    with pytest.raises(AssertionError):
        cometric.LANDCometric(centroids=centroids, reg_coef=-1.0)
    with pytest.raises(AssertionError):
        cometric.LANDCometric(centroids=centroids.unsqueeze(0), alpha=1)


############################################################
# LANDRBFCometric
############################################################


def test_land_rbf_cometric_fixed_weights():
    N = 12
    dim = 2
    data = torch.rand(N, dim)
    cm = cometric.LANDRBFCometric(data=data, K=5)
    assert cm.is_diag
    assert cm.K == 5
    # Fixed weights: w = softplus(1/K) for every kernel
    assert cm.w.shape == (5,)
    assert torch.allclose(cm.w, torch.full((5,), torch.nn.functional.softplus(torch.tensor(1.0 / 5))))
    x = torch.rand(3, dim)
    h = cm.h(x, cm.w)
    assert h.shape == (3,)
    out = cm.cometric_tensor(x)
    assert out.shape == (3, dim)
    assert (out > 0).all()


def test_land_rbf_cometric_learn_weights():
    N = 10
    dim = 2
    data = torch.rand(N, dim)
    cm = cometric.LANDRBFCometric(data=data, learn_weights=True)
    assert cm.K == N
    assert cm.w.shape == (N,)
    assert cm.loss_list.shape == (400,)
    # Loss should have decreased over the optimization
    assert cm.loss_list[-1] <= cm.loss_list[0]
    x = torch.rand(3, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (3, dim)
    assert (out > 0).all()


def test_land_rbf_cometric_kmedoids_and_invalid():
    N = 8
    dim = 2
    data = torch.rand(N, dim)
    cm = cometric.LANDRBFCometric(data=data, K=3)
    assert cm.K == 3
    # K larger than the number of points falls back to using all points
    cm_big = cometric.LANDRBFCometric(data=data, K=100)
    assert cm_big.K == N
    with pytest.raises(AssertionError):
        cometric.LANDRBFCometric(data=data, reg_coef=-1.0)
    with pytest.raises(AssertionError):
        cometric.LANDRBFCometric(data=data, kappa=0.0)
    with pytest.raises(AssertionError):
        cometric.LANDRBFCometric(data=data.unsqueeze(0), K=2)


############################################################
# DiagonalCometricModel cometric
############################################################
