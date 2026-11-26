import torch

from geodesic_toolbox import cometric, utils


def test_delta_and_pointcarre_distance_consistency():
    u = torch.tensor([[0.1, 0.0], [0.0, 0.2]])
    v = torch.tensor([[0.2, 0.0], [0.0, 0.1]])

    delta_val = utils.delta(u, v)
    manual_delta = 2 * torch.linalg.vector_norm(u - v, dim=-1) ** 2
    manual_delta /= (1 - torch.linalg.vector_norm(u, dim=-1) ** 2) * (
        1 - torch.linalg.vector_norm(v, dim=-1) ** 2
    )
    assert torch.allclose(delta_val, manual_delta)

    pc_dist = utils.pointcarre_dst(u, v)
    assert torch.allclose(pc_dist, torch.arccosh(1 + manual_delta))


def test_scaled_euclidean_distance():
    u = torch.tensor([[0.0, 0.0]])
    v = torch.tensor([[3.0, 4.0]])
    scale = torch.tensor([4.0])

    dist = utils.scaled_euclidean_dst(u, v, scale)
    assert torch.allclose(dist, torch.tensor([10.0]))


def test_get_bounds_with_margin():
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 2.0], [-1.0, -2.0]])
    bounds = utils.get_bounds(embeddings, margin=0.1)

    # Without margin the bounds would be [-1, 1, -2, 2]
    assert bounds[0] < -1.0 and bounds[1] > 1.0
    assert bounds[2] < -2.0 and bounds[3] > 2.0


def test_magnification_factor_matches_metric_and_cometric():
    points = torch.tensor([[0.0, 0.0], [0.5, -0.5]])

    diag_metric = cometric.IdentityCoMetric()
    full_metric = cometric.IdentityCoMetric(is_diag=False)

    mf_diag = utils.magnification_factor(diag_metric, points)
    mf_full = utils.magnification_factor(full_metric, points)
    mf_from_metric = utils.magnification_factor_metric(full_metric, points)

    assert torch.allclose(mf_diag, torch.ones_like(mf_diag))
    assert torch.allclose(mf_full, mf_from_metric)


def test_learning_rate_scaling_monotonicity():
    base_lr = 0.01
    low = utils.scale_lr_magnification(0.0, base_lr)
    high = utils.scale_lr_magnification(10.0, base_lr)

    assert low < base_lr
    assert high > low


def test_time_scaling_schedules_bounds():
    assert utils.constant_time_scaling_schedule(0.0) == 1
    assert utils.linear_time_scaling_schedule(0.0) > utils.linear_time_scaling_schedule(1.0)
    assert utils.cosine_time_scaling_schedule(0.0) > utils.cosine_time_scaling_schedule(1.0)


def test_batched_kronecker_products():
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    b = torch.tensor([[[0.0, 5.0], [6.0, 7.0]]])

    kro = utils.batched_kro(a, b)
    expected = torch.kron(a.squeeze(0), b.squeeze(0)).unsqueeze(0)
    assert torch.allclose(kro, expected)


def test_emp_cov_mat():
    x = torch.randn(32, 7)
    mean = x.mean(dim=0)
    cov = utils.empirical_cov_mat(x)
    cov_mu = utils.empirical_cov_mat(x, mu=mean)
    assert torch.allclose(cov, cov_mu)
    assert cov.shape == (7, 7)
    assert cov_mu.shape == (7, 7)
    # Check that covariance matrix is symmetric
    assert torch.allclose(cov, cov.T)
    assert torch.allclose(cov_mu, cov_mu.T)
    # Check that covariance matrix is positive semi-definite
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals_mu = torch.linalg.eigvalsh(cov_mu)
    assert torch.all(eigvals >= 0)
    assert torch.all(eigvals_mu >= 0)


def test_emp_diag_cov_mat():
    x = torch.randn(32, 7)
    mean = x.mean(dim=0)
    cov = utils.empirical_diag_cov_mat(x)
    cov_mu = utils.empirical_diag_cov_mat(x, mu=mean)
    assert torch.allclose(cov, cov_mu)
    assert cov.shape == (7, 7)
    assert cov_mu.shape == (7, 7)
    # Check that covariance matrix is symmetric
    assert torch.allclose(cov, cov.T)
    assert torch.allclose(cov_mu, cov_mu.T)
    # Check that covariance matrix is positive semi-definite
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals_mu = torch.linalg.eigvalsh(cov_mu)
    assert torch.all(eigvals >= 0)
    assert torch.all(eigvals_mu >= 0)
