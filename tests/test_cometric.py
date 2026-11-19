import pytest
import torch

from geodesic_toolbox import cometric


############################################################
# Identity cometric
############################################################
def test_identity_instantiate():
    print(cometric.IdentityCoMetric())


def test_identity_cometric_basic():
    # Test that IdentityCoMetric returns correct values
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
    metric_val = cm.metric(x, p)
    assert torch.allclose(metric_val, 1 / coscale * torch.sum(p**2, dim=1))
    v = torch.randn(batch, dim)
    cometric_val = cm.cometric(x, v)
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
    out = cm.cometric(x, v)
    true_out = torch.sum(v**2, dim=1)
    assert torch.allclose(out, true_out)


def test_invdot_diagonal():
    cm = cometric.IdentityCoMetric()
    x = torch.randn(4, 3)
    p = torch.randn(4, 3)
    out = cm.metric(x, p)
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
def test_sum_cometric_all_diag():
    cm_1 = cometric.IdentityCoMetric()
    cm_2 = cometric.IdentityCoMetric()
    x = torch.randn(4, 3)
    sum_cm = cm_1 + cm_2
    out = sum_cm.cometric_tensor(x)
    out_inv = sum_cm.metric_tensor(x)
    assert torch.allclose(out_inv, 0.5 * torch.ones_like(x))
    assert torch.allclose(out, 2 * torch.ones_like(x))


def test_sum_cometric_mixed_1():
    cm_2 = cometric.IdentityCoMetric()
    cm_1 = cometric.PointCarreCoMetric()
    x = torch.zeros(1, 2)
    sum_cm = cm_1 + cm_2
    out = sum_cm.cometric_tensor(x)
    out_inv = sum_cm.metric_tensor(x)
    assert torch.allclose(out, (1 + 1 / 4) * torch.eye(2).unsqueeze(0))
    assert torch.allclose(out_inv, 1 / (1 + 1 / 4) * torch.eye(2).unsqueeze(0))


def test_sum_cometric_mixed_2():
    cm_1 = cometric.IdentityCoMetric()
    cm_2 = cometric.PointCarreCoMetric()
    x = torch.zeros(1, 2)
    sum_cm = cm_1 + cm_2
    out = sum_cm.cometric_tensor(x)
    out_inv = sum_cm.metric_tensor(x)
    assert torch.allclose(out, (1 + 1 / 4) * torch.eye(2).unsqueeze(0))
    assert torch.allclose(out_inv, 1 / (1 + 1 / 4) * torch.eye(2).unsqueeze(0))


def test_sum_cometric_none_diag():
    cm_1 = cometric.PointCarreCoMetric()
    cm_2 = cometric.PointCarreCoMetric()
    x = torch.zeros(1, 2)
    sum_cm = cm_1 + cm_2
    out = sum_cm.cometric_tensor(x)
    out_inv = sum_cm.metric_tensor(x)
    assert torch.allclose(out, (1 / 4 + 1 / 4) * torch.eye(2).unsqueeze(0))
    assert torch.allclose(out_inv, 1 / (1 / 4 + 1 / 4) * torch.eye(2).unsqueeze(0))


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
    out = cm.cometric(x, v)
    norm_q_sqr = torch.linalg.vector_norm(x, dim=1) ** 2
    scalar = 1 / 4 * (1 - norm_q_sqr) ** 2
    true_out = scalar * torch.sum(v**2, dim=1)
    assert torch.allclose(out, true_out)


def test_invdot_not_diag():
    x = 2 * torch.rand(3, 2) - 1
    p = torch.randn(3, 2)
    cm = cometric.PointCarreCoMetric()
    out = cm.metric(x, p)
    norm_q_sqr = torch.linalg.vector_norm(x, dim=1) ** 2
    scalar = 4 / (1 - norm_q_sqr) ** 2
    true_out = scalar * torch.sum(p**2, dim=1)
    assert torch.allclose(out, true_out)


############################################################
# Centroids cometric
############################################################
def test_centroids_cometric_instantiante():
    print(cometric.CentroidsCometric())


def test_centroids_cometric_basic_not_diag():
    # Test CentroidsCometric interpolation
    dim = 2
    K = 4
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.stack([torch.eye(dim) for _ in range(K)])
    temperature = 1.0
    reg_coef = 1e-3
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        temperature=temperature,
        reg_coef=reg_coef,
        metric_weight=False,
        K=4,
    )
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        temperature=temperature,
        reg_coef=reg_coef,
        metric_weight=False,
        K=-1,
    )
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        temperature=temperature,
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


def test_centroids_cometric_basic_diag():
    # Test CentroidsCometric interpolation
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.ones_like(centroids)
    temperature = 1.0
    reg_coef = 1e-3
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        temperature=temperature,
        reg_coef=reg_coef,
        metric_weight=False,
    )
    # Test output shape
    x = torch.randn(3, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (3, dim)
    # Should be positive definite (diagonal dominant)
    assert torch.all(out > 0)


def test_centroids_cometric_metric_weight_true_not_diag():
    # Test CentroidsCometric with metric_weight=True
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.stack([torch.eye(dim) for _ in range(K)])
    temperature = 1.0
    reg_coef = 1e-3
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        temperature=temperature,
        reg_coef=reg_coef,
        metric_weight=True,
    )
    x = torch.randn(2, dim)
    out = cm.cometric_tensor(x)
    assert out.shape == (2, dim, dim)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_centroids_cometric_metric_weight_true_diag():
    # Test CentroidsCometric with metric_weight=True
    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    cometric_centroids = torch.ones_like(centroids)
    temperature = 1.0
    reg_coef = 1e-3
    cm = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        temperature=temperature,
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
    temperature = 1.0
    cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        temperature=temperature,
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
            temperature=1.0,
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
            temperature=1.0,
            reg_coef=1e-3,
            metric_weight=True,
            K=K,
        )


def _test_load_centroids(diag=True):
    # Test loading centroids from file
    import os


    K = 4
    dim = 2
    centroids = torch.randn(K, dim)
    if diag:
        cometric_centroids = torch.ones_like(centroids)
    else:
        cometric_centroids = torch.stack([torch.eye(dim) for _ in range(K)])
    temperature = 1.0
    reg_coef = 1e-3
    cm_ = cometric.CentroidsCometric(
        centroids=centroids,
        cometric_centroids=cometric_centroids,
        temperature=temperature,
        reg_coef=reg_coef,
        metric_weight=False,
        K=K,
    )
    filepath = "test_centroids.pth"
    torch.save(cm_.state_dict(), filepath)
    cm = cometric.CentroidsCometric(
        centroids=None,
        cometric_centroids=None,
        temperature=temperature,
        reg_coef=reg_coef,
        metric_weight=False,
        K=K,
    )
    state_dict = torch.load(filepath)
    cm.load_state_dict(state_dict)
    os.remove(filepath)


def test_load_centroids():
    _test_load_centroids(diag=True)
    _test_load_centroids(diag=False)


############################################################
# DiagonalCometricModel cometric
############################################################
def test_forward_pass_DiagonalCometricModel():
    in_dim = 2
    hidden_dim = 10
    latent_dim = 3
    lbd = 1.0
    model = cometric.DiagonalCometricModel(
        in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, lbd=lbd
    )
    x = torch.randn(4, in_dim)
    out = model.cometric_tensor(x)
    assert out.shape == (4, latent_dim)
    assert torch.all(out > 0)
    inv_out = model.metric_tensor(x)
    assert inv_out.shape == (4, latent_dim)
    assert torch.all(inv_out > 0)
    print(model)


############################################################
# CometricModel cometric
############################################################
def test_forward_pass_CometricModel():
    in_dim = 2
    hidden_dim = 10
    latent_dim = 3
    lbd = 1.0
    model = cometric.CometricModel(
        input_dim=in_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lbd=lbd,
    )
    x = torch.randn(4, in_dim)
    out = model.cometric_tensor(x)
    assert out.shape == (4, latent_dim, latent_dim)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)
    inv_out = model.metric_tensor(x)
    assert inv_out.shape == (4, latent_dim, latent_dim)
    for i in range(inv_out.shape[0]):
        eigvals = torch.linalg.eigvalsh(inv_out[i])
        assert torch.all(eigvals > 0)
    print(model)


############################################################
# SmallConvCometricModel cometric
############################################################
def test_forward_pass_SmallConvCometricModel():
    latent_dim = 13
    n_channels = 3
    width = 224
    lbd = 0.01
    model = cometric.SmallConvCometricModel(
        latent_dim=latent_dim,
        n_channels=n_channels,
        width=width,
        lbd=lbd,
    )
    x = torch.randn(4, n_channels, width, width)
    out = model.cometric_tensor(x)
    assert out.shape == (4, latent_dim, latent_dim)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


############################################################
# Cometric_MLP cometric
############################################################
def test_forward_pass_Cometric_MLP():
    input_dim = (5, 4, 3)
    latent_dim = 13
    lbd = 0.001
    cm = cometric.Cometric_MLP(
        input_dim=input_dim,
        latent_dim=latent_dim,
        lbd=lbd,
    )
    x = torch.randn(2, *input_dim)
    out = cm(x)
    assert out.shape == (2, latent_dim, latent_dim)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


############################################################
# SoftAbsCometric cometric
############################################################
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


############################################################
# Utils
############################################################
def test_emp_cov_mat():
    x = torch.randn(32, 7)
    mean = x.mean(dim=0)
    cov = cometric.empirical_cov_mat(x)
    cov_mu = cometric.empirical_cov_mat(x, mu=mean)
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
    cov = cometric.empirical_diag_cov_mat(x)
    cov_mu = cometric.empirical_diag_cov_mat(x, mu=mean)
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


############################################################
# FisherRaoCometric cometric
############################################################
def log_likelihood(x, mu):
    distr = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=torch.eye(mu.shape[-1]))
    return distr.log_prob(x)


def test_fisher_rao_cometric():
    cm = cometric.FisherRaoCometric(
        log_likelihood=log_likelihood,
        reg_coef=1e-3,
        softabs_alpha=1e5,
        N_pts=100,
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_fisher_rao_cometric_softabs():
    fim = cometric.FisherRaoCometric(
        log_likelihood=log_likelihood,
        N_pts=100,
    )
    theta = torch.randn(4, 3)
    out = fim.cometric_tensor(theta)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_diffeometric_nobase():
    class DummyDiffeoMorphism(torch.nn.Module):
        def forward(self, q):
            return q**3
    cm = cometric.DiffeoCometric(
        diffeo = DummyDiffeoMorphism(),
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)

def test_diffeometric_base():
    class DummyDiffeoMorphism(torch.nn.Module):
        def forward(self, q):
            return q**3
    cm = cometric.DiffeoCometric(
        diffeo = DummyDiffeoMorphism(),
        use_id=False,
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)

def test_diffeometric_no_vmap():
    class DummyDiffeoMorphism(torch.nn.Module):
        def forward(self, q):
            return q**3
    cm = cometric.DiffeoCometric(
        diffeo = DummyDiffeoMorphism(),
        vmap_ok=False,
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)

def test_diffeometric_with_jacobian():
    
    class DummyDiffeoMorphism(torch.nn.Module):
        def forward(self, q):
            return q**3

        def jacobian(self,x):
            batch_size, dim = x.shape
            J = torch.zeros(batch_size, dim, dim)
            for i in range(dim):
                J[:, i, i] = 3 * x[:, i] ** 2
            return J
        
    cm = cometric.DiffeoCometric(
        diffeo = DummyDiffeoMorphism(),
    )
    x = torch.randn(4, 3)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)

def test_lifted_cometric():
    base_cm = cometric.IdentityCoMetric()
    class DummyClassifier(torch.nn.Module):
        def forward(self, q):
            return 1/(q[:,0]+1e-3)
    h = DummyClassifier()
    lifted_cm = cometric.LiftedCometric(
        base_cometric=base_cm,
        h=h,
        beta=1.0,
    )
    x = torch.randn(4, 3)
    out = lifted_cm.cometric_tensor(x)
    assert out.shape == (4, 3, 3)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


def test_funcheightmap_cometric():
    def f(x,y):
        return torch.sin(x) * torch.cos(y)
    cm = cometric.FunctionnalHeightMapCometric(func=f)
    x = torch.randn(4, 2)
    out = cm.cometric_tensor(x)
    assert out.shape == (4, 2, 2)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)