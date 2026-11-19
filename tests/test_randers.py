import torch

from geodesic_toolbox import cometric


############################################################
# Randers metric
############################################################
class DummyAccuracyPredictor(torch.nn.Module):
    """
    Dummy accuracy predictor that always returns 1.0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.ones(z.shape[0], 1, device=z.device)


class DummyDirectionPredictor(torch.nn.Module):
    """
    Dummy direction predictor that always returns a random direction.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.rand_dir = torch.randn(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        dir = self.rand_dir.repeat(z.shape[0], 1)
        return dir


class Omega(torch.nn.Module):
    """
    Network to wrap the direction and amplitude of the Randers 1-form.
    The direction is normalized by the cometric so that the norm
    is only a function of the amplitude.

    Parameters:
    ----------
    direction : torch.nn.Module
        A neural network that predicts the direction of the Randers 1-form.
    amplitude : torch.nn.Module
        A neural network that predicts the amplitude of the Randers 1-form.
    cometric : CoMetric
        The cometric used to normalize the direction.
    """

    def __init__(
        self,
        direction: torch.nn.Module,
        amplitude: torch.nn.Module,
        cometric: cometric.CoMetric,
    ):
        super().__init__()
        self.direction = direction
        self.amplitude = amplitude
        self.cometric = cometric

    def forward(self, X):
        rho = self.amplitude(X).clip(0.0, 1.0)
        omega = self.direction(X)
        omega_norm = self.cometric.cometric(X, omega).sqrt()
        omega = omega / (omega_norm[:, None] + 1e-8)
        return rho * omega


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
    randers = cometric.RandersMetrics(
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
    randers = cometric.RandersMetrics(
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
    randers = cometric.RandersMetrics(
        base_cometric=cometric_model,
        omega=omega_model,
        beta=1.0,
        use_grad_g=True,
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
    randers = cometric.RandersMetrics(
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
    randers = cometric.RandersMetrics(
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
    randers = cometric.RandersMetrics(
        base_cometric=cometric_model,
        omega=omega_model,
        beta=1.0,
    )
    fund = randers.fundamental_tensor(x, v)
    inv_fund = randers.inverse_fundamental_tensor(x, v)
    assert torch.allclose(fund.inverse(), inv_fund, atol=1e-5)


############################################################
# SlopeMetrics metric
############################################################
def height_map(x: torch.Tensor) -> torch.Tensor:
    """A simple height map function."""
    return torch.sin(x[:, 0]) * torch.cos(x[:, 1])


def test_SlopeMetrics():
    model = cometric.SlopeMetrics(f=height_map)
    x = torch.randn(10, 2)
    v = torch.randn(10, 2)
    metric_val = model(x, v)
    assert metric_val.shape == (10,)
    assert (metric_val > 0).all()


############################################################
# ToyFinslerMetric metric
############################################################
def test_ToyFinslerMetric():
    model = cometric.ToyFinslerMetric()
    x = torch.randn(10, 3)
    v = torch.randn(10, 3)
    metric_val = model(x, v)
    assert metric_val.shape == (10,)
    assert (metric_val > 0).all()


############################################################
# MatsumotoMetrics metric
############################################################
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
    model = cometric.MatsumotoMetrics(alpha_inv=cometric_model, beta=beta)
    x = torch.randn(10, 3)
    v = torch.randn(10, 3)
    metric_val = model(x, v)
    assert metric_val.shape == (10,)
    assert (metric_val > 0).all()
