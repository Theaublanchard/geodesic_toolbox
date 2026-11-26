"""Shared dummy building blocks for the Finsler metric tests."""

import torch

from geodesic_toolbox import cometric


class DummyAccuracyPredictor(torch.nn.Module):
    """
    Dummy amplitude predictor that always returns 0.5.

    The amplitude stays strictly below 1 so that the resulting 1-form
    satisfies the Randers condition ||beta * omega|| < 1 (w.r.t. the base
    cometric) and F(x, v) > 0 for all directions v.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.ones(z.shape[0], 1, device=z.device)


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

    Parameters
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
