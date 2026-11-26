import torch
from torch import Tensor, nn

from ..cometric.base_class import CoMetric
from .base_class import FinslerMetric


class MatsumotoMetrics(FinslerMetric):
    """
    Matsumoto metrics with a fixed base metric and a variable 1-form.

    The 1-form must verify the condition that the resulting Matsumoto metric is positive.
    It is up to the user to ensure this condition is satisfied.

    Parameters:
    -----------
    alpha_inv : CoMetric
        Base cometric to use for the Matsumoto metric.
    beta : nn.Module
        1-form to use for the Matsumoto metric.
    """

    def __init__(self, alpha_inv: CoMetric, beta: nn.Module):
        super().__init__()
        self.alpha_inv = alpha_inv
        self.beta = beta

    def forward(self, x: Tensor, v: Tensor):
        """Compute F(x,v) = alpha**2 / (alpha - beta)"""
        alpha = self.alpha_inv.metric(x, v)  # norm of v w.r.t. alpha
        beta = self.beta(x, v)
        return alpha**2 / (alpha - beta)  # Matsumoto metric formula


class SlopeMetrics(FinslerMetric):
    """
    Computes F(x,v)= alpha**2 / (alpha - beta)
    where alpha and beta are given in "The geometry on the slope of a mountain"
    see : http://arxiv.org/abs/1811.02123
    Slope metrics are Matsumoto metrics derived from
    a height map.

    Parameters:
    -----------
    f : nn.Module (N,2)-> (N,)
        Function that takes in points on the manifold and outputs a scalar value.
        This function represents the height map. To define a valid metric,
        The partial derivatives of f are required to verify f_x^2 + f_y^2 < 1/3 everywhere.
    """

    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f
        self.f_no_batch = lambda x: self.f(x.unsqueeze(0)).squeeze(0)

    def df_(self, x: Tensor) -> Tensor:
        """
        Computes the gradient of the height map f at points x.

        Parameters:
        -----------
        x : Tensor (b,2)
            Points in the manifold

        Returns:
        -------
        df : Tensor (b,2)
            Gradient of f at points x
        """
        f_x = self.f(x)
        df = torch.autograd.grad(f_x.sum(), x, create_graph=True)[0]
        return df

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        df = self.df_(x)
        df_dx, df_dy = df[:, 0], df[:, 1]

        alpha = (
            (1 + df_dx**2) * v[:, 0] ** 2
            + (1 + df_dy**2) * v[:, 1] ** 2
            + 2 * df_dx * df_dy * v[:, 0] * v[:, 1]
        ).sqrt()
        beta = df_dx * v[:, 0] + df_dy * v[:, 1]
        F = alpha**2 / (alpha - beta)
        return F
