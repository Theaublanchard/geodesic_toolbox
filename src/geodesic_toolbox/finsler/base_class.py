import torch
from torch import Tensor, nn


class FinslerMetric(nn.Module):
    """
    Finsler metric base class
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Finsler metric at point x in the direction v.

        Parameters:
        -----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        --------
        F : Tensor (b,)
            Finsler metric values at (x,v)
        """
        raise NotImplementedError("FinslerMetric is an abstract class")

    def fundamental_tensor(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Compute the fundamental tensor of the Finsler metric at point x in the direction v.

        Parameters:
        -----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        --------
        G : Tensor (b,d,d)
            Fundamental tensor of the Finsler metric at (x,v)
        """

        def g(x1: Tensor, v2: Tensor) -> Tensor:
            F = lambda q, p: self.forward(q.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
            g_hessian = torch.func.hessian(lambda v1: 1 / 2 * F(x1, v1) ** 2)
            return g_hessian(v2)

        G = torch.vmap(g)
        return G(x, v)

    def inverse_fundamental_tensor(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Compute the inverse of the fundamental tensor of the Finsler metric at point x in the direction v.

        Parameters:
        -----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        --------
        G_inv : Tensor (b,d,d)
            Inverse of the fundamental tensor of the Finsler metric at (x,v)
        """
        G = self.fundamental_tensor(x, v)
        G_inv = torch.linalg.inv(G)
        return G_inv


class ToyFinslerMetric(FinslerMetric):
    """
    Compute F(x,v) = 1/|v| * (1 + lbd^2 * |x|^2 + lbd^2 * <x,v>^2 / |v|^2)
    This is a valid metric see:
    https://doi.org/10.1016/j.aim.2005.06.007

    Parameters:
    -----------
    lbd : float
        Regularization parameter
    """

    def __init__(self, lbd: float = 1):
        super().__init__()
        self.lbd = lbd
        self.lbd2 = lbd**2

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        x_norm = torch.linalg.vector_norm(x, dim=-1)
        v_norm = torch.linalg.vector_norm(v, dim=-1)
        xv = torch.einsum("bi,bi->b", x, v)
        F = 1 / (v_norm + 1e-8) * (1 + self.lbd2 * x_norm**2 + self.lbd2 * xv**2)
        return F
