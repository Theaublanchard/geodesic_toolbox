from typing import Self

import torch
from torch import Tensor


class CoMetric(torch.nn.Module):
    """
    Abstract class for cometrics.
    A cometric is here a function that takes a (batch of) point and returns the cometric tensor at that point.

    Parameters:
    -----------
    is_diag : bool
        If True, the cometric is diagonal and the forward method returns only the diagonal elements.
    """

    def __init__(self, is_diag: bool = False):
        super().__init__()
        self.is_diag = is_diag

    def legendre_transform(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Computes the Legendre transform of a batch of points q at tangent vectors v.
        That is, it computes p = G(q) v for a batch of points q at tangent vectors v.

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points
        v : Tensor (b, d)
            Batch of tangent vectors

        Returns:
        -------
        p : Tensor (b, d)
            Batch of momenta
        """
        G = self.metric_tensor(q)
        if not self.is_diag:
            return torch.einsum("bij,bj->bi", G, v)
        else:
            return G * v

    def inv_legendre_transform(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Computes the inverse Legendre transform of a batch of points q at momenta p.
        That is, it computes v = G^-1(q) p for a batch of points q at momenta p.

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points
        p : Tensor (b, d)
            Batch of momenta

        Returns:
        -------
        v : Tensor (b, d)
            Batch of tangent vectors
        """
        G_inv = self.cometric_tensor(q)
        if not self.is_diag:
            return torch.einsum("bij,bj->bi", G_inv, p)
        else:
            return G_inv * p

    def inv_logdet(self, q: Tensor) -> Tensor:
        """
        Computes log(det(G^-1(q))) for a batch of points q

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        -------
        res : Tensor (b,)
            log(det(G^-1(q)))
        """
        G_inv = self.cometric_tensor(q)
        if not self.is_diag:
            return torch.logdet(G_inv)
        else:
            return torch.sum(torch.log(G_inv), dim=1)

    def logdet(self, q: Tensor) -> Tensor:
        """
        Computes log(det(G(q))) for a batch of points q

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        --------
        res : Tensor (b,)
            log(det(G(q)))
        """
        G = self.metric_tensor(q)
        if not self.is_diag:
            return torch.logdet(G)
        else:
            return torch.sum(torch.log(G), dim=1)

    def cometric_tensor(self, q: Tensor) -> Tensor:
        """
        Computes G^-1(q) for a batch of points q

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        --------
        res : Tensor (b, d, d)
            Inverse metric tensor
            or Tensor (b, d) if is_diag is True
        """
        return self.forward(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        """
        Computes G(q) for a batch of points q

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        --------
        res : Tensor (b, d, d)
            Metric tensor
            or Tensor (b, d) if is_diag is True
        """
        if not self.is_diag:
            return self.cometric_tensor(q).inverse()
        else:
            return 1 / self.cometric_tensor(q)

    def dot(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Computes u^T G(q) v for a batch of points q at tangent vectors u and v

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points
        u : Tensor (b, d)
            First tangent vector
        v : Tensor (b, d)
            Second tangent vector

        Returns:
        -----------
        res : Tensor (b,)
            u^T G(q) v
        """
        G = self.metric_tensor(q)
        if not self.is_diag:
            return torch.einsum("bi,bij,bj->b", u, G, v)
        else:
            return torch.sum(u * G * v, dim=1)

    def inv_dot(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Computes u^T G_inv(q) v for a batch of points q at tangent vectors u and v

        Parameters:
        q : Tensor (b,d)
            Batch of points
        u : Tensor (b,d)
            First tangent vector
        v : Tensor (b,d)
            Second tangent vector

        Returns:
        -------
        res : Tensor (b,)
            u^T G_inv(q) v
        """
        G_inv = self.cometric_tensor(q)
        if self.is_diag:
            return torch.sum(u * G_inv * v, dim=1)
        else:
            return torch.einsum("bi,bij,bj->b", u, G_inv, v)

    def metric(self, q: Tensor, p: Tensor) -> Tensor:
        """Computes the norm sqrt(p^TG(q)p) for a batch of tangent vectors p at points q

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        p : Tensor (b, d)
            Batch of tangent vectors

        Returns:
        -------
        res : Tensor (b,)
            sqrt(p^TG(q)p)
        """
        return self.dot(q, p, p).sqrt()

    def energy(self, q: Tensor, p: Tensor) -> Tensor:
        """Computes p^TG(q)p for a batch of tangent vectors p at points q

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        p : Tensor (b, d)
            Batch of tangent vectors

        Returns:
        -------
        res : Tensor (b,)
            p^TG(q)p
        """
        return self.dot(q, p, p)

    def cometric(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Computes the dual norm sqrt(v^T G_inv(q) v) for a batch of points q at momenta v

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        v : Tensor (b, d)
            Batch of momenta
        Returns:
        -------
        res : Tensor (b,)
            sqrt(v^T G_inv(q) v)
        """
        return self.inv_dot(q, v, v).sqrt()

    def dual_energy(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Computes the dual energy v^T G_inv(q) v for a batch of points q at momenta v

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        v : Tensor (b, d)
            Batch of momenta

        Returns:
        -------
        res : Tensor (b,)
            v^T G_inv(q) v
        """
        return self.inv_dot(q, v, v)

    def forward(self, q: Tensor) -> Tensor:
        """Computes the cometric tensor G^-1(q) for a batch of points q

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        -------
        res : Tensor (b, d, d)
            Inverse metric tensor
            or Tensor (b, d) if is_diag is True
        """
        raise NotImplementedError

    def angle(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Computes the angle between two vectors u and v at a point q.

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        u : Tensor (b, d)
            First tangent vector
        v : Tensor (b, d)
            Second tangent vector

        Returns:
        -------
        angle : Tensor (b,)
            Angle between u and v at q
        """
        eps = 1e-8  # small value to avoid division by zero
        u_norm = self.metric(q, u)
        v_norm = self.metric(q, v)
        uv = self.dot(q, u, v)
        cos_angle = uv / (u_norm * v_norm + eps)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # clamp to avoid NaN
        angle = torch.acos(cos_angle)
        return angle

    def __add__(self, other: Self) -> Self:
        if isinstance(other, CoMetric):
            return SumOfCometric(self, other)
        else:
            raise TypeError(f"Cannot add {type(other)} to CoMetric")

    def __mul__(self, other: object) -> Self:
        if isinstance(other, (int, float)):
            return ScaledCometric(self, other)
        else:
            raise TypeError(f"Cannot multiply {type(other)} to CoMetric")

    def __rmul__(self, other: object) -> Self:
        return self.__mul__(other)

    def eye(self, x):
        """
        Helper function to create a batch of identity matrices on
        the proper device and with the proper dtype

        Parameters:
        ----------
        x : Tensor (b, d)
            Batch of points

        Returns:
        -------
        id : Tensor (b, d, d)
            Batch of identity matrices
            or (b, d) if is_diag is True
        """
        B, dim = x.shape
        if self.is_diag:
            return torch.ones_like(x)
        else:
            id = torch.eye(dim, dtype=x.dtype, device=x.device).unsqueeze(0)
            id = id.repeat(B, 1, 1)
            return id


class SumOfCometric(CoMetric):
    """
    Sum of two cometrics.

    WARNING : the sum is done at the metric tensor level.
        NOT on the cometric tensor level. This is because the sum of two cometrics is not a cometric in general.

    Parameters:
    -----------
    cometric1: CoMetric
        First cometric tensor
    cometric2: CoMetric
        Second cometric tensor
    beta : float
        Scaling factor for the sum of cometrics
    """

    def __init__(self, cometric1: CoMetric, cometric2: CoMetric):
        super().__init__()
        self.cometric1 = cometric1
        self.cometric2 = cometric2

        if self.cometric1.is_diag and self.cometric2.is_diag:
            self.is_diag = True
        else:
            self.is_diag = False

    def metric_tensor(self, q: Tensor) -> Tensor:
        G_1 = self.cometric1.metric_tensor(q)
        G_2 = self.cometric2.metric_tensor(q)
        if not self.cometric1.is_diag and self.cometric2.is_diag:
            G_2 = torch.diag_embed(G_2)
        elif self.cometric1.is_diag and not self.cometric2.is_diag:
            G_1 = torch.diag_embed(G_1)
        return G_1 + G_2

    def dot(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return self.cometric1.dot(q, u, v) + self.cometric2.dot(q, u, v)

    def forward(self, q: Tensor) -> Tensor:
        if self.is_diag:
            G = self.metric_tensor(q)
            return 1 / G
        else:
            G = self.metric_tensor(q)
            return torch.linalg.inv(G)


class ScaledCometric(CoMetric):
    """
    Cometric that is a scaled version of another cometric.
    The new metric is G'(q) = 1/scale * G(q) where G(q) is the metric of the original cometric.

    Parameters:
    -----------
    cometric : CoMetric
        The cometric to scale
    scale : float
        Scaling factor
    """

    def __init__(self, cometric: CoMetric, scale: float):
        super().__init__()
        self.cometric_ = cometric
        self.scale = scale
        self.is_diag = cometric.is_diag

    def forward(self, q: Tensor) -> Tensor:
        return self.scale * self.cometric_.forward(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        return 1 / self.scale * self.cometric_.metric_tensor(q)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class IdentityCoMetric(CoMetric):
    """
    Cometric that is the (scaled) identity matrix such that:
    G(q) = 1/coscale * I and G^-1(q) = coscale * I for all q.

    Parameters:
    -----------
    coscale : float
        Scaling factor for the cometric. Set to 1 for the identity cometric
    """

    def __init__(self, coscale: float = 1, is_diag=True):
        super().__init__(is_diag=is_diag)
        self.coscale = coscale

    def forward(self, q: Tensor) -> Tensor:
        return self.coscale * self.eye(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        return 1 / self.coscale * self.eye(q)

    def extra_repr(self) -> str:
        return f"coscale={self.coscale}"
