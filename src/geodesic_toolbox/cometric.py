import torch
from torch import Tensor
import torch.nn as nn
import weakref
import numpy as np
from tqdm import tqdm
import kmedoids
from einops import rearrange
from typing import Self

################################################################
# Utils
################################################################


def empirical_cov_mat(x: Tensor, mu: Tensor = None, eps: float = 1e-6) -> Tensor:
    """Computes the empirical covariance matrix of the data x.
    If mu is provided, the covariance is computed with respect to mu.
    Else the covariance is computed with respect to the mean of x.

    Parameters
    ----------
    x : Tensor (N,d)
        The data.
    mu : Tensor (d,)
        The mean of the data.
    eps : float
        A small value to add to the diagonal for numerical stability.

    Returns
    -------
    cov : Tensor (d,d)
        The covariance matrix.
    """
    if mu is None:
        mu = x.mean(dim=0)
    mu = mu[None, :]
    cov = (x - mu).T @ (x - mu) / (x.shape[0] - 1)
    cov += eps * torch.eye(x.shape[1], device=x.device)
    return cov


def empirical_diag_cov_mat(x: Tensor, mu: Tensor = None, eps: float = 1e-6) -> Tensor:
    """Computes the empirical covariance matrix of the data x.
    The matrix is here diagonal.
    If mu is provided, the covariance is computed with respect to mu.
    Else the covariance is computed with respect to the mean of x.

    Parameters
    ----------
    x : Tensor (N,d)
        The data.
    mu : Tensor (d,)
        The mean of the data.
    eps : float
        A small value to add to the diagonal for numerical stability.

    Returns
    -------
    cov : Tensor (d,d)
        The covariance matrix.
    """
    if mu is None:
        mu = x.mean(dim=0)
    mu = mu[None, :]
    var = torch.linalg.vector_norm(x - mu, dim=1).mean()
    cov = (var + eps) * torch.eye(x.shape[1], device=x.device)
    return cov


def mat_sqrt(A: Tensor) -> Tensor:
    """
    Compute the matrix square root of a positive definite matrix A.

    Parameters
    ----------
    A : Tensor (..., n, n)
        The matrix to compute the square root of.

    Returns
    -------
    Tensor (..., n, n)
        The matrix square root of A.
    """
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def SoftAbs(M: Tensor, alpha: float = 1e3) -> Tensor:
    """
    SoftAbs regularisation of a matrix M. It is used to ensure that the matrix is positive definite.
    This is especially useful when using the Fisher information matrix.
    Essentially, it is a soft version of the absolute value.

    To use around a sampler, just wrap your cometric in a SoftAbs :
    ```
    cometric = IdentityCoMetric()
    cometric = lambda x: SoftAbs(cometric(x))
    ```

    It is defined as:
    SoftAbs(M) = Q @ Diag(a_i * coth(alpha * a_i)) @ Q^T
    where M = Q @ Diag(a_i) @ Q^T is the eigendecomposition of M.

    Parameters
    ----------
    M : Tensor (..., n, n)
        The matrix to regularise.
    alpha : float
        The regularisation parameter.

    Returns
    -------
    Tensor (..., n, n)
        The regularised matrix.
    """
    D, Q = torch.linalg.eigh(M)
    D = D * 1 / torch.tanh(alpha * D)
    G = torch.bmm(torch.diag_embed(D), Q.mH)
    G = torch.bmm(Q, G)
    return G


################################################################
# Base Classes
################################################################


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
        u_norm = self.metric(q, u).sqrt()
        v_norm = self.metric(q, v).sqrt()
        uv = self.dot(q, u, v)
        cos_angle = uv / (u_norm * v_norm + eps)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # clamp to avoid NaN
        angle = torch.acos(cos_angle)
        return angle

    def __add__(self, other: Self) -> Self:
        if isinstance(other, CoMetric):
            return SumOfCometric(self, other)
        else:
            raise ValueError(f"Cannot add {type(other)} to CoMetric")

    def __mul__(self, other: object) -> Self:
        if isinstance(other, (int, float)):
            return ScaledCometric(self, other)
        else:
            raise ValueError(f"Cannot multiply {type(other)} to CoMetric")

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
            return G_1 + G_2
        elif self.cometric1.is_diag and not self.cometric2.is_diag:
            G_1 = torch.diag_embed(G_1)
            return G_1 + G_2
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
    Cometric that is the (scaled) identity matrix

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


class SoftAbsCometric(CoMetric):
    """
    Cometric that applies the SoftAbs regularisation to a base cometric.

    Parameters:
    -----------
    base_cometric : CoMetric
        The base cometric to regularise
    alpha : float
        Regularisation parameter for the SoftAbs
    """

    def __init__(self, base_cometric: CoMetric, alpha: float = 1e3):
        super().__init__()
        if base_cometric.is_diag:
            raise NotImplementedError("SoftAbs for diagonal cometrics not implemented yet")
        self.base_cometric = base_cometric
        self.alpha = alpha

    def metric_tensor(self, q: Tensor) -> Tensor:
        g = self.base_cometric.metric_tensor(q)
        g_soft = SoftAbs(g, self.alpha)
        return g_soft

    def forward(self, q: Tensor) -> Tensor:
        g_soft = self.metric_tensor(q)
        return torch.linalg.inv(g_soft)


################################################################
# Stand alone Cometrics
################################################################


class PointCarreCoMetric(CoMetric):
    """
    Cometric that is the pointcarre matrix, ie:
    G(x) = 0.25 * diag({1-||x||^2}^2)
    """

    def __init__(self):
        super().__init__()

    def forward(self, q: Tensor) -> Tensor:
        norm_q_sqr = torch.linalg.vector_norm(q, dim=1) ** 2
        scalar = (1 - norm_q_sqr) ** 2
        return 1 / 4 * scalar[:, None, None] * self.eye(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        norm_q_sqr = torch.linalg.vector_norm(q, dim=1) ** 2
        scalar = 1 / (1 - norm_q_sqr) ** 2
        return 4 * scalar[:, None, None] * self.eye(q)


################################################################
# Cometric from functions
################################################################


class FunctionnalHeightMapCometric(CoMetric):
    """
    Construct a cometric tensor from a parametric height map function.
    The metric tensor is simply  g_ij = <d_i r, d_j r> for r=(x,y,f(x,y)) where f is the height map function.
    for i,j in {x,y,z}.

    Parameters:
    -----------
    func : Callable
        The height map function such that z = func(x, y).
    reg : float
        Regularization parameter for the cometric tensor.
    """

    def __init__(self, func: callable, reg: float = 0):
        super().__init__()
        self.func = func
        self.reg = reg
        self.df_ = torch.func.jacrev(self.func, argnums=(0, 1))

    def get_dx_dy(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the partial derivatives of the height map function at points (x, y).

        Parameters:
        x : Tensor (B,)
            x-coordinates of the points
        y : Tensor (B,)
            y-coordinates of the points

        Returns:
        dx : Tensor (B,)
            Partial derivative with respect to x
        dy : Tensor (B,)
            Partial derivative with respect to y
        """
        dx, dy = self.df_(x, y)
        dx = dx.sum(dim=1)
        dy = dy.sum(dim=1)
        return dx, dy

    def metric_tensor(self, q: Tensor) -> Tensor:
        x, y = q.T
        df_dx, df_dy = self.get_dx_dy(x, y)

        # Compute the metric tensor g_ij = <d_i r, d_j r> ( r=(x,y,f(x,y)) )
        g = torch.zeros(x.shape[0], 2, 2, device=x.device, dtype=x.dtype)
        g[:, 0, 0] = 1 + df_dx**2
        g[:, 0, 1] = df_dx * df_dy
        g[:, 1, 0] = df_dx * df_dy
        g[:, 1, 1] = 1 + df_dy**2

        g += self.reg * self.eye(q)
        return g

    def forward(self, q: Tensor) -> Tensor:
        g = self.metric_tensor(q)
        g_inv = torch.linalg.inv(g)
        return g_inv


class PullBackCometric(CoMetric):
    """
    Class for the cometric given by the pullback of a diffeomorphism between manifolds.
    If J_f is the jacobian of the diffeomorphism f and G the base metric on the target manifold, the metric is given by:
    g(x) = J_f(x)^T @ G(f(x)) @ J_f(x) + reg_coef * I

    Parameters:
    -----------
    diffeo: torch.nn.Module
        Neural network model. It should have signature (B,d) -> (B,...) (ie flattened input)
        Don't forget to put in eval() mode if you can to save memory.
    base_cometric: CoMetric
        The base cometric. Default to Euclidean cometric.
    method: str
        Method to compute the jacobian of the diffeomorphism. Can be :
            - "finite_difference" : uses finite differences to compute the jacobian.
                This method is relatively fast and memory efficient. Default method.
            - "loop_jvp" : uses a loop over the batch to compute the jacobian using jvp.
                This method is relatively fast, memory efficient and exact. Recommended for low-dimensional outputs.
                But slightly slower than 'finite_difference'.
            - "jacfwd" : uses vmap(jacfwd). Recommended for low to high dimensional outputs
                This method is exact and always the fastest. The problem is that it can be very memory intensive for high-dimensional outputs.
                If memory issues, specify a chunk_size to compute the jacobian by batches.
            - "jacrev" : uses vmap(jacrev). Recommended for high to low dimensional outputs
                If memory issues, specify a chunk_size to compute the jacobian by batches.
            - "autograd" : uses autograd to compute the jacobian. Mega slow but precise, not recommended.
            - "jacobian_method" : uses the method 'jacobian' of the diffeomorphism if it exists.
                This method should have signature (B,d) -> (B,d_out,d).
    reg_coef: float
        Regularization coefficient for the metric
    chunk_size: int
        Chunk size to use for computing the jacobian. Specify a value if running in memory issues.
        Used only for the "jacfwd", "jacrev" methods with vmap and "finite_difference" method to batch several dimensions of the output.
    eps: float
        Small value to compute the jacobian using finite differences approximation.

    Note if method=='jacobian_method' it should have signature (B,d) -> (B,d_out,d)

    Important remark : the current implementation of the jacobian via autograd can be very slow for high-dimensional outputs.
    Moreover it doesn't support higher order derivatives, eg for christoffel symbols computation.
    """

    def __init__(
        self,
        diffeo: torch.nn.Module,
        base_cometric: CoMetric = IdentityCoMetric(is_diag=False),
        method: str = "finite_difference",
        reg_coef: float = 1e-3,
        chunk_size: int = 4,
        eps: float = 1e-4,
    ):
        super().__init__()
        valid_methods = [
            "jacobian_method",
            "jacfwd",
            "jacrev",
            "autograd",
            "finite_difference",
        ]

        self.diffeo = diffeo
        self.base_cometric = base_cometric
        self.eps = eps
        self.reg_coef = reg_coef
        self.method = method
        self.chunk_size = chunk_size
        self.no_batch_forward = lambda x: self.diffeo(x.unsqueeze(0)).flatten()

        if method == "jacobian_method":
            if hasattr(self.diffeo, "jacobian"):
                self.jacobian = self.diffeo.jacobian
            else:
                raise ValueError("Diffeomorphism does not have a 'jacobian' method")
        elif method == "jacrev":
            self.no_batch_forward = lambda x: self.diffeo(x.unsqueeze(0)).flatten()
            self.jacobian_ = torch.func.jacrev(self.no_batch_forward)
            self.jacobian = torch.vmap(self.jacobian_, chunk_size=chunk_size)
        elif method == "jacfwd":
            self.no_batch_forward = lambda x: self.diffeo(x.unsqueeze(0)).flatten()
            self.jacobian_ = torch.func.jacfwd(self.no_batch_forward)
            self.jacobian = torch.vmap(self.jacobian_, chunk_size=chunk_size)
        elif method == "autograd":
            self.jacobian = self.jacobian_autograd
        elif method == "loop_jvp":
            self.jacobian = self.jacobian_forward_mode
        elif method == "finite_difference":
            self.jacobian = self.jacobian_finite_difference
        else:
            raise ValueError(f"Invalid method {method}. Valid methods are {valid_methods}")

    @torch.enable_grad()
    def jacobian_autograd(self, x: Tensor) -> Tensor:
        """
        Computes the jacobian of the diffeomorphism at the points x using autograd.

        Parameters:
        -----------
        x: Tensor (B, d)
            Batch of points where to compute the pullback metric

        Returns:
        --------
        jacobian : Tensor (B,d_out,d)
            Batch of jacobians
        """
        x.requires_grad_(True)
        d = x.shape[1]
        y_flat = self.diffeo(x).flatten(start_dim=1)  # (B, hw)
        B, hw = y_flat.shape

        J = torch.zeros(B, hw, d, device=x.device, dtype=x.dtype)
        pbar = tqdm(range(hw), desc="Computing pullback metric via autograd", leave=False)
        for i in pbar:
            pbar.set_postfix({"Jacobian column": f"{i+1}/{hw}"})
            grad_i = torch.autograd.grad(
                y_flat[:, i].sum(),  # sum over batch to get batch gradients
                x,
                retain_graph=(i < hw - 1),
                create_graph=False,
                # change this line if higher order derivatives are needed
                # eg christoffel symbols
                # tips : it will crash of OOM. good luck
            )[0]
            J[:, i, :] = grad_i
        return J

    def jacobian_finite_difference(self, x: Tensor) -> Tensor:
        """
        Computes the jacobian of the diffeomorphism at the points x using finite differences.
        More precisely , for each point x_i in the batch, and each dimension j,
        we compute the j-th column of the jacobian as:
        J_ij = (f(x_i + h e_j) - f(x_i - h e_j)) / (2h)
        where e_j is the j-th standard basis vector and h is a small constant.

        Parameters:
        -----------
        x : Tensor (B,d)
            Batch of points where to compute the jacobian

        Returns:
        --------
        jacobian : Tensor (B,d_out,d)
            Batch of jacobians
        """
        B, d = x.shape
        flatten_diffeo = lambda x: self.diffeo(x).flatten(start_dim=1)
        y0 = flatten_diffeo(x)  # (B,d_out)
        d_out = y0.shape[1]
        J = torch.zeros(B, d_out, d, device=x.device, dtype=x.dtype)
        eye = torch.eye(d, device=x.device, dtype=x.dtype)

        if self.chunk_size is None or self.chunk_size < 1:
            chunk_size = 1
        else:
            chunk_size = self.chunk_size

        pbar = tqdm(
            range(0, d, chunk_size),
            desc="Computing pullback metric via finite differences",
            leave=False,
        )
        for start in pbar:
            end = min(start + chunk_size, d)
            eye_chunk = eye[start:end]  # (chunk_size, d)
            x_plus = x.unsqueeze(1) + self.eps * eye_chunk.unsqueeze(0)  # (B, chunk_size, d)
            x_minus = x.unsqueeze(1) - self.eps * eye_chunk.unsqueeze(0)  # (B, chunk_size, d)
            x_plus = rearrange(x_plus, "B C d -> (B C) d")  # (B * chunk_size, d)
            x_minus = rearrange(x_minus, "B C d -> (B C) d")  # (B * chunk_size, d)
            y_plus = flatten_diffeo(x_plus)  # (B * chunk_size, d_out)
            y_minus = flatten_diffeo(x_minus)  # (B * chunk_size, d_out)
            y_plus = rearrange(
                y_plus, "(B C) d_out -> B C d_out", B=B, C=end - start
            )  # (B, chunk_size, d_out)
            y_minus = rearrange(
                y_minus, "(B C) d_out -> B C d_out", B=B, C=end - start
            )  # (B, chunk_size, d_out)
            J[:, :, start:end] = (y_plus - y_minus).transpose(1, 2) / (
                2 * self.eps
            )  # (B, d_out, chunk_size)
        return J

    def jacobian_forward_mode(self, z: Tensor) -> Tensor:
        """
        Computes the jacobian of the diffeomorphism at the points z using forward mode autodiff.
        That is for each point z_i in the batch, and each dimension j,
        we compute the j-th column of the jacobian as:
        J_ij = d/dt f(z_i + t e_j) |_{t=0}
        where e_j is the j-th standard basis vector.
        """
        B, d = z.shape
        flatten_diffeo = lambda x: self.diffeo(x).flatten(start_dim=1)
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        cols = []
        for j in range(d):
            v = eye[j].unsqueeze(0).expand(B, -1)
            _, Jv = torch.func.jvp(flatten_diffeo, (z,), (v,))
            cols.append(Jv)
        return torch.stack(cols, dim=-1)  # (B, d_out, d)

    def metric_tensor(self, q: Tensor) -> Tensor:
        jacobian = self.jacobian(q)
        if not isinstance(self.base_cometric, IdentityCoMetric):
            g_base = self.base_cometric.metric_tensor(self.diffeo(q))
            g = jacobian.mT @ g_base @ jacobian
        else:
            g = jacobian.mT @ jacobian
        g = g + self.reg_coef * self.eye(q)
        return g

    def forward(self, q: Tensor) -> Tensor:
        g = self.metric_tensor(q)
        return torch.linalg.inv(g)

    # This version, albeit much faster and elegant still uses 
    # way too much memory for high-dimensional outputs.
    # So we resort to instantiating the metric tensor and computing
    # the regular dot product. This is not optimal but it works for now.
    # def dot(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
    #     flat_forward = lambda x: self.diffeo(x).flatten(start_dim=1)
    #     # If crash here because of forward AD : GLHF
    #     Jqu = torch.func.jvp(flat_forward, (q,), (u,))[1]
    #     Jqv = torch.func.jvp(flat_forward, (q,), (v,))[1]
    #     if not isinstance(self.base_cometric, IdentityCoMetric):
    #         g_base = self.base_cometric.metric_tensor(self.diffeo(q))
    #         return torch.einsum("bi,bij,bj->b", Jqu, g_base, Jqv)
    #     else:
    #         return torch.sum(Jqu * Jqv, dim=1)

    def extra_repr(self) -> str:
        return f"method={self.method}, reg_coef={self.reg_coef}"


class PBIG_Cometric_Gaussian(CoMetric):
    """Description of the Pullback information geometry metric.
    Here we only focus on the case where the decoder distributions are gaussians.

    Paper : Arvanitidis, Georgios, et al. "Pulling back information geometry." 25th International Conference on Artificial Intelligence and Statistics. 2022.


    Parameters:
    ----------
    decoder : torch.nn.Module
        The stochastic decoder of a VAE model. Its signature should be `decoder(z: torch.Tensor) -> (x_hat: torch.Tensor, logvar: torch.Tensor)`,
        where `x_hat` is the mean of the decoder distribution and `logvar` is the log-variance ; both of shape `(batch_size, ...)`,
        where `...` is the shape of the data space, typically `(batch_size, C, H, W)` for images.
    epsilon: float
        The small constant added to each vector basis in the latent space.
    rho : float
        The small constant added to the diagonal of the covariance matrix of the decoder distribution.
    """

    def __init__(self, decoder: torch.nn.Module, epsilon: float = 1e-4, rho: float = 1e-4):
        super().__init__()
        self.decoder = decoder
        self.epsilon = epsilon
        self.rho = rho

    def kl(
        self, normal_0: torch.distributions.Normal, normal_1: torch.distributions.Normal
    ) -> torch.Tensor:
        """Compute the KL divergence between two multivariate normal distributions.

        Parameters:
        ----------
        normal_0 : torch.distributions.Normal
            The first normal distribution.
        normal_1 : torch.distributions.Normal
            The second normal distribution.

        Returns:
        -------
        kl_div : torch.Tensor
            The KL divergence between the two distributions. Shape: (batch_size,)
        """
        kl = torch.distributions.kl_divergence(normal_0, normal_1)
        dims = normal_0.mean.shape[1:]
        kl = kl.sum(dim=tuple(range(1, len(dims) + 1)))  # Sum over all dimensions except batch
        return kl

    def dot(self, z: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes u^T G(q) v for a batch of points q at tangent vectors u and v.
        Here we use the polarization identity to avoid to compute the metric tensor explicitly.
        Ie :
        u^T G(q) v = 1/4 * ( (u+v)^T G(q) (u+v) - (u-v)^T G(q) (u-v) )

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
        uv_plus = u + v
        uv_minus = u - v

        fst = self.energy(z, uv_plus)
        snd = self.energy(z, uv_minus)
        dot_ = 0.25 * (fst - snd)
        return dot_

    def energy(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Computes p^TG(q)p for a batch of tangent vectors p at points q.
        Here the energy is easily given by the KL divergence between the decoder distributions at z and z+p.

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
        x_hat_base, logvar_base = self.decoder(z)
        normal_base = torch.distributions.Normal(x_hat_base, torch.exp(0.5 * logvar_base))

        z_plus = z + p
        x_hat_plus, logvar_plus = self.decoder(z_plus)
        normal_plus = torch.distributions.Normal(x_hat_plus, torch.exp(0.5 * logvar_plus))

        return self.kl(normal_base, normal_plus)

    def metric(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Computes the norm sqrt(p^TG(q)p) for a batch of tangent vectors p at points q.
        Here the norm is easily given by the KL divergence between the decoder distributions at z and z+p.

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        p : Tensor (b, d)
            Batch of tangent vectors

        Returns:
        -------
        res : Tensor (b,) sqrt(p^TG(q)p)
        """
        return self.energy(z, p).sqrt()

    def metric_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the metric tensor at a given point in the latent space.

        Parameters:
        ----------
        z : torch.Tensor
            The point in the latent space where the metric tensor is computed. Shape: (batch_size, latent_dim)

        Returns:
        -------
        g : torch.Tensor
            The metric tensor at point z. Shape: (batch_size, latent_dim, latent_dim)
        """
        batch_size, latent_dim = z.shape
        g = torch.zeros(batch_size, latent_dim, latent_dim, device=z.device)

        x_hat_base, logvar_base = self.decoder(z)
        normal_base = torch.distributions.Normal(x_hat_base, torch.exp(0.5 * logvar_base))
        # Fill the diagonal
        for i in range(latent_dim):
            z_plus = z.clone()
            z_plus[:, i] += self.epsilon
            x_hat_plus, logvar_plus = self.decoder(z_plus)
            normal_plus = torch.distributions.Normal(x_hat_plus, torch.exp(0.5 * logvar_plus))

            g[:, i, i] = 2 * self.kl(normal_base, normal_plus) / (self.epsilon**2)

        # Fill the off-diagonal
        for i in range(latent_dim):
            for j in range(i + 1, latent_dim):
                z_plus_i = z.clone()
                z_plus_i[:, i] += self.epsilon
                x_hat_plus_i, logvar_plus_i = self.decoder(z_plus_i)
                normal_plus_i = torch.distributions.Normal(
                    x_hat_plus_i, torch.exp(0.5 * logvar_plus_i)
                )

                z_plus_j = z.clone()
                z_plus_j[:, j] += self.epsilon
                x_hat_plus_j, logvar_plus_j = self.decoder(z_plus_j)
                normal_plus_j = torch.distributions.Normal(
                    x_hat_plus_j, torch.exp(0.5 * logvar_plus_j)
                )

                z_plus_ij = z.clone()
                z_plus_ij[:, i] += self.epsilon
                z_plus_ij[:, j] += self.epsilon
                x_hat_plus_ij, logvar_plus_ij = self.decoder(z_plus_ij)
                normal_plus_ij = torch.distributions.Normal(
                    x_hat_plus_ij, torch.exp(0.5 * logvar_plus_ij)
                )

                g[:, i, j] = (
                    self.kl(normal_base, normal_plus_ij) - self.kl(normal_base, normal_plus_i) - self.kl(normal_base, normal_plus_j)
                ) / (self.epsilon**2)

                g[:, j, i] = g[:, i, j]

        # Add rho to the diagonal for numerical stability
        g = g + self.rho * self.eye(z)
        return g

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the cometric tensor at a given point in the latent space.

        Parameters:
        ----------
        z : torch.Tensor
            The point in the latent space where the cometric tensor is computed. Shape: (batch_size, latent_dim)

        Returns:
        -------
        g : torch.Tensor
            The cometric tensor at point z. Shape: (batch_size, latent_dim, latent_dim)
        """
        return self.metric_tensor(z).inverse()


class PBIG_Cometric(CoMetric):
    """Description of the Pullback information geometry metric.
    Paper : Arvanitidis, Georgios, et al. "Pulling back information geometry." 25th International Conference on Artificial Intelligence and Statistics. 2022.


    Parameters:
    ----------
    decoder : torch.nn.Module
        The stochastic decoder of a VAE model. Its signature should be `decoder(z: torch.Tensor) -> torch.distributions.Distribution`.
    epsilon: float
        The small constant added to each vector basis in the latent space.
    rho : float
        The small constant added to the diagonal of the covariance matrix of the decoder distribution.
    """

    def __init__(self, decoder: torch.nn.Module, epsilon: float = 1e-4, rho: float = 1e-4):
        super().__init__()
        self.decoder = decoder
        self.epsilon = epsilon
        self.rho = rho

    def energy(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Computes p^TG(q)p for a batch of tangent vectors p at points q.
        Here the energy is easily given by the KL divergence between the decoder distributions at z and z+p.

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        p : Tensor (b, d)
            Batch of tangent vectors

        Returns:
        -------
        res : Tensor (b,) p^TG(q)p
        """
        base_distrib = self.decoder(z)
        plus_distrib = self.decoder(z + p)
        return torch.distributions.kl_divergence(base_distrib, plus_distrib)

    def dot(self, z: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes u^T G(q) v for a batch of points q at tangent vectors u and v.
        Here we use the polarization identity to avoid to compute the metric tensor explicitly.
        Ie :
        u^T G(q) v = 1/4 * ( (u+v)^T G(q) (u+v) - (u-v)^T G(q) (u-v) )

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
        uv_plus = u + v
        uv_minus = u - v

        fst = self.energy(z, uv_plus)
        snd = self.energy(z, uv_minus)
        dot_ = 0.25 * (fst - snd)
        return dot_

    def metric(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Computes the norm sqrt(p^TG(q)p) for a batch of tangent vectors p at points q.
        Here the norm is easily given by the KL divergence between the decoder distributions at z and z+p.

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        p : Tensor (b, d)
            Batch of tangent vectors

        Returns:
        -------
        res : Tensor (b,) sqrt(p^TG(q)p)
        """
        return self.energy(z, p).sqrt()

    def metric_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the metric tensor at a given point in the latent space.

        Parameters:
        ----------
        z : torch.Tensor
            The point in the latent space where the metric tensor is computed. Shape: (batch_size, latent_dim)

        Returns:
        -------
        g : torch.Tensor
            The metric tensor at point z. Shape: (batch_size, latent_dim, latent_dim)
        """
        batch_size, latent_dim = z.shape
        g = torch.zeros(batch_size, latent_dim, latent_dim, device=z.device)

        base_distrib = self.decoder(z)
        # Fill the diagonal
        for i in range(latent_dim):
            z_plus = z.clone()
            z_plus[:, i] += self.epsilon
            plus_distrib = self.decoder(z_plus)

            g[:, i, i] = (
                2
                * torch.distributions.kl_divergence(base_distrib, plus_distrib)
                / (self.epsilon**2)
            )

        # Fill the off-diagonal
        for i in range(latent_dim):
            for j in range(i + 1, latent_dim):
                z_plus_i = z.clone()
                z_plus_i[:, i] += self.epsilon
                plus_distrib_i = self.decoder(z_plus_i)

                z_plus_j = z.clone()
                z_plus_j[:, j] += self.epsilon
                plus_distrib_j = self.decoder(z_plus_j)

                z_plus_ij = z.clone()
                z_plus_ij[:, i] += self.epsilon
                z_plus_ij[:, j] += self.epsilon
                plus_distrib_ij = self.decoder(z_plus_ij)

                g[:, i, j] = (
                    torch.distributions.kl_divergence(base_distrib, plus_distrib_ij)
                    - torch.distributions.kl_divergence(base_distrib, plus_distrib_i)
                    - torch.distributions.kl_divergence(base_distrib, plus_distrib_j)
                ) / (self.epsilon**2)

                g[:, j, i] = g[:, i, j]

        # Add rho to the diagonal for numerical stability
        g = g + self.rho * self.eye(z)
        return g

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the cometric tensor at a given point in the latent space.

        Parameters:
        ----------
        z : torch.Tensor
            The point in the latent space where the cometric tensor is computed. Shape: (batch_size, latent_dim)

        Returns:
        -------
        g : torch.Tensor
            The cometric tensor at point z. Shape: (batch_size, latent_dim, latent_dim)
        """
        return self.metric_tensor(z).inverse()


class LiftedCometric(CoMetric):
    """
    Assume an original manifold of metric g.
    Let h be a function (eg.  1/classifier) that diverges on some regions of the manifold.
    This cometric implements a new metric that penalizes movement in the direction of the gradient of h.
    This will encourage geodesics to stay on the level sets of h. The metric is given by:
    g'(x) = g(x) + beta * grad(h(x)) @ grad(h(x))^T

    Parameters:
    -----------
    base_cometric: CoMetric
        The original metric tensor
    h: torch.nn.Module
        The function to condition the metric. It should have a signature (Batch, Dim) -> (Batch,1)
    beta: float
        The scaling factor for the conditioning
    """

    def __init__(self, base_cometric: CoMetric, h: torch.nn.Module, beta: float = 1):
        super().__init__()
        self.base_cometric = base_cometric
        self.h = h
        self.beta = beta

        self.diffeo = PullBackCometric(
            diffeo=self.h,
            reg_coef=0,
        )

    def metric_tensor(self, q: Tensor) -> Tensor:
        g_base = self.base_cometric.metric_tensor(q)
        if self.base_cometric.is_diag:
            g_base = torch.diag_embed(g_base)
        g_h = self.diffeo.metric_tensor(q)
        g = g_base + self.beta * g_h
        return g

    def forward(self, q: Tensor) -> Tensor:
        g = self.metric_tensor(q)
        return torch.linalg.inv(g)

    def extra_repr(self) -> str:
        return f"beta={self.beta}"


class FisherRaoCometric(CoMetric):
    """
    Cometric based on the Fisher-Rao metric, ie the hessian of the log-likelihood function.
    The metric is given by:
    g(x) = SoftAbs(-H_f(x)) + reg_coef * Id
    where H_f is the hessian of the log-likelihood function at x.

    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function of signature (X,theta)-> log_prob(X|theta)
        Where X is of shape (B,d)
    reg_coef : float
        Regularization coefficient for the metric
    softabs_alpha : float
        Regularization parameter for the softabs function. If None, no regularization is applied.
    data_sampler : callable
        Function to sample data from p(X|theta).
        It should have signature (N_pts:int,theta) -> Tensor (N_pts,d)
        Where N_pts is the number of points to sample, and d is the dimension of the data.
        If None, the sampling is done using a N(0,1) distribution.
    N_pts : int
        Number of points to sample for the empirical fisher information matrix.
    """

    def __init__(
        self,
        log_likelihood: callable,
        reg_coef: float = 1e-3,
        softabs_alpha=None,
        data_sampler=None,
        N_pts: int = 1000,
    ):
        super().__init__()
        self.N_pts = N_pts
        self.log_likelihood = log_likelihood
        self.reg_coef = reg_coef
        self.softabs_alpha = softabs_alpha
        if data_sampler is not None:
            self.data_sampler = data_sampler
        else:
            self.data_sampler = self.normal_sampling

    def log_no_batch(self, x, theta):
        """
        Log-likelihood function without batch dimension.

        Parameters
        ----------
        x : Tensor (d,)
            Data point
        theta : Tensor (p,)
            Parameter of the distribution
        """
        return self.log_likelihood(x.unsqueeze(0), theta).squeeze(0)

    def hessian_no_batch_all(self, x: Tensor, theta: Tensor):
        """
        Computes the hessian of the log-likelihood function at a single data point x.

        Parameters
        ----------
        x : Tensor (d,)
            Data point
        theta : Tensor (p,)
            Parameter of the distribution

        Returns
        -------
        hess : Tensor (p,p)
            Hessian of the log-likelihood function at x
        """
        hess = torch.func.hessian(self.log_no_batch, argnums=1)(x, theta)
        return hess

    def hessian_no_batch_param(self, x: Tensor, theta):
        """
        Computes the hessian of the log-likelihood function at a batch of data points x.

        Parameters
        ----------
        x : Tensor (B,d)
            Batch of data points
        theta : Tensor (p,)
            Parameter of the distribution

        Returns
        -------
        hess : Tensor (B,p,p)
            Batch of Hessians of the log-likelihood function at x
        """
        B, d = x.shape
        hess = []
        for i in range(B):
            hess_i = self.hessian_no_batch_all(x[i], theta)
            hess.append(hess_i)
        hess = torch.stack(hess, dim=0)
        return hess

    def normal_sampling(self, N_pts: int, theta: Tensor):
        d = theta.shape[1]
        return torch.randn(N_pts, d, device=theta.device, dtype=theta.dtype)

    def inf_matrix(self, theta):
        """
        Computes the empirical fisher information matrix at theta.
        Uses a Monte Carlo estimate with N_pts samples.

        inf_mat = -E_x [ H_f(x,theta) ]

        Parameters
        ----------
        theta : Tensor (B,p)
            Batch of parameters of the distribution

        Returns
        -------
        fim : Tensor (B,p,p)
            Batch of empirical fisher information matrices at theta
        """
        x = self.data_sampler(self.N_pts, theta)
        B, p = theta.shape
        hess = []
        for i in range(B):
            hess_i = self.hessian_no_batch_param(x, theta[i])
            hess.append(hess_i)
        hess = torch.stack(hess, dim=0)  # (B,N_pts,p,p)
        fim = -hess.mean(dim=1)  # (B,p,p)
        return fim

    def metric_tensor(self, theta: Tensor):
        g = self.inf_matrix(theta)
        if self.softabs_alpha is not None:
            g = SoftAbs(g, alpha=self.softabs_alpha)
        g += self.reg_coef * self.eye(theta)
        return g

    def forward(self, q: Tensor):
        g = self.metric_tensor(q)
        return torch.linalg.inv(g)


################################################################
# Interpolation cometrics
################################################################


# First some utils functions
def compute_kmedoids(data: Tensor, K: int) -> np.ndarray:
    """
    Compute the K-Medoids clustering of the data and return the indices of the medoids.

    Parameters:
    -----------
    data : Tensor (N,d)
        The data to cluster
    K : int
        The number of clusters

    Returns:
    -----------
    medoid_indices : np.ndarray (K,)
        The indices of the medoids in the original data
    """
    dst_mat = torch.cdist(data, data, p=2).sqrt().cpu().detach().numpy()
    k_medoids_model = kmedoids.KMedoids(n_clusters=K, metric="precomputed", random_state=1312)
    k_medoids_model.fit(dst_mat)
    return k_medoids_model.medoid_indices_


def compute_global_temperature_square(centroids: Tensor, neighbor_k: int = 1) -> float:
    """
    Compute the global temperature for the gaussian interpolation of the cometric at the centroids.
    The temperature is set to the maximum of the n-th smallest distance between centroids.

    Parameters:
    -----------
    centroids : Tensor (K,d)
        The centroids of the clusters
    neighbor_k : int
        The index of the neighbor to consider for the distance computation. Default is 1 (second nearest neighbor).

    Returns:
    -----------
    temperature_square : float
        The global temperature for the gaussian interpolation of the cometric at the centroids.
    """
    dst_mat = torch.cdist(centroids, centroids, p=2)
    dst_mat[dst_mat == 0] = float("inf")  # Avoid zero self distances
    sorted_distances, _ = torch.sort(dst_mat, dim=1)
    second_min_distances = sorted_distances[
        :, neighbor_k
    ]  # Get the n-th smallest distance for each centroid
    return second_min_distances.max() ** 2


def compute_centroid_scales(
    centroids: Tensor,
    data: Tensor,
    kappa: Tensor,
    min_cluster_size: int = 3,
    neighbor_rank: int = 5,
    min_scale_quantile: float = 0.25,
    max_scale_quantile: float = 0.95,
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute the (squared) bandwidths of the RBF kernels as:

        tau_k^2 = kappa * scale_k^2

    where scale_k^2 is an adaptive local-scale estimate for centroid c_k,
    blended between two sources:

    1. Local in-cluster scale (reliable when the cluster has enough points):
        local_scale_k^2 = (1 / Card(C_k)) * sum_{x_i in C_k} ||c_k - x_i||^2
    where C_k is the set of data points assigned to centroid c_k (i.e. c_k
    is their nearest centroid).

    2. Neighbor-based robust scale (used as a fallback when Card(C_k) is
    small and the local estimate would be unreliable/pathological):
        nn_scale_k^2 = ||c_k - c_(neighbor_rank)||^2
    i.e. the squared distance from c_k to its `neighbor_rank`-th nearest
    neighboring centroid (not the 1st, to avoid overly peaked kernels).
    This is clamped between global quantiles of nn_scale^2 across all
    centroids (min_scale_quantile, max_scale_quantile) so that outlier
    centroids do not produce pathologically small or large bandwidths.

    The two estimates are blended per-centroid via a reliability weight
    r_k in [0, 1], based on the cluster cardinality relative to
    min_cluster_size:

        r_k     = clamp((Card(C_k) - 1) / (min_cluster_size - 1), 0, 1)
        scale_k^2 = r_k * local_scale_k^2 + (1 - r_k) * nn_scale_k^2

    The blended scale is then re-clamped to the same global [min, max]
    bounds, and the final bandwidth is:

        tau_k^2 = kappa * scale_k^2

    Parameters:
    -----------
    centroids : Tensor (K, d)
        The centroids of the clusters, used to compute the neighbor-based scale.
    data : Tensor (N, d)
        The original data points, used to compute the clusters of centroids.
    kappa : Tensor
        The scaling factor applied to the blended scale to obtain tau^2.
    min_cluster_size : int
        Cluster cardinality at or above which the local in-cluster scale is
        trusted fully (reliability = 1).
    neighbor_rank : int
        Which nearest-neighbor centroid (1-indexed, excluding self) to use
        for the robust fallback scale.
    min_scale_quantile, max_scale_quantile : float
        Quantiles (over all centroids) used to clamp the neighbor-based
        scale, preventing outlier centroids from dominating.
    eps : float
        Small constant to avoid division by zero / degenerate scales.

    Returns:
    --------
    tau_squared : Tensor (K,)
        The computed squared bandwidths for each centroid.
    """
    K = centroids.shape[0]

    # Assign each sample to the closest centroid.
    dst_data = torch.cdist(centroids, data, p=2)  # (K,N)
    closest_centroid = dst_data.argmin(dim=0)  # (N,)

    # Robust fallback scale from centroid geometry.
    # Use a higher-order neighbor to avoid over-peaked kernels when K is large.
    dst_centroids = torch.cdist(centroids, centroids, p=2)  # (K,K)
    dst_centroids.fill_diagonal_(float("inf"))
    sorted_dst, _ = torch.sort(dst_centroids, dim=1)
    rank = min(max(neighbor_rank - 1, 0), max(K - 2, 0))  # (K,)
    nn_dist = sorted_dst[:, rank]  # (K,)
    nn_scale2 = nn_dist.pow(2).clamp_min(eps)

    # Global robust floor/ceiling so outlier clusters do not dominate smoothness.
    min_scale2 = torch.quantile(nn_scale2, min_scale_quantile).clamp_min(eps)
    max_scale2 = torch.quantile(nn_scale2, max_scale_quantile).clamp_min(eps)

    # Local in-cluster scale. May be unreliable when cluster cardinality is very small.
    local_scale2 = torch.zeros(K, device=centroids.device, dtype=centroids.dtype)
    counts = torch.bincount(closest_centroid, minlength=K).to(centroids.dtype)
    for k in range(K):
        cluster_points = data[closest_centroid == k]  # (Card(C_k),d)
        if cluster_points.shape[0] > 0:
            c_k = centroids[k : k + 1]
            dist2 = torch.cdist(c_k, cluster_points, p=2).pow(2)
            local_scale2[k] = dist2.mean().clamp_min(eps)

    # Blend local scale with centroid-neighborhood scale.
    # For small clusters (K ~ N), this prevents pathological very narrow kernels.
    reliability = ((counts - 1) / max(min_cluster_size - 1, 1)).clamp(0.0, 1.0)
    scale2 = reliability * local_scale2 + (1.0 - reliability) * nn_scale2
    scale2 = scale2.clamp(min=min_scale2, max=max_scale2)

    tau_squared = kappa * scale2

    # Fix any potential numerical issues with the bandwidths
    if not torch.isfinite(tau_squared).all():
        finite_mask = torch.isfinite(tau_squared)
        if finite_mask.any():
            fill_value = tau_squared[finite_mask].median()
        else:
            fill_value = torch.tensor(1.0, device=tau_squared.device, dtype=tau_squared.dtype)
        tau_squared = torch.where(finite_mask, tau_squared, fill_value)
    return tau_squared


class CentroidsCometric(CoMetric):
    """Cometric based on the cometric computed on centroids.
    New cometric is computed as a gaussian interpolation of the cometric at the centroids.

    G^{-1}(z) = sum_{k=1}^K w_k(z) G^{-1}(c_k)
    Where w_k(z) = exp(-||z-c_k||^2 / (2*tau_k^2))         if not metric_weight
                 = exp(-||z-c_k||^2_G(c_k) / (2*tau_k^2))  else

    Parameters:
    -----------
    centroids : Tensor (K,d)
        The centroids of the clusters
    cometric_centroids: Tensor (K,d,d)
        The cometric tensor at the centroids
    reg_coef : float
        Regularization coefficient for the cometric
    K: int, Default None
        If None, uses all the centroids
        If not None, the number of centroids to use, computed by KMedoids clustering.
    metric_weight: bool
        If True, the interpolation weights is given by N(c_k,Sigma_k) else it is N(c_k,Id).
    kappa: float
        The scaling factor for the bandwidths of the RBF kernels.
    use_global_temperature: bool
        If True, the temperature is the same for all centroids.
    """

    def __init__(
        self,
        centroids: Tensor,
        cometric_centroids: Tensor,
        reg_coef: float = 1e-3,
        K: int = None,
        metric_weight: bool = False,
        kappa: float = 1.0,
        use_global_temperature: bool = False,
    ):
        super().__init__()

        self.register_buffer("centroids", centroids)
        # if cometric_centroids is not None:
        #     self.register_buffer("cometric_centroids", cometric_centroids)
        if cometric_centroids.ndim == 2:
            self.is_diag = True
        else:
            self.is_diag = False
        cometric_centroids = self.assess_cometric_tensor_symmetry(cometric_centroids)
        self.register_buffer("cometric_centroids", cometric_centroids)
        self.register_buffer("reg_coef", Tensor([reg_coef]))

        if K is not None:
            if K > self.centroids.shape[0]:
                print(
                    f"Warning: K={K} is greater than the number of centroids {self.centroids.shape[0]}. Using all centroids."
                )
                K = self.centroids.shape[0]
            centroids_idx = compute_kmedoids(self.centroids, K)
            self.centroids = self.centroids[centroids_idx]
            self.cometric_centroids = self.cometric_centroids[centroids_idx]
            self.K = K

        if use_global_temperature:
            tau_squared = compute_global_temperature_square(self.centroids)
            tau_squared = (
                kappa
                * tau_squared
                * torch.ones(self.centroids.shape[0], device=self.centroids.device)
            )
        else:
            tau_squared = compute_centroid_scales(self.centroids, self.centroids, kappa)
        self.register_buffer("tau_squared", tau_squared)

        self.metric_weight = metric_weight
        # if K is not None and centroids is not None:
        #     self.process_centroids(K)
        # elif K is None and centroids is not None:
        #     self.K = self.centroids.size(0)
        # else:
        #     self.K = K

        # if cometric_centroids is not None:
        #     self.cometric_centroids: Tensor = self.assess_cometric_tensor_symmetry(
        #         self.cometric_centroids
        #     )
        # self.metric_weight = metric_weight

    def assess_cometric_tensor_symmetry(self, cometric_centroids: Tensor) -> Tensor:
        """
        Check if the cometric tensor is symmetric positive semi-definite.

        Parameters:
        -----------
        cometric_centroids : Tensor (K,d,d) or (K,d)
            The cometric tensor at the centroids

        Returns:
        -----------
        Tensor (K,d,d) or (K,d)
            The (possibly symmetrized) cometric tensor at the centroids
        """
        assert cometric_centroids.ndim in [
            2,
            3,
        ], f"Cometric centroids should be of shape (K,d) or (K,d,d), got {cometric_centroids.shape}"
        assert (
            cometric_centroids.shape[1] == self.centroids.shape[1]
        ), f"Cometric centroids should have the same shape as centroids ({self.centroids.shape}), got {cometric_centroids.shape}"

        # When diagonal cometric is used, cometric_centroids can be 2D
        if cometric_centroids.ndim == 2:
            self.is_diag = True
            return cometric_centroids
        else:
            assert (
                cometric_centroids.shape[1] == cometric_centroids.shape[2]
            ), f"Cometric centroids should be square matrices, got {cometric_centroids.shape}"

        if not torch.allclose(cometric_centroids, cometric_centroids.mT):
            # Make it symmetric
            print(
                "Warning: Cometric centroids are not symmetric. Making them symmetric by using (A+A^T)/2."
            )
            cometric_centroids = (cometric_centroids + cometric_centroids.mT) / 2
        return cometric_centroids

    def forward(self, z: Tensor) -> Tensor:
        # Expand the computation to save memory when latentdim >> 1
        if self.metric_weight:
            if self.is_diag:
                z_term = torch.einsum("bd,kd,bd->bk", z, self.cometric_centroids, z)  # (b,k)
                cross_term = torch.einsum(
                    "bd,kd->bk", z, self.cometric_centroids * self.centroids
                )  # (b,k)
                c_term = torch.einsum(
                    "kd,kd,kd->k", self.centroids, self.cometric_centroids, self.centroids
                ).unsqueeze(
                    0
                )  # (1,k)
            else:
                z_term = torch.einsum("bj,kij,bi->bk", z, self.cometric_centroids, z)
                cross_term = torch.einsum(
                    "bj,kij,ki->bk", z, self.cometric_centroids, self.centroids
                )
                c_term = torch.einsum(
                    "kj,kij,ki->k", self.centroids, self.cometric_centroids, self.centroids
                ).unsqueeze(0)
        else:
            z_term = (torch.linalg.vector_norm(z, dim=-1) ** 2).unsqueeze(-1)  # (b,1)
            c_term = (torch.linalg.vector_norm(self.centroids, dim=-1) ** 2).unsqueeze(
                0
            )  # (1,k)
            cross_term = torch.einsum("bd,kd->bk", z, self.centroids)  # (b,k)

        dz = z_term + c_term - 2 * cross_term
        weights = torch.exp(-(dz**2) / (2 * self.tau_squared))  # (b,K)
        G_inv = self.cometric_centroids  # (k,d,d) | (k,d)
        if not self.is_diag:
            G_inv = torch.einsum("bk,kij->bij", weights, G_inv)
        else:
            G_inv = torch.einsum("bk,kd->bd", weights, G_inv)

        G_inv = G_inv + self.reg_coef * self.eye(z)  # (b,d,d) | (b,d)
        return G_inv

    def extra_repr(self) -> str:
        return f"K={self.K}, reg_coef={self.reg_coef.item():.3f}, metric_weight={self.metric_weight}, is_diag={self.is_diag}, tau_squared={self.tau_squared}"


class LANDCometric(CoMetric):
    """
    Cometric based on the LAND metric.
    The cometric is given by:
    G_inv(x) = diag(h(x)) + reg_coef * Id
    where h(x) = sum_k w_k (x_k^alpha - x^alpha)^2
    Where w_k(x) = exp(-||x_k - x||^2 / (2 * sigma^2_k))        if not metric_weight
                 = exp(-||x_k - x||^2_G(x_k) / (2 * sigma^2_k)) else
    and where x_k are the centroids, and alpha is a parameter that controls the shape of the metric.

    Parameters:
    -----------
    centroids : Tensor (K,d)
        The centroids of the clusters
    alpha : int
        The alpha parameter of the LAND metric. It controls the shape of the metric. Default to 1.
    kappa : float
        The kappa parameter of the LAND metric. It controls the width of the Gaussian kernel.
        Default to 1.0.
    reg_coef : float
        The regularization coefficient. Default to 1e-5.
    K: int, Default None
        If not None, the number of centroids to use, computed by KMedoids clustering.
        If None, uses all centroids.
    use_global_temp: bool
        If True, the sigma parameter is the same for all centroids.
        Otherwise, the sigma parameter is not the same for all centroids.
    """

    def __init__(
        self,
        centroids: Tensor,
        alpha: int = 1,
        kappa: float = 1.0,
        reg_coef: float = 1e-3,
        K: int = None,
        use_global_temp: bool = False,
    ):
        super().__init__(is_diag=True)

        assert (
            centroids.ndim == 2
        ), f"Centroids should be of shape (K,d), got {centroids.shape}"
        assert alpha > 0 and isinstance(
            alpha, int
        ), f"Alpha should be a positive integer, got {alpha}"
        assert reg_coef >= 0, f"Reg_coef should be a non-negative float, got {reg_coef}"

        self.register_buffer("centroids", centroids)
        self.register_buffer("alpha", Tensor([alpha]))
        self.register_buffer("reg_coef", Tensor([reg_coef]))

        if K is not None:
            if K > self.centroids.shape[0]:
                print(
                    f"Warning: K={K} is greater than the number of centroids {self.centroids.shape[0]}. Using all centroids."
                )
                K = self.centroids.shape[0]
            self.centroids = self.centroids[compute_kmedoids(self.centroids, K)]

        self.K = self.centroids.shape[0]
        self.d = self.centroids.shape[1]
        if use_global_temp:
            tau_squared = compute_global_temperature_square(self.centroids)  # float
            tau_squared = (
                kappa
                * tau_squared
                * torch.ones(self.K, device=self.centroids.device, dtype=self.centroids.dtype)
            )
        else:
            tau_squared = compute_centroid_scales(
                self.centroids, self.centroids, kappa=kappa
            )  # (K,)
        self.register_buffer("tau_squared", tau_squared)

    def h(self, x: Tensor) -> Tensor:
        """
        Computes the h(x) function of the LAND metric.

        Parameters:
        x : Tensor (B,d)
            The input points

        Returns:
        Tensor (B,)
            The computed h(x) values
        """
        x_alpha = x**self.alpha  # (B,d)
        centroids_alpha = self.centroids**self.alpha  # (K,d)
        diff = x_alpha[:, None, :] - centroids_alpha[None, :, :]  # (B,K,d)
        dst = torch.cdist(x, self.centroids, p=2)  # (B,K)
        weights = torch.exp(-(dst**2) / (2 * self.tau_squared[None, :]))  # (B,K)
        h_x = weights[:, :, None] * (diff**2)  # (B,K,d)
        h_x = h_x.sum(dim=1)  # (B,d)
        return h_x

    def forward(self, x: Tensor) -> Tensor:
        h_x = self.h(x)
        G_inv = h_x + self.reg_coef * self.eye(x)
        return G_inv

    def extra_repr(self) -> str:
        return f"K={self.K}, alpha={self.alpha.item()}, reg_coef={self.reg_coef.item():.3f}, tau_squared={self.tau_squared}"


class LANDRBFCometric(CoMetric):
    """
    Generalisation of the LAND cometric using RBF kernels instead of a single Gaussian kernel.
    The cometric is given by:
    G_inv(x) = diag(h(x)) + reg_coef * Id
    where h(x) = sum_k w_k exp(- ||x - c_k||^2 / (2 * tau_k^2))
    where c_k are the centroids, and tau_k are the bandwidths of the RBF kernels.
    The weights w_k can be learned or fixed to 1/K.

    Parameters:
    -----------
    centroids : Tensor (N,d)
        The centroids of the clusters
    K: int, Default None
        If not None, the number of centroids to use, computed by KMedoids clustering.
        If None, uses all centroids.
    kappa : float. Default to 1.0.
        The scaling factor for the bandwidths of the RBF kernels.
    reg_coef : float. Default to 1e-5.
        The regularization coefficient.
    learn_weights : bool. Default to False.
        Whether to learn the weights w_k of the RBF kernels. If False, they are fixed to 1/K.
    """

    def __init__(
        self,
        data: Tensor,
        K: int,
        kappa: float = 1.0,
        reg_coef: float = 1e-3,
        learn_weights: bool = False,
    ):
        super().__init__(is_diag=True)

        assert data.ndim == 2, f"data should be of shape (N,d), got {data.shape}"
        assert reg_coef >= 0, f"Reg_coef should be a non-negative float, got {reg_coef}"
        assert kappa > 0, f"kappa should be a positive float, got {kappa}"

        self.register_buffer("reg_coef", Tensor([reg_coef]))
        self.register_buffer("kappa", Tensor([kappa]))

        self.register_buffer("centroids", data)

        if K is not None:
            if K > data.shape[0]:
                print(
                    f"Warning: K={K} is greater than the number of data points {data.shape[0]}. Using all data points."
                )
                K = data.shape[0]
            self.centroids = self.centroids[compute_kmedoids(self.centroids, K)]

        self.K = self.centroids.shape[0]

        tau_squared = compute_centroid_scales(self.centroids, data, kappa)
        self.register_buffer("tau_squared", tau_squared)

        # We do it this way so that w is then fixed.
        if learn_weights:
            w_ = self.learn_w(data)
        else:
            w_ = torch.ones(self.K, device=data.device, dtype=data.dtype) / self.K
        self.register_buffer("w_", w_)

    @property
    def w(self) -> Tensor:
        """
        Get the weights w_k of the RBF kernels.

        Returns:
        Tensor (K,)
            The weights of the RBF kernels
        """
        return torch.nn.functional.softplus(self.w_)

    def learn_w(self, data: Tensor, n_iters: int = 400) -> None:
        """
        Learn the weights w_k of the RBF kernels by minimizing the mean squared error between the cometric at the centroids and the cometric given by the RBF interpolation at the centroids.
        """
        w_ = nn.Parameter(torch.ones(self.K, device=data.device, dtype=data.dtype) / self.K)
        optimizer = torch.optim.Adam([w_], lr=1e-2)
        loss_list = []
        pbar = tqdm(range(n_iters), desc="Learning RBF weights", leave=False)
        for _ in pbar:
            optimizer.zero_grad()
            h_x = self.h(data, torch.nn.functional.softplus(w_))
            loss = (1 - h_x).pow(2).mean()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        self.loss_list = torch.tensor(loss_list)
        return w_.detach().data

    def h(self, x: Tensor, w: Tensor) -> Tensor:
        """
        Computes the h(x) function of the RBF cometric.

        Parameters:
        x : Tensor (B,d)
            The input points
        w : Tensor (K,)
            The positive weights of the RBF kernels

        Returns:
        Tensor (B,)
            The computed h(x) values
        """
        dst = torch.cdist(x, self.centroids, p=2)  # (B,K)
        rbf = dst**2 / (2 * self.tau_squared[None, :] + 1e-12)  # (B,K)
        rbf = torch.exp(-rbf)
        h_x = torch.einsum("k,bk->b", w, rbf)  # (B,)
        return h_x

    def cometric_tensor(self, x: Tensor) -> Tensor:
        h_x = self.h(x, self.w)[:, None]  # (B,1)
        G_inv = h_x.expand(-1, x.shape[1]) + self.reg_coef * self.eye(x)  # (B,d)
        return G_inv


#################################################################
# Parametric cometrics
#################################################################
class DiagonalCometricModel(CoMetric):
    """
    Parametric diagonal cometric model. All diagonal values can either be different or the same depending on
    the value of latent_dim. If latent_dim=1, all diagonal values are the same, the tensor is a scaled identity matrix.
    Otherwise, the diagonal values are different.

    Parameters:
    -----------
    in_dim : int
        Dimension of the input features
    hidden_dim : int
        Dimension of the hidden layer
    latent_dim : int
        Dimension of the latent space
    lbd : float
        Regularization parameter
    """

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, lbd: float = 1):
        super().__init__(is_diag=True)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lbd = lbd

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the model to output the euclidean distance"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        diag_val = self.layers(x)
        diag_val = torch.exp(diag_val)
        G_inv = (diag_val + self.lbd) * self.eye(diag_val)
        return G_inv

    def metric_tensor(self, q: Tensor) -> Tensor:
        diag_val = self.layers(q)
        diag_val = torch.exp(diag_val)
        return (1 / diag_val + 1 / self.lbd) * self.eye(diag_val)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, latent_dim={self.latent_dim}, lbd={self.lbd}"


class CometricModel(CoMetric):
    """
    General parametric cometric model. The cometric tensor is a symmetric positive definite matrix.
    The parametrization here uses the Cholesky decomposition of the cometric tensor.

    Parameters:
    -----------
    input_dim : int
        Dimension of the input features
    hidden_dim : int
        Dimension of the hidden layer
    latent_dim : int
        Dimension of the latent space
    lbd : float
        Regularization parameter
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, lbd: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lbd = lbd

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.diag = nn.Linear(hidden_dim, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(hidden_dim, k)

        self.indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the model to output the euclidean distance"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.diag.weight)
        nn.init.zeros_(self.diag.bias)
        nn.init.zeros_(self.lower.weight)
        nn.init.zeros_(self.lower.bias)

    def forward(self, features: Tensor) -> Tensor:
        x = self.layers(features)
        log_diag = self.diag(x)
        lower = self.lower(x)

        L = torch.zeros(
            x.size(0), self.latent_dim, self.latent_dim, device=x.device, dtype=x.dtype
        )
        L[:, self.indices[0], self.indices[1]] = lower
        L += torch.diag_embed(log_diag.exp())

        G_inv = torch.bmm(L, L.transpose(1, 2))

        id = self.eye(G_inv[:, :, 0])

        return G_inv + self.lbd * id

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, latent_dim={self.latent_dim}, lbd={self.lbd}"


class SmallConvCometricModel(CoMetric):
    """
    Simple convolutional metric backbone
    It expects to receive square image of shape (B, C, W, W) where

    Parameters:
    -----------
    latent_dim : int
        Dimension of the latent space
    n_channels : int
        Number of channels of the image (BW or RBG)
    width : int
        Width of the input image (assumed to be square)
    lbd : float
        Regularization parameter to avoid singularities in the metric tensor

    Returns:
    --------
    G_inv : Tensor (B, latent_dim, latent_dim)
        The inverse of the metric tensor for the input images
    """

    def __init__(
        self, latent_dim: int, n_channels: int = 1, width: int = 64, lbd: float = 1e-10
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.width = width
        self.lbd = lbd

        self.l1 = nn.Sequential(
            nn.Conv2d(
                self.n_channels, 128, kernel_size=(4, 4), stride=2, padding=1
            ),  # (B, 128, W/2, W/2)
            nn.InstanceNorm2d(num_features=128),
            nn.Softplus(),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1),  # (B, 256, W/4, W/4)
            nn.InstanceNorm2d(num_features=256),
            nn.Softplus(),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1),  # (B, 512, W/8, W/8)
            nn.InstanceNorm2d(num_features=512),
            nn.Softplus(),
            nn.Flatten(1),  # Flatten to (B, 512 * W/8 * W/8)
        )

        w3 = self.get_dim_out_()
        last_dim = 512 * w3 * w3  # Output dimension after the conv layers

        k = int(self.latent_dim * (self.latent_dim - 1) / 2)

        self.diag = nn.Linear(last_dim, self.latent_dim)
        self.lower = nn.Linear(last_dim, k)

        self.indices = torch.tril_indices(self.latent_dim, self.latent_dim, offset=-1)

        self.layers = nn.Sequential(
            self.l1,
            self.l2,
            self.l3,
        )

    def get_out_conv_dim_(self, W_in: int, pad: int, ker_size: int, stride: int) -> int:
        """
        Returns the output dimension of the conv layers

        Parameters:
        -----------
        W_in : int
            Input width
        pad : int
            Padding
        ker_size : int
            Kernel size
        stride : int
            Stride
        """
        W_out = (W_in + 2 * pad - ker_size) / stride + 1
        return torch.floor(Tensor([W_out])).int()

    def get_dim_out_(self) -> int:
        """
        Returns the output dimension of the conv layers
        """
        W1 = self.get_out_conv_dim_(self.width, 1, 4, 2)
        W2 = self.get_out_conv_dim_(W1, 1, 4, 2)
        W3 = self.get_out_conv_dim_(W2, 1, 4, 2)
        return int(W3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)  # (B, 512 * W4 * W4)
        log_diag = self.diag(x)
        lower = self.lower(x)

        L = torch.zeros(
            x.size(0), self.latent_dim, self.latent_dim, device=x.device, dtype=x.dtype
        )
        L[:, self.indices[0], self.indices[1]] = lower
        L += torch.diag_embed(log_diag.exp())

        G_inv = torch.bmm(L, L.transpose(1, 2))

        id = self.lbd * self.eye(G_inv[:, :, 0])

        return G_inv + self.lbd * id


class Cometric_MLP(CoMetric):
    """
    A cometric model based on a simple MLP architecture.
    The cometric tensor is parametrized via its Cholesky decomposition.

    Parameters:
    -----------
    input_dim : int or tuple[int, ...]
        Dimension of the input features. If tuple, it is assumed to be the shape of an image.
    latent_dim : int
        Dimension of the latent space
    lbd : float
        Regularization parameter to avoid singularities in the metric tensor
    """

    def __init__(self, input_dim: int | tuple[int, ...], latent_dim: int, lbd: float = 0.01):
        super().__init__()

        self.input_dim = np.prod(input_dim) if isinstance(input_dim, tuple) else input_dim
        self.latent_dim = latent_dim
        self.lbd = lbd

        self.layers = nn.Sequential(nn.Linear(self.input_dim, 400), nn.ReLU())
        self.diag = nn.Linear(400, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(400, k)

    def forward(self, x: Tensor) -> Tensor:

        h1 = self.layers(x.reshape(-1, self.input_dim))
        h21, h22 = self.diag(h1), self.lower(h1)

        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(x.device)
        indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)

        # get non-diagonal coefficients
        L[:, indices[0], indices[1]] = h22

        # add diagonal coefficients
        L = L + torch.diag_embed(h21.exp())

        M = L @ torch.transpose(L, 1, 2)  # LL^T

        M = M + torch.eye(self.latent_dim).to(x.device) * self.lbd  # add regularization
        return M


#################################################################
# Randers metrics
#################################################################
class FinslerMetric(nn.Module):
    """
    Finsler metric base class
    """

    def __init__(self):
        super(FinslerMetric, self).__init__()

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
        super(SlopeMetrics, self).__init__()
        self.f = f
        self.f_no_batch = lambda x: self.f(x.unsqueeze(0)).squeeze(0)
        self.df_ = torch.vmap(torch.func.jacrev(self.f_no_batch))

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


class RandersMetrics(FinslerMetric):
    """
    Compute F(x,v) = |v|_{G} + beta *  omega(x) . v
    Randers metrics with a fixed base metric and a variable 1-form.

    The 1-form must verify the condition that the resulting Randers metric is positive.
    It is up to the user to ensure this condition is satisfied.

    Parameters
    ----------
    base_cometric : CoMetric
        Base cometric to use for the Randers metric.
    omega : nn.Module
        1-form to use for the Randers metric. It should be a function that takes
        in points on the manifold and outputs a vector of the same size as the points.
    beta : float
        Scaling factor for the 1-form. Default is 1.0. Must be within the range [0,1].
        When beta=0, the Randers metric reduces to the base cometric.
    """

    def __init__(
        self,
        base_cometric: CoMetric,
        omega: nn.Module,
        beta: float = 1.0,
    ):
        super(RandersMetrics, self).__init__()
        self.base_cometric = base_cometric
        self.omega = omega
        assert 0 <= beta <= 1, "Beta must be in the range [0, 1]"
        self.beta = beta

    def beta_form(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Computes the beta form of the Randers metric.

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        beta : Tensor (b,)
            Beta form of the Randers metric at x in the direction of v
        """
        omega_x = self.omega(x)
        beta = torch.einsum("bi,bi->b", omega_x, v)
        return self.beta * beta

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        alpha = self.base_cometric.metric(x, v)
        beta = self.beta_form(x, v)
        F = alpha + beta
        return F

    def fund_tensor_analytic_(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Computes the fundamental tensor of the Randers metric using the analytic formula.
        See Lemma 11.1.4 from 'An Introduction to Riemann-Finsler Geometry' by Bao, Chern, Shen.

        Parameters:
        ----------
        z : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at z

        Returns:
        -------
        g : Tensor (b,d,d)
            Fundamental tensor of the Randers metric at z in the direction of v
        """
        F_z_v = self.forward(z, v)
        v_norm = self.base_cometric.metric(z, v)
        b = self.beta * self.omega(z)
        a = self.base_cometric.metric_tensor(z)
        if self.base_cometric.is_diag:
            l_tilde = (a * v) / v_norm[:, None]
        else:
            l_tilde = torch.einsum("bij,bj->bi", a, v) / v_norm[:, None]

        l = l_tilde + b
        ll_tilde = torch.einsum("bi,bj->bij", l_tilde, l_tilde)
        ll = torch.einsum("bi,bj->bij", l, l)

        if self.base_cometric.is_diag:
            delta_term = -ll_tilde
            diag_idx = torch.arange(0, a.shape[-1])
            delta_term[:, diag_idx, diag_idx] += a
        else:
            delta_term = a - ll_tilde

        c = (F_z_v / v_norm)[:, None, None]
        g = c * delta_term + ll

        return g

    def inv_fund_tensor_analytic_(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Computes the inverse of the fundamental tensor of the Randers metric using the analytic formula.
        See Lemma 11.2.1 from 'An Introduction to Riemann-Finsler Geometry' by Bao, Chern, Shen.

        Parameters:
        ----------
        q : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at q

        Returns:
        -------
        g_inv : Tensor (b,d,d)
            Inverse of the fundamental tensor of the Randers metric at q in the direction of v
        """
        F = self.forward(q, v)
        alpha = self.base_cometric.metric(q, v)

        a = self.base_cometric.metric_tensor(q)
        if self.base_cometric.is_diag:
            a_inv = torch.diag_embed(1.0 / a, dim1=-2, dim2=-1)
        else:
            a_inv = a.inverse()

        fst_term = (alpha / F)[:, None, None] * a_inv

        b = self.omega(q)
        beta = self.beta * torch.einsum("bi,bi->b", b, v)
        b_tilde_top = torch.einsum("bij,bj->bi", a_inv, b)
        b_tilde_norm = torch.einsum("bi,bi->b", b_tilde_top, b)
        l_tilde = v / alpha[:, None]

        ll = torch.einsum("bi,bj->bij", l_tilde, l_tilde)
        snd_term = (
            (alpha**2 / F**3)[:, None, None]
            * (beta + alpha * b_tilde_norm)[:, None, None]
            * ll
        )

        li_bj = torch.einsum("bi,bj->bij", l_tilde, b_tilde_top)
        lj_bi = torch.einsum("bj,bi->bij", l_tilde, b_tilde_top)

        trd_term = (alpha**2 / F**2)[:, None, None] * (li_bj + lj_bi)

        g_inv = fst_term + snd_term - trd_term
        return g_inv

    def fundamental_tensor(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Computes the fundamental tensor of the Randers metric
        at the point x in the direction v.
        g_ij(x,y) =1/2 d^2F^2(x,y)/(dy_i*dy_j)

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        g : Tensor (b,d,d)
            Fundamental tensor of the Randers metric at x in the direction of v
        """
        return self.fund_tensor_analytic_(x, v)

    def inverse_fundamental_tensor(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Computes the inverse of the fundamental tensor of the Randers metric
        at the point x in the direction v.
        g^ij(x,y) = (g_ij(x,y))^-1

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        g_inv : Tensor (b,d,d)
            Inverse of the fundamental tensor of the Randers metric at x in the direction of v
        """
        return self.inv_fund_tensor_analytic_(x, v)


class _DualOmegaWrapper(nn.Module):
    """Wrapper module that computes dual 1-form on-the-fly when called by parent class methods."""

    def __init__(self, dual_randers_instance):
        super().__init__()
        # Keep a non-Module reference to avoid creating a recursive module tree.
        self._dual_randers = weakref.proxy(dual_randers_instance)

    def forward(self, x: Tensor) -> Tensor:
        return self._dual_randers.omega_star(x)


class _DualCometricWrapper(CoMetric):
    """Wrapper cometric that computes dual metric tensor on-the-fly when called by parent class methods."""

    def __init__(self, dual_randers_instance):
        super().__init__()
        # Keep a non-Module reference to avoid creating a recursive module tree.
        self._dual_randers = weakref.proxy(dual_randers_instance)
        self.is_diag = False

    def forward(self, x: Tensor) -> Tensor:
        """Returns the inverse of the dual metric tensor (the dual cometric)."""
        G_star = self._dual_randers.G_star(x)
        return torch.linalg.inv(G_star)

    def metric_tensor(self, x: Tensor) -> Tensor:
        """Returns the dual metric tensor."""
        return self._dual_randers.G_star(x)


class DualRandersMetrics(RandersMetrics):
    """
    Dual Randers metric class. The dual Randers metric is defined as F*(x,p) = sup_{v} (p.v - F(x,v))
    where F is the primal Randers metric.

    Tips : Use the parameter beta of the Randers metric to be absolutely sure that
    the primal Randers metric is positive.

    Parameters:
    -----------
    randers_metric : RandersMetrics
        The primal Randers metric to dualize.
    epsilon : float
        Small regularization parameter to allow for better differentiability.
        Hence we have F_star_eps(x,v) = sqrt(F_star(x,v)^2 + epsilon^2)
    """

    def __init__(self, randers_metric: RandersMetrics, epsilon: float = 1e-8):
        super(DualRandersMetrics, self).__init__(
            base_cometric=randers_metric.base_cometric,
            omega=randers_metric.omega,
            beta=1.0,
        )
        self.primal_randers = randers_metric
        self.omega = _DualOmegaWrapper(self)
        self.base_cometric = _DualCometricWrapper(self)
        self.beta = 1.0
        self.epsilon = epsilon

    def omega_star(self, x: Tensor) -> Tensor:
        """
        Compute the dual 1-form omega* at point x.

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold

        Returns:
        -------
        omega_star : Tensor (b,d)
            Dual 1-form at point x
        """
        omega = self.primal_randers.beta * self.primal_randers.omega(x)  # (b,d)
        G_inv = self.primal_randers.base_cometric.cometric_tensor(x)  # (b,d,d) | (b,d)

        if self.primal_randers.base_cometric.is_diag:
            G_inv_w = G_inv * omega
        else:
            G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)  # (b,d)

        alpha = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)  # (b,)

        omega_star = -1 / alpha[:, None] * G_inv_w  # (b,d)
        return omega_star

    def G_star(self, x: Tensor) -> Tensor:
        """
        Compute the dual metric tensor G* at point x.

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold

        Returns:
        G_star : Tensor (b,d,d)
            Dual metric tensor at point x
        """
        omega = self.primal_randers.beta * self.primal_randers.omega(x)  # (b,d)
        G_inv = self.primal_randers.base_cometric.cometric_tensor(x)  # (b,d,d) | (b,d)

        if self.primal_randers.base_cometric.is_diag:
            G_inv_w = G_inv * omega  # (b,d)
        else:
            G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)  # (b,d)

        alpha = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)  # (b,)

        G_star = torch.einsum("bi,bj->bij", G_inv_w, G_inv_w)  # (b,d,d)
        if self.primal_randers.base_cometric.is_diag:
            alpha_G_inv = alpha[:, None] * G_inv  # (b,d)
            G_star = (G_star + torch.diag_embed(alpha_G_inv)) / alpha[
                :, None, None
            ] ** 2  # (b,d,d)
        else:
            alpha_G_inv = alpha[:, None, None] * G_inv  # (b,d,d)
            G_star = (G_star + alpha_G_inv) / alpha[:, None, None] ** 2  # (b,d,d)
        return G_star

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Compute the dual Randers metric F*(x,v) = sup_{b} <b,v> with F(x,b) <= 1.

        The expression is extracted from http://arxiv.org/abs/2404.03999.

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        F_star : Tensor (b,)
            Dual Randers metric values at (x,v)
        """
        omega = self.primal_randers.beta * self.primal_randers.omega(x)  # (b,d)
        G_inv = self.primal_randers.base_cometric.cometric_tensor(x)  # (b,d,d) | (b,d)

        if self.primal_randers.base_cometric.is_diag:
            G_inv_w = G_inv * omega  # (b,d)
        else:
            G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)  # (b,d)

        alpha = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)  # (b,)

        omega_star = -1 / alpha[:, None] * G_inv_w  # (b,d)

        G_star = torch.einsum("bi,bj->bij", G_inv_w, G_inv_w)  # (b,d,d)
        if self.primal_randers.base_cometric.is_diag:
            alpha_G_inv = alpha[:, None] * G_inv  # (b,d)
            G_star = (G_star + torch.diag_embed(alpha_G_inv)) / alpha[
                :, None, None
            ] ** 2  # (b,d,d)
        else:
            alpha_G_inv = alpha[:, None, None] * G_inv  # (b,d,d)
            G_star = (G_star + alpha_G_inv) / alpha[:, None, None] ** 2  # (b,d,d)

        v_norm = torch.einsum("bi,bij,bj->b", v, G_star, v).sqrt()  # (b,)
        omega_star_v = torch.einsum("bi,bi->b", omega_star, v)  # (b,)
        F_star = v_norm + omega_star_v  # (b,)
        reg_F_star = torch.sqrt(F_star**2 + self.epsilon**2)  # (b,)
        return reg_F_star
