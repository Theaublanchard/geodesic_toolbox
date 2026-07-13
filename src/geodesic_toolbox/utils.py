import torch
from torch import Tensor
from .cometric import CoMetric, RandersMetrics
from einops import rearrange
from math import ceil, exp, cos
from tqdm import tqdm
import numpy as np


def sample_hypersphere(batch_size: int, dim: int, radius: float = 0.95) -> torch.Tensor:
    """
    Sample points from a D-dimensional hypersphere.

    Parameters:
    ----------
    batch_size : int
        Number of points to sample.
    dim : int
        Dimension of the hypersphere.
    radius : float
        Radius of the hypersphere. Default is 1.0.

    Returns:
    -------
    torch.Tensor (batch_size, dim)
        Tensor containing the sampled points.
    """
    # Sample from standard normal distribution
    normal_samples = torch.randn(batch_size, dim)

    # Normalize to the unit sphere
    unit_sphere_samples = normal_samples / torch.norm(normal_samples, dim=1, keepdim=True)

    # Scale by radius and a random factor to fill the sphere
    random_radii = torch.rand(batch_size, 1) ** (1 / dim)
    sphere_samples = unit_sphere_samples * random_radii * radius

    return sphere_samples


def delta(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    num = 2 * torch.linalg.vector_norm(u - v, dim=-1) ** 2
    denum = (1 - torch.linalg.vector_norm(u, dim=-1) ** 2) * (
        1 - torch.linalg.vector_norm(v, dim=-1) ** 2
    )
    return num / denum


def pointcarre_dst(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the distance between two points on the Poincaré ball.

    Parameters:
    ----------
    u : torch.Tensor (batch_size, dim)
        First point.
    v : torch.Tensor (batch_size, dim)
        Second point.

    Returns:
    -------
    torch.Tensor (batch_size,)
        Tensor containing the distances.
    """
    return torch.arccosh(1 + delta(u, v))


def scaled_euclidean_dst(
    u: torch.Tensor, v: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Computes the distance between two points on a manifold where the metric is given by scale * I.

    Parameters:
    ----------
    u : torch.Tensor (batch_size, dim)
        First point.
    v : torch.Tensor (batch_size, dim)
        Second point.
    scale : torch.Tensor (batch_size,)
        Scaling factor.

    Returns:
    -------
    torch.Tensor (batch_size,)
        Tensor containing the distances.
    """
    return scale.sqrt() * torch.linalg.vector_norm(u - v, dim=-1)


def get_bounds(embeddings: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Compute the bounds of the embeddings.

    Parameters:
    ----------
    embeddings : torch.Tensor (n_points, 2)
        The embeddings of the points.
    margin : float
        Margin scaling factor to add to the bounds. (default is 0)
        This is useful to avoid points being too close to the edges of the plot.

    Returns:
    -------
    torch.Tensor (4,)
        [min_x, max_x, min_y, max_y], the bounds of the embeddings.
    """
    min_x, max_x = embeddings[:, 0].min(), embeddings[:, 0].max()
    min_y, max_y = embeddings[:, 1].min(), embeddings[:, 1].max()
    # Add margin to the bounds
    min_x -= margin * (max_x - min_x)
    max_x += margin * (max_x - min_x)
    min_y -= margin * (max_y - min_y)
    max_y += margin * (max_y - min_y)
    # Ensure the bounds are in the correct order
    min_x, max_x = min(min_x, max_x), max(min_x, max_x)
    min_y, max_y = min(min_y, max_y), max(min_y, max_y)
    # Create a tensor with the bounds
    bounds = [min_x, max_x, min_y, max_y]
    bounds = torch.tensor(bounds)
    return bounds


def magnification_factor(g_inv: CoMetric, z: torch.Tensor) -> torch.Tensor:
    """
    Return the magnification factor as sqrt det G(z).
    This is always well defined because G(z) is positive definite.
    This implementation is based on the fact that det G(z) = 1 / det G_inv(z).

    Parameters:
    ----------
    g_inv : CoMetric
        Function that outputs the inverse metric tensor as a (b,d,d) matrix
    z : torch.Tensor (b,d)
        Point at which to compute the magnification factor

    Returns:
    -------
    mf : torch.Tensor (b,)
        Magnification factor
    """
    G_inv = g_inv.cometric_tensor(z)
    if not g_inv.is_diag:
        return torch.det(G_inv).abs().pow(-0.5)
    else:
        return torch.prod(G_inv, dim=1).abs().pow(-0.5)


def magnification_factor_metric(g_inv: CoMetric, z: torch.Tensor) -> torch.Tensor:
    """
    Return the magnification factor as sqrt det G(z).
    This is always well defined because G(z) is positive definite.
    This implementation uses the metric tensor instead of the inverse metric tensor.

    Parameters:
    ----------
    g_inv : CoMetric
        Function that outputs the inverse metric tensor as a (b,d,d) matrix
    z : torch.Tensor (b,d)
        Point at which to compute the magnification factor

    Returns:
    -------
    mf : torch.Tensor (b,)
        Magnification factor
    """
    G = g_inv.metric_tensor(z)
    if not g_inv.is_diag:
        return torch.det(G).abs().pow(0.5)
    else:
        return torch.prod(G, dim=1).abs().pow(0.5)


@torch.jit.script
def hamiltonian(G_inv: Tensor, p: Tensor) -> Tensor:
    """
    Computes the hamiltonian at point q yielding cometric G_inv(q) for the momentum p.
    It is defined as H(p,q) = p^T G_inv(q) p.

    Parameters:
    ----------
    G_inv : Tensor (b,d,d)
        Cometric tensor.
    p : Tensor (b,d)
        Momentum.

    Returns:
    -------
    res : Tensor (b,)
        Hamiltonian.
    """
    res = torch.einsum("...i,...ij,...j->...", p, G_inv, p)
    return res


def scale_lr_magnification(mf: float, base_lr: float) -> float:
    """
    Scales the learning rate according to the magnification factor.
    This is to avoid shooting being stuck in high curvature region.

    Parameters:
    ----------
    mf : float
        Magnification factor.
    base_lr : float
        Base learning rate.

    Returns:
    -------
    new_lr : float
        Scaled learning rate.
    """
    max_scale = 50
    min_scale = 0.1
    coef = 0.5
    new_lr = (1 - exp(-coef * mf)) * max_scale * base_lr + min_scale * base_lr
    return new_lr


def constant_time_scaling_schedule(x: float) -> float:
    """
    Constant time scaling schedule.
    Results in ceil(1/dt)*(n_step + 1) integration steps.

    Parameters:
    ----------
    x : float
        Current iteration progress (0 to 1)

    Returns:
    -------
    scaling : float
        Scaling factor
    """
    return 1


def linear_time_scaling_schedule(x: float, max_scaling: float = 5) -> float:
    """Linear ramp from max_scaling to 1.
    Results in ()*(n_step + 1) integration steps.

    Parameters:
    ----------
    x : float
        Current iteration progress (0 to 1)
    max_scaling : float
        Maximum scaling factor

    Returns:
    -------
    scaling : float
        Scaling factor
    """
    return 1 + (max_scaling - 1) * (1 - x)


def cosine_time_scaling_schedule(x: float, max_scaling: float = 5) -> float:
    """Cosine decay from max_scaling to 1

    Parameters:
    ----------
    x : float
        Current iteration progress (0 to 1)
    max_scaling : float
        Maximum scaling factor

    Returns:
    -------
    scaling : float
        Scaling factor
    """
    return 1 + (max_scaling - 1) * (1 + cos(x * 3.1415)) / 2


def get_mf_image(
    cometric: CoMetric,
    embeddings: torch.Tensor = None,
    bounds: list = None,
    resolution: int = 200,
    use_mf_metric: bool = False,
    max_b_size: int = 512,
    verbose=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor:
    """
    Compute the magnification factor on the latent space so as to visualize the distortion of the space.

    Parameters:
    ----------
    cometric : CoMetric
        The CoMetric object.
    embeddings : torch.Tensor (n_points, 2)
        The embeddings of the points.
    bounds : list
        [min_x, max_x, min_y, max_y], the bounds of the embeddings.
    resolution : int
        The resolution of the grid.
    use_mf_metric : bool
        Whether to use the metric tensor to compute the magnification factor instead of the cometric tensor.
    max_b_size : int
        Maximum batch size for the computation.
    verbose : bool
        Whether to print the progress.

    Returns:
    -------
    mf_image : torch.Tensor (resolution, resolution)
        The magnification factor image.
    """
    if bounds is None and embeddings is None:
        raise ValueError("Either bounds or embeddings must be provided.")
    if bounds is None:
        bounds = get_bounds(embeddings)
    min_x, max_x, min_y, max_y = bounds

    if embeddings is not None:
        device = embeddings.device
        dtype = embeddings.dtype

    x_plot = torch.linspace(min_x, max_x, resolution, device=device, dtype=dtype)
    y_plot = torch.linspace(min_y, max_y, resolution, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x_plot, y_plot, indexing="ij")
    Q = torch.stack([xx, yy], dim=-1)
    W, H, _ = Q.shape
    Q = rearrange(Q, "w h c -> (w h) c")
    mf_image = torch.zeros(W * H, device=device)
    fn = magnification_factor_metric if use_mf_metric else magnification_factor
    if not verbose:
        pbar = range(0, W * H, max_b_size)
    else:
        pbar = tqdm(
            range(0, W * H, max_b_size),
            desc="Computing magnification factor",
            unit="batch",
            total=ceil(W * H / max_b_size),
        )
    with torch.no_grad():
        # batch computation to avoid memory issues
        for i in pbar:
            mf_image[i : i + max_b_size] = fn(cometric, Q[i : i + max_b_size].to(device))
    mf_image = rearrange(mf_image, "(w h) -> w h", w=W, h=H).T
    return mf_image.cpu()


def batched_kro(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the kronecker product between two batch of matrices p and q

    Parameters
    ----------
    a : Tensor
        (batch_size, m,n)
    b : Tensor
        (batch_size, p,q)

    Returns
    -------
    Tensor
        (batch_size, m*p, n*q)
    """
    kro_prod = torch.einsum("bmn,bpq->bmpnq", a, b)
    kro_prod = rearrange(kro_prod, "b m p n q -> b (m p) (n q)")
    return kro_prod


def vector_kro(p: Tensor, q: Tensor) -> Tensor:
    """Compute the kronecker product between two batches of vector p and q

    Parameters
    ----------
    p : Tensor
        (b,m)
    q : Tensor
        (b,n)

    Returns
    -------
    Tensor
        (b, m*n)
    """
    kro_prod = batched_kro(p.unsqueeze(2), q.unsqueeze(2))
    kro_prod = kro_prod.squeeze(2)
    return kro_prod


def vec(M: Tensor) -> Tensor:
    """Stack the columns of M

    Parameters
    ----------
    M : Tensor
        (b,m, n)

    Returns
    -------
    Tensor
        (b, m*n)
    """
    return rearrange(M, "b m n -> b (m n)")


def sample_cometric_normal(cometric: CoMetric, q0: torch.Tensor) -> torch.Tensor:
    """
    Sample tangent vectors from the normal distribution N(0, G_inv(q0))

    Parameters
    ----------
    cometric : CoMetric
    q0 : torch.Tensor (N_pts, d)
        Point at which to sample

    Returns
    -------
    v : torch.Tensor (N_pts, d)
        Sampled points
    """
    # Sample according to N(0, G_inv)
    z = torch.randn_like(q0)
    G_inv = cometric.cometric_tensor(q0)
    if cometric.is_diag:
        L = G_inv.sqrt()
        v = z * L
    else:
        L = torch.linalg.cholesky(G_inv)  # (b,d,d)
        v = torch.einsum("bij,bj->bi", L, z)
    return v


def sample_cone(
    x_0: Tensor, randers: RandersMetrics, N_pts=1, theta: float = torch.pi / 4, max_iter=1000
) -> Tensor:
    """
    Sample a random vector in the cone defined by the angle theta around the direction w.
    Uses rejection sampling from the hypersphere.

    Parameters
    ----------
    x_0 : torch.Tensor (d,)
        The point at which to sample the vector.
    randers : RandersMetrics
        The Randers metric defining the cone.
    N_pts : int
        The number of points to sample.
    theta : float
        The angle defining the cone. Must be in (0, pi).

    Returns
    -------
    v : torch.Tensor (N_pts, d)
        A random vector in the cone.
    """
    cos_theta = torch.cos(torch.tensor(theta, device=x_0.device))
    d = x_0.shape[-1]
    omega_x0 = randers.omega(x_0.unsqueeze(0)).squeeze(0)
    G_inv = randers.base_cometric.cometric_tensor(x_0.unsqueeze(0)).squeeze(0)
    if randers.base_cometric.is_diag:
        L = G_inv.sqrt()
    else:
        L = torch.linalg.cholesky(G_inv)

    v = torch.zeros((N_pts, d), device=x_0.device, dtype=x_0.dtype)
    mask = torch.zeros(
        N_pts, dtype=torch.bool, device=x_0.device
    )  # 0 if not found, 1 if found
    n_iter = 0
    while mask.sum() < N_pts and n_iter < max_iter:
        n_iter += 1
        v_tmp = torch.randn((N_pts, d), device=x_0.device, dtype=x_0.dtype)
        if randers.base_cometric.is_diag:
            v_tmp = L * v_tmp
        else:
            v_tmp = torch.einsum("ij,bj->bi", L, v_tmp)
        cosim = torch.nn.functional.cosine_similarity(v_tmp, -omega_x0, dim=-1)
        mask_tmp = cosim >= cos_theta  # New points that are ok
        v[mask_tmp & ~mask] = v_tmp[mask_tmp & ~mask]
        mask = mask | mask_tmp
    if n_iter == max_iter:
        print(
            f"Warning: max_iter reached when sampling cone with theta={theta}. Only {mask.sum().item()} points were sampled."
        )
    return v


def sample_cone_VMF(
    x_0: Tensor, randers: RandersMetrics, kappa: float, alpha: float = 1.0
) -> Tensor:
    """
    Samples point from a VMF distribution centered around - omega(x_0) with concentration kappa.
    Then flips the sample so that it is in the cone defined by -omega(x_0).
    Finally samples are scaled to have norm alpha.

    Parameters
    ----------
    x_0 : torch.Tensor (b,d)
        The point at which to sample the vector.
    randers : RandersMetrics
        The Randers metric defining the cone.
    kappa : float
        The concentration parameter for the VMF distribution.
    alpha : float
        The norm of the sampled vector.

    Returns
    -------
    v : torch.Tensor (b,d)
        A random vector in the cone.
    """
    omega_x0 = randers.omega(x_0)
    v = random_VMF(-omega_x0, kappa)
    dot_prod = torch.einsum("bi,bi->b", v, -omega_x0)
    mask = dot_prod < 0
    v[mask] = -v[mask]
    v = alpha * v / torch.linalg.norm(v, dim=-1, keepdim=True)
    return v


def random_VMF(mu, kappa, size=None):
    """
    Von Mises - Fisher distribution sampler with
    mean direction mu and concentration kappa.
    Source : https://hal.science/hal-040004568

    Parameters
    ----------
    mu : torch.Tensor (b,d)
        Mean direction.
    kappa : float
        Concentration parameter.
    size : tuple, optional
        Output shape. If None, returns a single sample per batch element.

    Returns
    -------
    torch.Tensor (b, *size, d)
        Samples from the Von Mises-Fisher distribution.
    """
    if mu.dim() == 1:
        return _random_VMF(mu, kappa, size)
    elif mu.dim() == 2:
        out = []
        for i in range(mu.shape[0]):
            out.append(_random_VMF(mu[i], kappa, size))
        return torch.stack(out, dim=0)
    else:
        raise ValueError("mu must be a 1D or 2D tensor.")


def _random_VMF(mu, kappa, size=None):
    """
    Von Mises - Fisher distribution sampler with
    mean direction mu and concentration kappa.
    Source : https://hal.science/hal-04004568/

    Parameters
    ----------
    mu : torch.Tensor (d,)
        Mean direction.
    kappa : float
        Concentration parameter.
    size : tuple, optional
        Output shape. If None, returns a single sample.

    Returns
    -------
    torch.Tensor
        Samples from the Von Mises-Fisher distribution.
    """
    # parse input parameters
    if size is None:
        n = 1
        shape = ()
    else:
        size = torch.flatten(torch.tensor(size, dtype=torch.int64))
        n = int(torch.prod(size).item())
        shape = tuple(size.tolist())
    mu = mu / torch.linalg.norm(mu)
    d = mu.shape[0]
    # z component: radial samples perpendicular to mu
    z = torch.randn((n, d), device=mu.device, dtype=mu.dtype)
    z = z / torch.linalg.norm(z, dim=1, keepdim=True)
    z = z - (z @ mu[:, None]) * mu[None, :]
    z = z / torch.linalg.norm(z, dim=1, keepdim=True)
    # sample angles (in cos and sin form)
    cos = _random_VMF_cos(d, kappa, n).to(device=mu.device, dtype=mu.dtype)
    sin = torch.sqrt(1 - cos**2)
    # combine angles with the z component
    x = z * sin[:, None] + cos[:, None] * mu[None, :]
    return x.reshape((*shape, d))


def _random_VMF_cos(d: int, kappa: float, n: int):
    """
    Generate i.i.d. samples t with density function given by
    p(t)=someConstant*(1-t**2)**((d-2)/2)*exp(kappa*t)
    """
    # b=Eq.4 of https://doi.org/10.1080/03610919408813161
    b = (d - 1) / (2 * kappa + (4 * kappa**2 + (d - 1) ** 2) ** 0.5)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    found = 0
    out = []
    while found < n:
        m = min(n, int((n - found) * 1.5))
        z = torch.distributions.Beta((d - 1) / 2, (d - 1) / 2).sample((m,))
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        test = kappa * t + (d - 1) * np.log(1 - x0 * t) - c
        accept = test >= -torch.distributions.Exponential(1).sample((m,))
        out.append(t[accept])
        found += len(out[-1])
    return torch.cat(out)[:n]
