import torch
from torch import Tensor

from .base_class import CoMetric

# We need to do some tricks to allow for differentiable evaluation.


def safe_eigh(A: Tensor) -> tuple[Tensor, Tensor]:
    """
    Batched symmetric eigendecomposition returning NaN for non-finite inputs
    instead of raising. Branch-free, so it survives torch.vmap / torch.func.

    ``torch.linalg.eigh`` raises _LinAlgError as soon as ONE matrix in the batch
    holds a non-finite entry, and names no sample, so callers can only reject
    the whole batch. Here those matrices are swapped for the identity and their
    eigenpairs returned as NaN.

    In the sampling setting, ``proposal_rate`` turns into alpha = 0 for
    that sample alone.

    Parameters
    ----------
    A : Tensor (b, n, n)
        Symmetric matrices to decompose

    Returns
    -------
    eigenvalues : Tensor (b, n)
        The eigenvalues of A
    eigenvectors : Tensor (b, n, n)
        The eigenvectors of A
    """
    ok = torch.isfinite(A).all(dim=-1).all(dim=-1)
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    lam, Phi = torch.linalg.eigh(torch.where(ok.unsqueeze(-1).unsqueeze(-1), A, eye))
    nan = torch.full((), float("nan"), device=A.device, dtype=A.dtype)

    safe_eigenvalues = torch.where(ok.unsqueeze(-1), lam, nan)
    safe_eigenvectors = torch.where(ok.unsqueeze(-1).unsqueeze(-1), Phi, nan)
    return safe_eigenvalues, safe_eigenvectors


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
    L, Q = safe_eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def _softabs_g(lam: Tensor, alpha: float) -> Tensor:
    """
    Compute the SoftAbs regularization function for a
    batch of eigenvalues lam and a regularization parameter alpha.
    The eigendecomposition is expected to be of the METRIC tensor.
    So that this function output the regularized eigenvalues
    of the COMETRIC tensor as:
        reg_eigenvalue = 1/lam * tanh(alpha*lam)

    when lam is not too small, and a Taylor expansion around 0 otherwise.

    Parameters
    ----------
    lam : Tensor (b, n)
        Eigenvalues of the metric tensor
    alpha : float
        Regularization parameter for the SoftAbs

    Returns
    -------
    Tensor (b, n)
        Regularized eigenvalues of the cometric tensor
    """
    alpha = float(alpha)
    u = alpha * lam
    small = u.abs() < 1e-3
    u_s = torch.where(small, u, torch.zeros_like(u))
    series = alpha * (1 - u_s**2 / 3 + 2 * u_s**4 / 15)
    lam_d = torch.where(small, torch.ones_like(lam), lam)
    return torch.where(small, series, torch.tanh(u) / lam_d)


def _softabs_dg(lam: Tensor, alpha: float) -> Tensor:
    """
    Derivative of ``_softabs_g`` with respect to lam.
    It is defined as:
        g'(lam) = alpha*sech^2(alpha*lam)/lam - tanh(alpha*lam)/lam^2
    where sech^2 = 1 - tanh^2 (so it underflows to 0 rather than overflowing for
    large alpha*lam).
    Near lam = 0 the two terms are both ~alpha/lam and cancel
    catastrophically, so the Taylor branch -2*alpha^3*lam/3 is used there.

    Parameters
    ----------
    lam : Tensor (b, n)
        Eigenvalues of the cometric tensor
    alpha : float
        Regularization parameter for the SoftAbs

    Returns
    -------
    Tensor (b, n)
        Derivative of the SoftAbs regularization function with respect to lam
    """
    alpha = float(alpha)
    u = alpha * lam
    small = u.abs() < 1e-3
    # Expressed in u rather than lam: the equivalent form in lam needs alpha**5,
    # which overflows int64 when alpha is passed as a python int (e.g. 10**6).
    u_s = torch.where(small, u, torch.zeros_like(u))
    series = alpha**2 * (-2 * u_s / 3 + 8 * u_s**3 / 15)
    lam_d = torch.where(small, torch.ones_like(lam), lam)
    t = torch.tanh(u)
    direct = alpha * (1 - t**2) / lam_d - t / lam_d**2
    return torch.where(small, series, direct)


def _softabs_d2g(lam: Tensor, alpha: float) -> Tensor:
    """
    Second derivative g''(lam) of ``_softabs_g``, needed for the SECOND-order
    divided differences (see ``_softabs_gamma2``). With t = tanh(alpha*lam),

        g'' = -2 a^2 t (1-t^2)/lam - 2 a (1-t^2)/lam^2 + 2 t/lam^3,

    which is again a cancelling sum of ~alpha/lam terms near lam = 0, so a
    Taylor branch alpha^3 (-2/3 + 8 u^2/5), u = alpha*lam, is used there.

    Parameters
    ----------
    lam : Tensor (b, n)
        Eigenvalues of the cometric tensor
    alpha : float
        regularization parameter for the SoftAbs
    """
    alpha = float(alpha)
    u = alpha * lam
    small = u.abs() < 1e-3
    u_s = torch.where(small, u, torch.zeros_like(u))
    series = alpha**3 * (-2.0 / 3.0 + 8.0 * u_s**2 / 5.0)
    lam_d = torch.where(small, torch.ones_like(lam), lam)
    t = torch.tanh(u)
    sech2 = 1 - t**2
    direct = (
        -2 * alpha**2 * t * sech2 / lam_d - 2 * alpha * sech2 / lam_d**2 + 2 * t / lam_d**3
    )
    return torch.where(small, series, direct)


def _softabs_gamma(lam: Tensor, alpha: float) -> Tensor:
    """
    Loewner / Daleckii-Krein matrix, shape (..., n, n):

        Gamma_ij = (g(lam_i) - g(lam_j)) / (lam_i - lam_j),  i != j
        Gamma_ii = g'(lam_i)

    Coincident eigenvalues fall back to the limit g' at the midpoint. This is
    what makes the funnel usable: its theta block is d-fold degenerate, where
    forming 1/(lam_i - lam_j) separately (as eigh's backward does) loses all
    precision.

    Parameters
    ----------
    lam : Tensor (b, n)
        Eigenvalues of the metric tensor
    alpha : float
        Regularization parameter for the SoftAbs

    Returns
    -------
    Tensor (b, n, n)
        Loewner / Daleckii-Krein matrix
    """
    g = _softabs_g(lam, alpha)
    dg_num = g.unsqueeze(-1) - g.unsqueeze(-2)
    li, lj = lam.unsqueeze(-1), lam.unsqueeze(-2)
    dlam = li - lj
    scale = torch.maximum(li.abs(), lj.abs()).clamp_min(1.0)
    degenerate = dlam.abs() < 1e-7 * scale
    dlam_d = torch.where(degenerate, torch.ones_like(dlam), dlam)
    return torch.where(degenerate, _softabs_dg((li + lj) / 2, alpha), dg_num / dlam_d)


def _softabs_gamma2(lam: Tensor, alpha: float) -> Tensor:
    """
    SECOND divided differences g[lam_i, lam_k, lam_j], shape (..., n, n, n):

        g[x, y, z] = (g[y, z] - g[x, y]) / (z - x)

    with the coincidence limits analytic -- two coinciding:
    g[x,y,x] = (g'(x) - g[x,y])/(x - y); all three: g[x,x,x] = g''(x)/2.
    Makes the SoftAbs map twice differentiable without ever dividing by an
    eigenvalue gap, which FHMC needs (its field Jacobian is a second derivative
    of this map).
    """
    g1 = _softabs_gamma(lam, alpha)  # (..., n, n)
    dg = _softabs_dg(lam, alpha)  # (..., n)
    d2g = _softabs_d2g(lam, alpha)  # (..., n)

    li = lam.unsqueeze(-1).unsqueeze(-1)  # index i
    lk = lam.unsqueeze(-2).unsqueeze(-1)  # index k
    lj = lam.unsqueeze(-2).unsqueeze(-2)  # index j
    scale = lam.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    tol = 1e-7 * scale.unsqueeze(-1).unsqueeze(-1)

    g1_kj = g1.unsqueeze(-3)  # g[k, j]
    g1_ik = g1.unsqueeze(-1)  # g[i, k]
    d_ij = lj - li
    d_ik = li - lk

    # generic branch: (g[k,j] - g[i,k]) / (lam_j - lam_i)
    d_ij_safe = torch.where(d_ij.abs() < tol, torch.ones_like(d_ij), d_ij)
    generic = (g1_kj - g1_ik) / d_ij_safe

    # lam_i == lam_j, lam_k distinct: (g'(i) - g[i,k]) / (lam_i - lam_k)
    d_ik_safe = torch.where(d_ik.abs() < tol, torch.ones_like(d_ik), d_ik)
    dg_i = dg.unsqueeze(-1).unsqueeze(-1)
    two_equal = (dg_i - g1_ik) / d_ik_safe

    # all three coincide
    all_equal = (d2g / 2).unsqueeze(-1).unsqueeze(-1).expand_as(generic)

    out = torch.where(d_ik.abs() < tol, all_equal, two_equal)
    return torch.where(d_ij.abs() < tol, out, generic)


class _SoftAbsD1(torch.autograd.Function):
    """
    First differential of the SoftAbs map, as a Function so that it is itself
    DIFFERENTIABLE.

    ``_SoftAbsCoMetric.backward`` must delegate here rather than compute the
    expression inline: inline it is built from lam/Q out of ``saved_tensors``,
    which carry no graph back to H, so autograd sees a constant linear map and
    every SECOND derivative comes back as exactly 0 -- silently wrong for FHMC,
    whose field Jacobian is a second derivative of this map. Do not inline it.

    d/dM is the same self-adjoint map; d/dH comes from the second divided
    differences (``_softabs_gamma2``), so both orders are degeneracy-safe.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(M, H, lam, Q, alpha):
        # H is unused in the value; it is an input only to give autograd a slot
        # for the second-order gradient. lam/Q passed in to avoid a second eigh.
        return _SoftAbsCoMetric._apply_gamma(lam, Q, M, alpha)

    @staticmethod
    def setup_context(ctx, inputs, output):
        M, _H, lam, Q, alpha = inputs
        ctx.alpha = alpha
        ctx.save_for_backward(M, lam, Q)
        ctx.save_for_forward(M, lam, Q)

    @staticmethod
    def _d2(lam, Q, A, B, alpha):
        """
        Second differential, in the eigenbasis with A~ = Q^T A Q:
            (D^2 g[A,B])~_ij = sum_k g[l_i, l_k, l_j] (A~_ik B~_kj + B~_ik A~_kj)
        Symmetric in A and B. O(n^3) per sample.
        """
        g2 = _softabs_gamma2(lam, alpha)  # (..., n, n, n)
        At, Bt = Q.mT @ A @ Q, Q.mT @ B @ Q
        inner = torch.einsum("...ikj,...ik,...kj->...ij", g2, At, Bt) + torch.einsum(
            "...ikj,...ik,...kj->...ij", g2, Bt, At
        )
        out = Q @ inner @ Q.mT
        return 0.5 * (out + out.mT)

    @staticmethod
    def backward(ctx, grad_out):
        M, lam, Q = ctx.saved_tensors
        alpha = ctx.alpha
        # one gradient per input: (M, H, lam, Q, alpha)
        grad_M = _SoftAbsCoMetric._apply_gamma(lam, Q, grad_out, alpha)
        grad_H = _SoftAbsD1._d2(lam, Q, grad_out, M, alpha)
        return grad_M, grad_H, None, None, None

    @staticmethod
    def jvp(ctx, M_t, H_t, _lam_t, _Q_t, _alpha_t):
        M, lam, Q = ctx.saved_tensors
        alpha = ctx.alpha
        # d/dt D1(M(t), H(t)) = D1(dM) + D^2[dH, M]
        out = _SoftAbsCoMetric._apply_gamma(lam, Q, M_t, alpha)
        if H_t is not None:
            out = out + _SoftAbsD1._d2(lam, Q, H_t, M, alpha)
        return out


class _SoftAbsCoMetric(torch.autograd.Function):
    """
    For a matrix H, computes its SoftAbs cometric reciprocal G^-1(H) = softabs_alpha(H)^-1, with an analytic, degeneracy-safe derivative.
    The SoftAbs map is a smooth approximation to the absolute value, and is
    defined as:
        G^-1(H) = Q diag(tanh(alpha*lam)/lam) Q^T,
    where lam, Q are the eigenpairs of H.
    Has analytic backward (Daleckii-Krein) rather than autodiff through ``torch.linalg.eigh``.

        dF = Q [Gamma * (Q^T dH Q)] Q^T

    And the map is self-adjoint so the pullback is the same expression.
    Necessary because eigh's backward carries separate
    1/(lam_i - lam_j) factors, which return NaN on the funnel's degenerate theta
    block; Gamma forms that ratio as one bounded quantity instead. Same
    formulation as Betancourt (2013) / Brofos & Lederman's ``_j_matrix``.

    alpha is a hyperparameter and is never differentiated.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(H, alpha):
        s = H.abs().amax(dim=-1).amax(dim=-1).clamp_min(1.0)
        s_mat = s.unsqueeze(-1).unsqueeze(-1)
        lam_n, Q = safe_eigh(H / s_mat)
        lam = lam_n * s.unsqueeze(-1)
        g = _softabs_g(lam, alpha)
        G_inv = torch.einsum("...ij,...j,...kj->...ik", Q, g, Q)
        return G_inv, lam, Q

    @staticmethod
    def setup_context(ctx, inputs, output):
        H, alpha = inputs
        _, lam, Q = output
        ctx.alpha = alpha
        ctx.save_for_backward(H, lam, Q)
        ctx.save_for_forward(H, lam, Q)

    @staticmethod
    def _apply_gamma(lam, Q, M, alpha):
        """Q [Gamma * (Q^T M Q)] Q^T, symmetrized -- serves as both the
        differential and its adjoint (Gamma symmetric, map self-adjoint)."""
        out = Q @ (_softabs_gamma(lam, alpha) * (Q.mT @ M @ Q)) @ Q.mT
        return 0.5 * (out + out.mT)

    @staticmethod
    def backward(ctx, grad_G_inv, *_):
        H, lam, Q = ctx.saved_tensors
        # Via _SoftAbsD1 so the backward is itself differentiable; inlining it
        # makes every second derivative come out identically zero.
        return _SoftAbsD1.apply(grad_G_inv, H, lam, Q, ctx.alpha), None

    @staticmethod
    def jvp(ctx, H_tangent, _alpha_tangent):
        H, lam, Q = ctx.saved_tensors
        dG = _SoftAbsD1.apply(H_tangent, H, lam, Q, ctx.alpha)
        # One tangent per output; lam/Q need explicit ZERO tangents -- returning
        # None for them trips an internal assert in torch's forward-AD.
        return dG, torch.zeros_like(lam), torch.zeros_like(Q)


def softabs_cometric(H: Tensor, alpha: float) -> Tensor:
    """
    SoftAbs cometric G^-1 = softabs_alpha(H)^-1 of a symmetric matrix H, with an
    analytic, degeneracy-safe derivative (see ``_SoftAbsCoMetric``).

    Parameters
    ----------
    H : Tensor (b, n, n)
        Batch of symmetric matrices ( eg the Hessian of the log density or a metric tensor).
    alpha : float
        SoftAbs sharpness. G^-1 -> |H|^-1 as alpha -> infinity.

    Returns
    -------
    Tensor (b, n, n)
        The SoftAbs cometric.
    """
    return _SoftAbsCoMetric.apply(H, alpha)[0]


class SoftAbsCometric(CoMetric):
    """
    Cometric that applies the SoftAbs regularization to a base cometric.
    'A General Metric for Riemannian Manifold Hamiltonian Monte Carlo'

    Parameters:
    -----------
    base_cometric : CoMetric
        The base cometric to regularize
    alpha : float
        Regularization parameter for the SoftAbs
    """

    def __init__(self, base_cometric: CoMetric, alpha: float = 1e3):
        super().__init__()
        if base_cometric.is_diag:
            raise NotImplementedError("SoftAbs for diagonal cometrics not implemented yet")
        self.base_cometric = base_cometric
        self.alpha = alpha

    def cometric_tensor(self, q: Tensor) -> Tensor:
        g = self.base_cometric.metric_tensor(q)
        if self.base_cometric.is_diag:
            raise NotImplementedError("SoftAbs for diagonal cometrics not implemented yet")
        g_soft = softabs_cometric(g, self.alpha)
        return g_soft

    def metric_tensor(self, q: Tensor) -> Tensor:
        g_soft = self.cometric_tensor(q)
        return torch.linalg.inv(g_soft)

    def forward(self, q: Tensor) -> Tensor:
        return self.cometric_tensor(q)
