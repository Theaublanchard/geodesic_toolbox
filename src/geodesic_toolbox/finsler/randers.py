import weakref

import torch
from torch import Tensor, nn

from ..cometric.base_class import CoMetric
from .base_class import FinslerMetric


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
        super().__init__()
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

        b = self.beta * self.omega(q)
        beta = torch.einsum("bi,bi->b", b, v)
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

    def legendre_transform(self, x: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Computes the Legendre transform of the Randers metric
        at the point x in the direction v.

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        p : Tensor (b,d)
            Cotangent vectors at x
        """
        G = self.base_cometric.metric_tensor(x)
        omega = self.beta * self.omega(x)

        if self.base_cometric.is_diag:
            v_norm = torch.sqrt(torch.einsum("bi,bi->b", G * v, v))
            Gv = G * v
        else:
            v_norm = torch.sqrt(torch.einsum("bij,bi,bj->b", G, v, v))
            Gv = torch.einsum("bij,bj->bi", G, v)

        beta = torch.einsum("bi,bi->b", omega, v)
        F = v_norm + beta

        v_norm = v_norm.clamp(min=eps)[:, None]  # Avoid division by zero

        p = F[:, None] * (Gv / v_norm + omega)
        return p


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
        super().__init__(
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


class BinetLegendreRanders(CoMetric):
    """
    Binet-Legendre cometric associated with a Randers metric.

    Parameters:
    -----------
    randers : RandersMetrics
        The Randers metric for which to compute the Binet-Legendre cometric.
    """

    def __init__(self, randers: RandersMetrics):
        super().__init__()
        self.randers = randers

    def cometric_tensor(self, x: Tensor) -> Tensor:
        base_cometric = self.randers.base_cometric.cometric_tensor(x)
        if self.randers.base_cometric.is_diag:
            base_cometric = torch.diag_embed(base_cometric)

        omega = self.randers.beta * self.randers.omega(x)
        beta_norm_squared = self.randers.base_cometric.dual_energy(x, omega)
        radius_squared = 1 / (1 - beta_norm_squared)

        omega_sharp = torch.einsum("bij,bj->bi", base_cometric, omega)
        omega_sharp_outer = torch.einsum("bi,bj->bij", omega_sharp, omega_sharp)
        c_sharp = radius_squared[:, None] * omega_sharp
        c_sharp_outer = torch.einsum("bi,bj->bij", c_sharp, c_sharp)

        m = base_cometric.shape[-1]
        C_inv = base_cometric + radius_squared[:, None, None] * omega_sharp_outer
        g_inv = radius_squared[:, None, None] * C_inv + (m + 2) * c_sharp_outer
        return g_inv

    def forward(self, x: Tensor) -> Tensor:
        return self.cometric_tensor(x)
