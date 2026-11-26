import torch
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from .base_class import CoMetric, IdentityCoMetric
from .softabs import softabs_cometric


class FunctionalHeightMapCometric(CoMetric):
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
        chunk_size: int = 16,
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
            pbar.set_postfix({"Jacobian column": f"{i + 1}/{hw}"})
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
        pbar = tqdm(
            range(d),
            desc="Computing pullback metric via forward mode autodiff",
            leave=False,
        )
        for j in pbar:
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
    """
    Description of the Pullback information geometry metric.
    Here we only focus on the case where the decoder distributions are gaussians N(mu, sigma).
    Consequently, we know the exact expression of the Fisher-Rao metric and we can compute it without any approximation.
    This allows to then compute the actual pullback J^T G_{Fisher-Rao} J metric tensor.
    This approach is less memory intensive than the one using the KL divergence to compute the metric tensor.

    Here G_{Fisher-Rao} = diag(1/sigma^2, 2/sigma^2) where sigma is the standard deviation of the decoder distribution.
    So the metric is given by:
    G = J_mu^T diag(1/sigma^2) J_mu + J_std^T diag(2/sigma^2) J_std + reg_coef * I

    Parameters:
    ----------
    decoder : torch.nn.Module
        The stochastic decoder of a VAE model. It should have two branches:
        - decoder.mu_branch Tensor[b, d] -> Tensor[b, ...]
            It should return the mean of the decoder distribution.
        - decoder.std_branch Tensor[b, d] -> Tensor[b, ...]
            It should return the standard deviation sigma of the decoder distribution.
        And the regular forward method should be `decoder.complete_forward(z: torch.Tensor) -> Tuple[Tensor, Tensor]` returning the mean and log variance of the decoder distribution.
    reg_coef : float
        The regularization coefficient for the metric tensor. Default to 1e-3.
    """

    def __init__(self, decoder: torch.nn.Module, reg_coef: float = 1e-3):
        super().__init__()
        self.decoder = decoder
        self.reg_coef = reg_coef

        self.pb_mu = PullBackCometric(
            diffeo=self.decoder.mu_branch,
            method="finite_difference",
            reg_coef=0.0,
        )
        self.pb_std = PullBackCometric(
            diffeo=self.decoder.std_branch,
            method="finite_difference",
            reg_coef=0.0,
        )

    def metric_tensor(self, z: torch.Tensor) -> torch.Tensor:
        _mu, logvar = self.decoder.complete_forward(z)
        var = torch.exp(logvar.clamp(-8, 8)).flatten(start_dim=1)  # (batch_size, data_dim)
        J_mu = self.pb_mu.jacobian(z)  # (batch_size, data_dim, latent_dim)
        J_std = self.pb_std.jacobian(z)  # (batch_size, data_dim, latent_dim)

        inv_var = (1 / var).unsqueeze(-1)  # (batch_size, data_dim, 1)
        pb_mu = J_mu.mT @ (inv_var * J_mu)  # (batch_size, latent_dim, latent_dim)

        two_over_var = (2 / var).unsqueeze(-1)  # (batch_size, data_dim, 1)
        pb_var = J_std.mT @ (two_over_var * J_std)  # (batch_size, latent_dim, latent_dim)

        metric_tensor = pb_mu + pb_var + self.reg_coef * self.eye(z)
        return metric_tensor


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
        B, _d = x.shape
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
        B, _p = theta.shape
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
            # disgusting code
            g_inv = softabs_cometric(g.inverse(), alpha=self.softabs_alpha)
            g = g_inv.inverse()
        g += self.reg_coef * self.eye(theta)
        return g

    def forward(self, q: Tensor):
        g = self.metric_tensor(q)
        return torch.linalg.inv(g)
