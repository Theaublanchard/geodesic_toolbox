import kmedoids
import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from .base_class import CoMetric


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
        K: int | None = None,
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
        else:
            self.K = self.centroids.shape[0]

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
        ], (
            f"Cometric centroids should be of shape (K,d) or (K,d,d), got {cometric_centroids.shape}"
        )
        assert cometric_centroids.shape[1] == self.centroids.shape[1], (
            f"Cometric centroids should have the same shape as centroids ({self.centroids.shape}), got {cometric_centroids.shape}"
        )

        # When diagonal cometric is used, cometric_centroids can be 2D
        if cometric_centroids.ndim == 2:
            self.is_diag = True
            return cometric_centroids
        else:
            assert cometric_centroids.shape[1] == cometric_centroids.shape[2], (
                f"Cometric centroids should be square matrices, got {cometric_centroids.shape}"
            )

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
                    "kd,kd,kd->k",
                    self.centroids,
                    self.cometric_centroids,
                    self.centroids,
                ).unsqueeze(0)  # (1,k)
            else:
                z_term = torch.einsum("bj,kij,bi->bk", z, self.cometric_centroids, z)
                cross_term = torch.einsum(
                    "bj,kij,ki->bk", z, self.cometric_centroids, self.centroids
                )
                c_term = torch.einsum(
                    "kj,kij,ki->k",
                    self.centroids,
                    self.cometric_centroids,
                    self.centroids,
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
        K: int | None = None,
        use_global_temp: bool = False,
    ):
        super().__init__(is_diag=True)

        assert centroids.ndim == 2, (
            f"Centroids should be of shape (K,d), got {centroids.shape}"
        )
        assert alpha > 0 and isinstance(alpha, int), (
            f"Alpha should be a positive integer, got {alpha}"
        )
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
    Generalization of the LAND cometric using RBF kernels instead of a single Gaussian kernel.
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
        K: int | None = None,
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
