"""
GEORCE solver with statistics (Riemannian and Finsler).
"""

from dataclasses import field

import torch
from torch import Tensor
from tqdm import tqdm

from ..cometric import CoMetric
from ..finsler import FinslerMetric
from .base import GeodesicDistanceSolver


class GEORCELineSearchStats:
    initial_energy: float
    final_energy: float
    num_iterations: int
    accepted: bool


class GEORCEIterationStats:
    iteration: int
    grad_norm: float
    distance: float
    energy: float
    alpha: float
    line_search: GEORCELineSearchStats


class GEORCERunStats:
    initial_distance: float | None
    initial_energy: float | None
    final_distance: float | None
    final_energy: float | None
    final_grad_norm: float | None
    converged: bool
    iterations: list[GEORCEIterationStats] = field(default_factory=list)
    line_search_warnings: int = 0
    n_iter: int = 0


class GEORCE(GeodesicDistanceSolver):
    """
    Computes the geodesic distances between points using the GEORCE algorithm.

    Paper : GEORCE: A Fast New Control Algorithm for  Computing Geodesics
    Code is a translation of the original code from :
    https://github.com/FrederikMR/georce/blob/main/georce/

    Parameters:
    ----------
    cometric : CoMetric
        The cometric to use for the geodesic distances.
    T : int
        The number of time steps to use for the geodesic trajectory.
    max_iter : int
        The maximum number of iterations to perform in the optimization.
    line_search_iter : int
        The maximum number of iterations to perform in the line search.
    tol : float
        The tolerance for the optimization. If the norm of the energy
        gradient is below this value, the optimization stops.
    c : float
        The constant used in the line search.
    alpha_0 : float
        The initial step size for the line search.
    rho : float
        The factor by which the step size is reduced in the line search.
    pbar : bool
        If True, a progress bar is displayed during the optimization. Default is False.
    """

    def __init__(
        self,
        cometric: CoMetric,
        T=100,
        max_iter=200,
        line_search_iter: int = 50,
        tol=1e-6,
        rho=0.5,
        c=0.9,
        alpha_0: float = 1.0,
        pbar: bool = False,
    ):
        super().__init__()
        self.cometric = cometric
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        self.rho = rho
        self.c = c
        self.alpha_0 = alpha_0
        self.pbar = pbar
        self.line_search_iter = line_search_iter

        self.georce_stats: GEORCERunStats | None = None

    def compute_energy(self, z_t: Tensor, z0: Tensor, zT: Tensor) -> Tensor:
        """
        Compute the energy of the geodesic trajectory.

        Parameters:
        ----------
        z_t: Tensor (T-1, d)
            The geodesic trajectory points. It does not include z0 and zT.
        z0: Tensor (d,)
            The starting point of the geodesic.
        zT: Tensor (d,)
            The ending point of the geodesic.
        Returns:
        -------
        energy: Tensor
            The total energy of the geodesic trajectory.
        """
        traj = torch.cat([z0[None, :], z_t, zT[None, :]], dim=0)  # (T+1, d)
        d_traj = torch.diff(traj, dim=0)  # (T, d)
        energy = self.cometric.energy(traj[:-1], d_traj)  # (T,)
        return 0.5 * energy.sum()

    def grad_E(self, z_t: Tensor, z0: Tensor, zT: Tensor) -> Tensor:
        """
        Compute the gradient of the energy with respect to the trajectory points.

        Parameters:
        ----------
        z_t: Tensor (T-1, d)
            The geodesic trajectory points. It does not include z0 and zT.
        z0: Tensor (d,)
            The starting point of the geodesic.
        zT: Tensor (d,)
            The ending point of the geodesic.

        Returns:
        -------
        grad: Tensor (T-1, d)
            The gradient of the energy with respect to the trajectory points z_t.
        """
        E = self.compute_energy(z_t, z0, zT)
        return torch.autograd.grad(E, z_t, materialize_grads=True)[0]

    def _energy_grad_norm(self, grad_E_t: Tensor) -> Tensor:
        """Compute normalized gradient norm used for GEORCE stopping criteria."""
        return torch.linalg.vector_norm(grad_E_t.reshape(-1)) / grad_E_t.numel()

    def get_mut_t(self, v_t: Tensor, G_inv_t: Tensor, diff: Tensor) -> Tensor:
        """
        Compute the optimal mu_t for the given v_t and G_inv_t.
        Broadly corresponds to line 7.

        Parameters:

        v_t: Tensor (T-1, d)
            The gradient of the energy at each time step.
        G_inv_t: Tensor (T, d, d) | (T,d)
            The inverse of the cometric at each time step.
        diff: Tensor (d,)
            The difference between the end points zT and z0.

        Returns:
        -------
        mu_t: Tensor (T, d)
            The optimal mu_t for the geodesic trajectory.
        """
        v_cumsum = torch.cumsum(v_t.flip(0), dim=0).flip(0)
        G_inv_sum = torch.sum(G_inv_t, dim=0)  # (d, d) | (d,)
        if G_inv_sum.ndim == 2:
            rhs = torch.einsum("tij,tj->ti", G_inv_t[:-1], v_cumsum)
        else:
            rhs = G_inv_t[:-1] * v_cumsum
        rhs = rhs.sum(dim=0) + 2.0 * diff  # (d,)

        # Solve the linear system
        if G_inv_sum.ndim == 2:
            mu_T = -torch.linalg.solve(G_inv_sum, rhs)  # (d,)
        else:
            mu_T = -rhs / G_inv_sum
        mu_t = torch.cat([mu_T[None, :] + v_cumsum, mu_T[None, :]], dim=0)  # (T, d)
        return mu_t  # (T, d)

    def line_search(self, x_0, x_T, u_t, u_t_i, x_t_i) -> tuple[float, GEORCELineSearchStats]:
        """
        Perform a Armiijo's line search to find the optimal step size alpha.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        u_t: Tensor (T, d)
            The current velocity at each time step.
        u_t_i: Tensor (T, d)
            The initial guess of the velocity at each time step.
        x_t_i: Tensor (T-1, d)
            The current trajectory points. It does not include x_0 and x_T.

        Returns:
        -------
        alpha: float
            The optimal step size found by the line search.
        stats: GEORCELineSearchStats
            The statistics of the line search, including initial and final energy, number of iterations, and whether the step size was accepted.
        """
        # Compute the initial energy
        E_0 = self.compute_energy(x_t_i, x_0, x_T)
        grad_E_0 = self.grad_E(x_t_i, x_0, x_T).reshape(-1)

        p_k = -grad_E_0  # Initial search direction
        alpha = self.alpha_0  # Initial step size

        curr_iter = 0

        # Evaluate new energy without building a graph (Armijo condition only)
        with torch.no_grad():
            x_t_new = x_0 + torch.cumsum(alpha * u_t[:-1] + (1 - alpha) * u_t_i[:-1], dim=0)
            E_new = self.compute_energy(x_t_new, x_0, x_T)

        # Compute Armijo's condition
        val = E_0 + self.c * alpha * torch.dot(grad_E_0, p_k)
        condition = E_new > val

        while condition and curr_iter < self.line_search_iter:
            alpha *= self.rho  # Reduce step size
            curr_iter += 1

            with torch.no_grad():
                x_t_new = x_0 + torch.cumsum(
                    alpha * u_t[:-1] + (1 - alpha) * u_t_i[:-1], dim=0
                )
                E_new = self.compute_energy(x_t_new, x_0, x_T)
            val = E_0 + self.c * alpha * torch.dot(grad_E_0, p_k)
            condition = E_new > val

        stats = GEORCELineSearchStats(
            initial_energy=E_0.detach().item(),
            final_energy=E_new.detach().item(),
            num_iterations=curr_iter,
            accepted=curr_iter < self.line_search_iter,
        )

        return alpha, stats

    def dst_func(self, x_0: Tensor, x_T: Tensor, x_t: Tensor) -> Tensor:
        """
        Compute the geodesic distance between x_0 and x_T along the trajectory x_t.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        x_t: Tensor (T-2, d)
            The geodesic trajectory points. It does not include x_0 and x_T.

        Returns:
        --------
        distance: Tensor
            The geodesic distance between x_0 and x_T along the trajectory x_t.
        """
        full_traj = torch.cat([x_0[None, :], x_t, x_T[None, :]], dim=0)  # (T, d)
        dx = full_traj[1:] - full_traj[:-1]  # (T-1, d)
        distance = self.cometric.metric(full_traj[:-1], dx)  # (T-1,)
        return distance.sum()

    def main_stop_cond(self, norm_grad_E_t: Tensor, i: int) -> bool:
        """
        Check the stopping condition for the GEORCE algorithm.

        Parameters:
        ----------
        norm_grad_E_t: Tensor
            The norm of the gradient of the energy at iteration t.
        i: int
            The current iteration number.

        Returns:
        -------
        stop: bool
            True if the stopping condition is met, False otherwise.
        """
        return (norm_grad_E_t <= self.tol) | (i >= self.max_iter)

    def georce_solver(
        self,
        x_0: Tensor,
        x_T: Tensor,
        x_t_0: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Solve the geodesic equation using the GEORCE algorithm.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        x_t_0: Tensor (T-2, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_T.
            It must not contain x_0 and x_T.

        Returns:
        -------
        x_final: Tensor (T, d)
            The final geodesic trajectory including x_0 and x_T.
        dst_list: Tensor (T+1,)
            The geodesic distance at each iteration.
        norm_gE_list: Tensor (T+1,)
            The norm of the gradient of the energy at each iteration.
        E_list: Tensor (T+1,)
            The energy at each iteration.
        alpha_list: Tensor (T+1,)
            The step size alpha at each iteration.
        """
        # Initialize everything
        i = 0
        d = x_0.shape[0]
        self.georce_stats = GEORCERunStats(
            initial_distance=None,
            initial_energy=None,
            final_distance=None,
            final_energy=None,
            final_grad_norm=None,
            converged=False,
            iterations=[],
            line_search_warnings=0,
            n_iter=0,
        )

        if x_t_0 is None:
            t = torch.linspace(0, 1, self.T, device=x_0.device, dtype=x_0.dtype)
            x_t_0 = x_0[None, :] + t[1:-1, None] * (x_T - x_0)[None, :]  # (T-2, d)
        else:
            assert x_t_0.shape == (
                self.T - 2,
                d,
            ), (
                f"x_t_0 must have shape {(self.T - 2, d)=} got {x_t_0.shape=}. Be sure to exclude x_0 and x_T."
            )

        # Precompute cometric at start without building graph
        with torch.no_grad():
            G_inv_0 = self.cometric.cometric_tensor(x_0[None, :]).squeeze(0)

        diff = x_T - x_0

        x_t_i = x_t_0.clone().requires_grad_(True)  # (T-2, d), not including x_0 and x_T
        u_t_i = (
            diff * torch.ones(self.T - 1, d, device=x_0.device, dtype=x_0.dtype) / (self.T - 1)
        )  # (T-1, d) Initial guess of segment velocities not including x_T

        # L4
        grad_E_t = self.grad_E(x_t_i, x_0, x_T)
        norm_grad_E_t = self._energy_grad_norm(grad_E_t)

        with torch.no_grad():
            initial_distance = self.dst_func(x_0, x_T, x_t_i).item()
            initial_energy = self.compute_energy(x_t_i, x_0, x_T).item()
        dst_list = [initial_distance]
        norm_gE_list = [norm_grad_E_t.item()]
        E_list = [initial_energy]
        alpha_list = [self.alpha_0]

        self.georce_stats.initial_distance = initial_distance
        self.georce_stats.initial_energy = initial_energy

        if self.pbar:
            pbar = tqdm(range(self.max_iter), total=self.max_iter, desc="GEORCE")
        # for i in pbar:
        while not self.main_stop_cond(norm_grad_E_t, i):
            # L5
            G_inv_t = self.cometric.cometric_tensor(x_t_i)
            G_inv_t = torch.cat(
                [G_inv_0[None, :], G_inv_t], dim=0
            )  # (T-1, d, d) | (T-1, d, d)

            # L6
            if G_inv_t.ndim == 3:
                # Avoid explicit inverse: solve G_inv * x = u, then use u * x.
                G_u = torch.linalg.solve(G_inv_t[1:], u_t_i[1:].unsqueeze(-1)).squeeze(-1)
                v_t = u_t_i[1:] * G_u  # (T-2, d)
            else:
                G_t = 1 / G_inv_t  # (T-1,d)
                v_t = u_t_i[1:] * G_t[1:] * u_t_i[1:]  # (T-2, d)
            v_t = torch.autograd.grad(
                v_t.sum(),
                x_t_i,
                materialize_grads=True,
            )[0]

            # L7
            mu_t = self.get_mut_t(v_t, G_inv_t, diff)  # (T-1, d)

            # L8
            if G_inv_t.ndim == 3:
                u_t = -0.5 * torch.einsum("tij,tj->ti", G_inv_t, mu_t)  # (T-1, d)
            else:
                u_t = -0.5 * G_inv_t * mu_t  # (T-1, d)
            # L9/19
            alpha, ls_stats = self.line_search(x_0, x_T, u_t, u_t_i, x_t_i)

            if not ls_stats.accepted:
                self.georce_stats.line_search_warnings += 1

            # L11/L12: update state without tracking to avoid graph growth
            with torch.no_grad():
                u_t_i = alpha * u_t + (1 - alpha) * u_t_i
                x_t_i = x_0 + torch.cumsum(u_t_i[:-1], dim=0)  # (T-2, d)
            # Re-enable grad only on trajectory points for next iteration
            x_t_i = x_t_i.detach().requires_grad_(True)

            # Prepare stop condition, ie L4
            grad_E_t = self.grad_E(x_t_i, x_0, x_T)
            norm_grad_E_t = self._energy_grad_norm(grad_E_t)
            i += 1

            # Logging
            with torch.no_grad():
                dst = self.dst_func(x_0, x_T, x_t_i).item()
                E = self.compute_energy(x_t_i, x_0, x_T).item()
            dst_list.append(dst)
            norm_gE_list.append(norm_grad_E_t.item())
            E_list.append(E)
            alpha_list.append(alpha)

            iter_stats = GEORCEIterationStats(
                iteration=i,
                grad_norm=norm_grad_E_t.item(),
                distance=dst,
                energy=E,
                alpha=alpha,
                line_search=ls_stats,
            )
            self.georce_stats.iterations.append(iter_stats)

            if self.pbar:
                pbar.set_description(
                    f"iter={i:0>3} alpha={alpha:.3E} E={E:.3E} grad={norm_grad_E_t.item():.3E} dst={dst:.3E}"
                )
                pbar.update(1)

        self.georce_stats.final_distance = dst_list[-1]
        self.georce_stats.final_energy = E_list[-1]
        self.georce_stats.final_grad_norm = norm_gE_list[-1]
        self.georce_stats.n_iter = i
        self.georce_stats.converged = bool(norm_grad_E_t <= self.tol)

        x_final = torch.cat([x_0[None, :], x_t_i, x_T[None, :]], dim=0)  # (T, d)
        dst_list = torch.tensor(dst_list)
        norm_gE_list = torch.tensor(norm_gE_list)
        E_list = torch.tensor(E_list)
        return x_final, dst_list, norm_gE_list, E_list, alpha_list

    def get_trajectories(self, x_0: Tensor, x_1: Tensor, x_t_0: Tensor = None) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Parameters:
        ----------
        x_0: Tensor (B, d)
            The starting points of the geodesic.
        x_1: Tensor (B, d)
            The ending points of the geodesic.
        x_t_0: Tensor (B,T-2, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_1.
            It must not contain x_0 and x_1.

        Returns:
        -------
        trajectories: Tensor (B, T, d)
            The geodesic trajectories between x_0 and x_1 (both included).
        """
        trajectories = []
        B, d = x_0.shape
        assert x_1.shape == (B, d), f"Both tensors must have the same shape {(B, d)=}"
        x_0.requires_grad_(True)
        x_1.requires_grad_(True)
        if x_t_0 is not None:
            x_t_0.requires_grad_(True)
        for b in range(B):
            if x_t_0 is not None:
                x_final, _, _, _, _ = self.georce_solver(x_0[b], x_1[b], x_t_0=x_t_0[b])
            else:
                x_final, _, _, _, _ = self.georce_solver(x_0[b], x_1[b])
            trajectories.append(x_final)
        trajectories = torch.stack(trajectories, dim=0)  # (B, T, d)
        return trajectories

    def forward(self, x_0: Tensor, x_T: Tensor, x_t_0: Tensor = None) -> Tensor:
        """
        Compute the geodesic distance between two points x_0 and x_T.

        Parameters:
        ----------
        x_0: Tensor (B, d)
            The starting points of the geodesic.
        x_T: Tensor (B, d)
            The ending points of the geodesic.
        x_t_0: Tensor (B,T-2, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_1.
            It must not contain x_0 and x_T

        Returns:
        -------
        dst: Tensor (B,)
            The geodesic distances between x_0 and x_T.
        """
        B, d = x_0.shape
        assert x_T.shape == (B, d), f"Both tensors must have the same shape {(B, d)=}"
        x_0.requires_grad_(True)
        x_T.requires_grad_(True)
        if x_t_0 is not None:
            x_t_0.requires_grad_(True)

        dst = torch.zeros(B, device=x_0.device)

        for b in range(B):
            if x_t_0 is not None:
                x_final, _, _, _, _ = self.georce_solver(x_0[b], x_T[b], x_t_0=x_t_0[b])
            else:
                x_final, _, _, _, _ = self.georce_solver(x_0[b], x_T[b])
            dst[b] = self.dst_func(x_0[b], x_T[b], x_final[1:-1])
        return dst


class GEORCEFinsler(torch.nn.Module):
    """
    Computes the geodesic distances between points using the GEORCE algorithm
    on a Finsler metric.

    Paper : GEORCE: A Fast New Control Algorithm for  Computing Geodesics
    Code is a translation of the original code from :
    https://github.com/FrederikMR/georce/blob/main/georce/

    Parameters:
    ----------
    finsler : FinslerMetric
        The Finsler metric to use for the geodesic distances.
    T : int
        The number of time steps to use for the geodesic trajectory.
    max_iter : int
        The maximum number of iterations to perform in the optimization.
    line_search_iter : int
        The maximum number of iterations to perform in the line search.
    tol : float
        The tolerance for the optimization. If the norm of the energy
        gradient is below this value, the optimization stops.
    c : float
        The constant used in the line search.
    alpha_0 : float
        The initial step size for the line search.
    rho : float
        The factor by which the step size is reduced in the line search.
    pbar : bool
        If True, a progress bar is displayed during the optimization. Default is False.
    """

    def __init__(
        self,
        finsler: FinslerMetric,
        T=100,
        max_iter=200,
        line_search_iter: int = 50,
        tol=1e-6,
        rho=0.5,
        c=0.9,
        alpha_0: float = 1.0,
        pbar: bool = False,
    ):
        super().__init__()
        self.finsler = finsler
        self.T = T
        self.max_iter = max_iter
        self.line_search_iter = line_search_iter
        self.tol = tol
        self.rho = rho
        self.c = c
        self.alpha_0 = alpha_0
        self.pbar = pbar
        self.georce_finsler_stats: GEORCERunStats | None = None

    def compute_distance(
        self, traj: torch.Tensor, tangent_vectors: torch.Tensor = None
    ) -> torch.Tensor:
        """Given a trajectory and the tangent vectors, compute the distance
        under the finsler metric.

        Parameters
        ----------
        traj : torch.Tensor (b,T,d)
            The trajectory. There are b trajectories of T points in d dimensions
        tangent_vectors : torch.Tensor (b,T,d)
            The tangent vectors at each point of the trajectory
            If None, the tangent vectors are computed as the difference between consecutive points.
        Returns
        -------
        dst : torch.Tensor (b,)
            The distance between the two points
        """
        if tangent_vectors is None:
            # Compute the tangent vectors as the difference between consecutive points
            tangent_vectors = torch.zeros_like(traj)
            tangent_vectors[:, :-1, :] = traj[:, 1:, :] - traj[:, :-1, :]
            tangent_vectors[:, -1, :] = traj[:, -1, :] - traj[:, -2, :]
        distances = torch.stack(
            [self.finsler(m, seg) for m, seg in zip(traj, tangent_vectors)]
        )  # (B, T)
        distances = distances.relu().sum(dim=1)  # (B,)
        return distances

    def compute_energy(self, z_t: Tensor, z0: Tensor, zT: Tensor, dx=None) -> Tensor:
        """
        Compute the energy of the geodesic trajectory.

        Parameters:
        ----------
        z_t: Tensor (T-2, d)
            The geodesic trajectory points. It does not include z0 and zT.
        z0: Tensor (d,)
            The starting point of the geodesic.
        zT: Tensor (d,)
            The ending point of the geodesic.
        dx: Tensor (T-1, d)
            The difference between consecutive points in the trajectory.
            If not provided, it will be computed as z_t[1:] - z_t[:-1].
        Returns:
        -------
        energy: Tensor
            The total energy of the geodesic trajectory.
        """
        traj = torch.cat([z0[None, :], z_t, zT[None, :]], dim=0)  # (T, d)
        if dx is None:
            dx = traj[1:] - traj[:-1]  # (T-1, d)
        energy = self.finsler(traj[:-1], dx) ** 2  # (T-1,)
        return energy.sum()

    def dot_sum(self, x_t: Tensor, u_t: Tensor) -> Tensor:
        """
        Compute the sum of the dot product of the velocity with the fundamental tensor.

        Parameters:
        ----------
        x_t: Tensor (T,d)
            The points.
        u_t: Tensor (T,d)
            The velocity at each time step.

        Returns:
        -------
        dot: Tensor
            The sum of the dot product of the velocity with the fundamental tensor.
        """
        G_t = self.finsler.fundamental_tensor(x_t, u_t)
        dot = torch.einsum("ti,tij,tj->t", u_t, G_t, u_t)
        return dot.sum()

    def dot_sum_u(self, x_t: Tensor, u_t_0: Tensor, u_t_1: Tensor) -> Tensor:
        """
        Compute the sum of the dot product of the velocity with the fundamental tensor.
        The fundamental tensor is computed at the points x_t with the velocities u_t_0.
        The second velocity u_t_1 is used to compute the energy.

        Parameters
        ----------
        x_t: Tensor (T,d)
            The points.
        u_t_0: Tensor (T,d)
            The velocity at each time step used to compute the fundamental tensor.
        u_t_1: Tensor (T,d)
            The velocity at each time step used to compute the energy.

        Returns
        -------
        dot: Tensor
            The sum of the dot product of the velocity with the fundamental tensor.
        """
        G_t = self.finsler.fundamental_tensor(x_t, u_t_0)
        dot = torch.einsum("ti,tij,tj->t", u_t_1, G_t, u_t_1)
        return dot.sum()

    def get_v_t(self, x_t_i: Tensor, u_t_i: Tensor) -> Tensor:
        """
        Compute the gradient of the energy with respect to the trajectory points.

        Parameters:
        ----------
        x_t_i: Tensor (T, d)
            The geodesic trajectory points.
        u_t_i: Tensor (T, d)
            The velocity at each time step.

        Returns:
        -------
        v_t: Tensor (T, d)
            The gradient of the energy with respect to the trajectory points x_t_i.
        """
        # v_t = torch.func.grad(self.dot_sum, (x_t_i, u_t_i), argnums=(0,))
        v_t_func = torch.func.grad(self.dot_sum, argnums=(0,))
        v_t = v_t_func(x_t_i, u_t_i)[0]
        return v_t

    def get_zeta_t(self, x_t_i: Tensor, u_t_i: Tensor) -> Tensor:
        """
        Compute the gradient of the energy with respect to the velocity.
        Parameters:
        ----------
        x_t_i: Tensor (T, d)
            The geodesic trajectory points.
        u_t_i: Tensor (T, d)
            The velocity at each time step.

        Returns:
        -------
        zeta_t: Tensor (T, d)
            The gradient of the energy with respect to the velocity u_t_i.
        """
        zeta_t_func = torch.func.grad(self.dot_sum_u, argnums=1)
        zeta_t = zeta_t_func(x_t_i, u_t_i, u_t_i)[0]
        return zeta_t

    def get_mut_t(self, v_t: Tensor, zeta_t: Tensor, G_inv_t: Tensor, diff: Tensor) -> Tensor:
        """
        Compute the optimal mu_t for the given v_t and G_inv_t.
        Broadly corresponds to line 8.

        Parameters:
        ----------
        v_t: Tensor (T-1, d)
            The gradient of the energy relative to the position at each time step.
        zeta_t: Tensor (T-1, d)
            The gradient of the energy relative to the velocity at each time step.
        G_inv_t: Tensor (T, d, d)
            The inverse of the cometric at each time step.
        diff: Tensor (d,)
            The difference between the end points zT and z0.

        Returns:
        -------
        mu_t: Tensor (T, d)
            The optimal mu_t for the geodesic trajectory.
        """
        v_cumsum = torch.cumsum(v_t.flip(0), dim=0).flip(0)
        G_inv_sum = torch.sum(G_inv_t, dim=0)  # (d, d)
        rhs = torch.einsum("tij,tj->ti", G_inv_t[:-1], v_cumsum + zeta_t)
        rhs = rhs.sum(dim=0) + 2.0 * diff

        # Solve the linear system
        mu_T = -torch.linalg.solve(G_inv_sum, rhs)  # (d,)
        mu_t = torch.cat([mu_T[None, :] + v_cumsum + zeta_t, mu_T[None, :]], dim=0)  # (T, d)
        return mu_t  # (T, d)

    def _energy_grad_norm(self, grad_E_t: Tensor) -> Tensor:
        """Compute normalized gradient norm used for GEORCE stopping criteria."""
        return torch.linalg.vector_norm(grad_E_t.reshape(-1)) / grad_E_t.numel()

    def line_search(
        self, x_0: Tensor, x_T: Tensor, u_t: Tensor, u_t_i: Tensor, x_t_i: Tensor
    ) -> tuple[float, GEORCELineSearchStats]:
        """
        Perform a Armiijo's line search to find the optimal step size alpha.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        u_t: Tensor (T, d)
            The current velocity at each time step.
        u_t_i: Tensor (T, d)
            The initial guess of the velocity at each time step.
        x_t_i: Tensor (T-1, d)
            The current trajectory points. It does not include x_0 and x_T.

        Returns:
        -------
        alpha: float
            The optimal step size found by the line search.
        stats: GEORCELineSearchStats
            The statistics of the line search, including initial and final energy, number of iterations, and whether the step size was accepted.
        """
        # Compute the initial energy
        E_0 = self.compute_energy(x_t_i, x_0, x_T)
        grad_E_0 = torch.autograd.grad(E_0, x_t_i, materialize_grads=True)[0].reshape(-1)

        p_k = -grad_E_0  # Initial search direction
        alpha = self.alpha_0  # Initial step size

        curr_iter = 0

        # Evaluate candidate energies without building a graph (Armijo condition only)
        with torch.no_grad():
            x_t_new = x_0 + torch.cumsum(alpha * u_t[:-1] + (1 - alpha) * u_t_i[:-1], dim=0)
            E_new = self.compute_energy(x_t_new, x_0, x_T)

        # Compute Armijo's condition
        val = E_0 + self.c * alpha * torch.dot(grad_E_0, p_k)
        condition = E_new > val

        while condition and curr_iter < self.line_search_iter:
            alpha *= self.rho  # Reduce step size
            curr_iter += 1

            with torch.no_grad():
                x_t_new = x_0 + torch.cumsum(
                    alpha * u_t[:-1] + (1 - alpha) * u_t_i[:-1], dim=0
                )
                E_new = self.compute_energy(x_t_new, x_0, x_T)
            val = E_0 + self.c * alpha * torch.dot(grad_E_0, p_k)
            condition = E_new > val

        stats = GEORCELineSearchStats(
            initial_energy=E_0.detach().item(),
            final_energy=E_new.detach().item(),
            num_iterations=curr_iter,
            accepted=curr_iter < self.line_search_iter,
        )

        return alpha, stats

    def dst_func(self, x_0: Tensor, x_T: Tensor, x_t: Tensor) -> Tensor:
        """
        Compute the geodesic distance between x_0 and x_T along the trajectory x_t.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        x_t: Tensor (T-2, d)
            The geodesic trajectory points. It does not include x_0 and x_T.

        Returns:
        --------
        distance: Tensor
            The geodesic distance between x_0 and x_T along the trajectory x_t.
        """
        full_traj = torch.cat([x_0[None, :], x_t, x_T[None, :]], dim=0)  # (T, d)
        dx = full_traj[1:] - full_traj[:-1]  # (T-1, d)
        distance = self.finsler(full_traj[:-1], dx).abs()  # (T-1,)
        return distance.sum()

    def main_stop_cond(self, norm_grad_E_t: Tensor, i: int) -> bool:
        """
        Check the stopping condition for the GEORCE algorithm.

        Parameters:
        ----------
        norm_grad_E_t: Tensor
            The norm of the gradient of the energy at iteration t.
        i: int
            The current iteration number.

        Returns:
        -------
        stop: bool
            True if the stopping condition is met, False otherwise.
        """
        stats = self.georce_stats
        if stats is None:
            return (norm_grad_E_t <= self.tol) | (i >= self.max_iter)

        # # Else we also check if the gradient norm is not decreasing for a long time
        # # Combine this with stationnary energy to avoid stopping too early when the optimization is stuck in a plateau
        # if len(stats.iterations) > 10:
        #     recent_iters = stats.iterations[-10:]
        #     grad_norms = torch.tensor([iter.grad_norm for iter in recent_iters])
        #     energies = torch.tensor([iter.energy for iter in recent_iters])
        #     grad_norm_decreasing = (grad_norms[0] - grad_norms[-1]) / (grad_norms[0] + 1e-8) > 1e-3
        #     energy_stationary = (energies.max() - energies.min()) / (energies.max() + 1e-8) < 1e-3
        #     if not grad_norm_decreasing and energy_stationary:
        #         return True
        return (norm_grad_E_t <= self.tol) | (i >= self.max_iter)

    def georce_solver(
        self,
        x_0: Tensor,
        x_T: Tensor,
        x_t_0: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Solve the geodesic equation using the GEORCE algorithm.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        x_t_0: Tensor (T-2, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_T.
            It must not contain x_0 and x_T.

        Returns:
        -------
        x_final: Tensor (T, d)
            The final geodesic trajectory including x_0 and x_T.
        dst_list: Tensor (T+1,)
            The geodesic distance at each iteration.
        norm_gE_list: Tensor (T+1,)
            The norm of the gradient of the energy at each iteration.
        E_list: Tensor (T+1,)
            The energy at each iteration.
        alpha_list: Tensor (T+1,)
            The step size alpha at each iteration.
        """
        # Initialize everything
        i = 0
        d = x_0.shape[0]

        self.georce_stats = GEORCERunStats(
            initial_distance=None,
            initial_energy=None,
            final_distance=None,
            final_energy=None,
            final_grad_norm=None,
            converged=False,
            iterations=[],
            line_search_warnings=0,
            n_iter=0,
        )

        if x_t_0 is None:
            t = torch.linspace(0, 1, self.T, device=x_0.device, dtype=x_0.dtype)
            x_t_0 = x_0[None, :] + t[1:-1, None] * (x_T - x_0)[None, :]  # (T-2, d)
        else:
            assert x_t_0.shape == (
                self.T - 2,
                d,
            ), (
                f"x_t_0 must have shape {(self.T - 2, d)=} got {x_t_0.shape=}. Be sure to exclude x_0 and x_T."
            )

        diff = x_T - x_0
        # (T-2, d), not including x_0 and x_T
        x_t_i = x_t_0.clone().requires_grad_(True)
        # (T-1, d) Initial guess of segment velocities
        u_t_i = (
            diff * torch.ones(self.T - 1, d, device=x_0.device, dtype=x_0.dtype) / (self.T - 1)
        )

        # Precompute fundamental tensor at start without building graph
        with torch.no_grad():
            G_0 = self.finsler.fundamental_tensor(
                x_0[None, :], u_t_i[0, :].unsqueeze(0)
            ).squeeze(0)

        # L4
        grad_E_t = torch.autograd.grad(
            self.compute_energy(x_t_i, x_0, x_T), x_t_i, materialize_grads=True
        )[0]  # (T-1, d)
        norm_grad_E_t = self._energy_grad_norm(grad_E_t)
        # norm_grad_E_before = norm_grad_E_t.clone()

        with torch.no_grad():
            initial_distance = self.dst_func(x_0, x_T, x_t_i).item()
            initial_energy = self.compute_energy(x_t_i, x_0, x_T).item()
        dst_list = [initial_distance]
        norm_gE_list = [norm_grad_E_t.item()]
        E_list = [initial_energy]
        alpha_list = [self.alpha_0]

        self.georce_stats.initial_distance = initial_distance
        self.georce_stats.initial_energy = initial_energy

        if self.pbar:
            pbar = tqdm(range(self.max_iter), total=self.max_iter, desc="GEORCE")

        # while (norm_grad_E_t > self.tol) and (i < self.max_iter) and (norm_grad_E_t / (norm_grad_E_before + 1e-8) > 1e-3):
        while not self.main_stop_cond(norm_grad_E_t, i):
            # norm_grad_E_before = norm_grad_E_t.clone()

            # L5
            G_t = self.finsler.fundamental_tensor(x_t_i, u_t_i[1:])  # (T-2, d, d)
            G_t = torch.cat([G_0[None, :], G_t], dim=0)  # (T-1, d, d)
            G_inv_t = G_t.inverse()  # (T-1, d, d)

            # L6/7
            v_t = self.get_v_t(x_t_i, u_t_i[1:])  # (T-2, d)
            zeta_t = self.get_zeta_t(x_t_i, u_t_i[1:])  # (T-2, d)

            # L8
            mu_t = self.get_mut_t(v_t, zeta_t, G_inv_t, diff)

            # L9
            u_t = -0.5 * torch.einsum("tij,tj->ti", G_inv_t, mu_t)  # (T-1, d)

            # L10/11
            alpha, ls_stats = self.line_search(x_0, x_T, u_t, u_t_i, x_t_i)

            if not ls_stats.accepted:
                self.georce_stats.line_search_warnings += 1

            # L12/L13: update state without tracking to avoid graph growth
            with torch.no_grad():
                u_t_i = alpha * u_t + (1.0 - alpha) * u_t_i
                x_t_i = x_0 + torch.cumsum(u_t_i[:-1], dim=0)  # (T-2, d)
            # Re-enable grad only on trajectory points for next iteration
            x_t_i = x_t_i.detach().requires_grad_(True)

            # Prepare stop condition, ie L4
            grad_E_t = torch.autograd.grad(
                self.compute_energy(x_t_i, x_0, x_T), x_t_i, materialize_grads=True
            )[0]
            # (T-2, d)
            norm_grad_E_t = self._energy_grad_norm(grad_E_t)
            i += 1

            # Logging
            with torch.no_grad():
                dst = self.dst_func(x_0, x_T, x_t_i).item()
                E = self.compute_energy(x_t_i, x_0, x_T).item()
            dst_list.append(dst)
            norm_gE_list.append(norm_grad_E_t.item())
            E_list.append(E)
            alpha_list.append(alpha)

            iter_stats = GEORCEIterationStats(
                iteration=i,
                grad_norm=norm_grad_E_t.item(),
                distance=dst,
                energy=E,
                alpha=alpha,
                line_search=ls_stats,
            )
            self.georce_stats.iterations.append(iter_stats)

            if self.pbar:
                pbar.set_description(
                    f"iter={i:0>3} alpha={alpha:.3E} E={E:.3E} grad={norm_grad_E_t.item():.3E} dst={dst:.3E}"
                )
                pbar.update(1)

        if norm_grad_E_t.isnan():
            self.georce_stats.line_search_warnings += 1
            t = torch.linspace(0, 1, self.T, device=x_0.device, dtype=x_0.dtype)
            x_t_i = x_0[None, :] + t[1:-1, None] * (x_T - x_0)[None, :]  # (T-2, d)

        self.georce_stats.final_distance = dst_list[-1]
        self.georce_stats.final_energy = E_list[-1]
        self.georce_stats.final_grad_norm = norm_gE_list[-1]
        self.georce_stats.n_iter = i
        self.georce_stats.converged = bool(norm_grad_E_t <= self.tol)

        x_final = torch.cat([x_0[None, :], x_t_i, x_T[None, :]], dim=0)  # (T, d)
        dst_list = torch.tensor(dst_list)
        norm_gE_list = torch.tensor(norm_gE_list)
        E_list = torch.tensor(E_list)
        return x_final, dst_list, norm_gE_list, E_list, alpha_list

    def get_trajectories(self, x_0: Tensor, x_1: Tensor, x_t_0: Tensor = None) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Parameters:
        ----------
        x_0: Tensor (B, d)
            The starting points of the geodesic.
        x_1: Tensor (B, d)
            The ending points of the geodesic.
        x_t_0: Tensor (B,T-2, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_1.
            It must not contain x_0 and x_1.

        Returns:
        -------
        trajectories: Tensor (B, T, d)
            The geodesic trajectories between x_0 and x_1 (both included).
        """
        trajectories = []
        B, d = x_0.shape
        assert x_1.shape == (B, d), f"Both tensors must have the same shape {(B, d)=}"
        for b in range(B):
            if x_t_0 is not None:
                x_final, _, _, _, _ = self.georce_solver(x_0[b], x_1[b], x_t_0=x_t_0[b])
            else:
                x_final, _, _, _, _ = self.georce_solver(x_0[b], x_1[b])
            trajectories.append(x_final)
        trajectories = torch.stack(trajectories, dim=0)  # (B, T, d)
        return trajectories

    def forward(self, x_0: Tensor, x_T: Tensor, x_t_0: Tensor = None) -> Tensor:
        """
        Compute the geodesic distance between two points x_0 and x_T.

        Parameters:
        ----------
        x_0: Tensor (B, d)
            The starting points of the geodesic.
        x_T: Tensor (B, d)
            The ending points of the geodesic.
        x_t_0: Tensor (B,T-2, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_1.
            It must not contain x_0 and x_T

        Returns:
        -------
        dst: Tensor (B,)
            The geodesic distances between x_0 and x_T.
        """
        B, d = x_0.shape
        assert x_T.shape == (B, d), f"Both tensors must have the same shape {(B, d)=}"

        dst = torch.zeros(B, device=x_0.device)

        for b in range(B):
            if x_t_0 is not None:
                x_final, _, _, _, _ = self.georce_solver(x_0[b], x_T[b], x_t_0=x_t_0[b])
            else:
                x_final, _, _, _, _ = self.georce_solver(x_0[b], x_T[b])
            dst[b] = self.dst_func(x_0[b], x_T[b], x_final[1:-1])
        return dst
