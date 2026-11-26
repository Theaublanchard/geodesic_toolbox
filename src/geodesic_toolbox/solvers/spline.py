"""
Spline-based geodesic solvers (Riemannian and Finsler).
"""

import torch
import torchcurves.functional as tcf
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from ..cometric import CoMetric
from ..finsler import FinslerMetric
from .base import GeodesicDistanceSolver
from .graph import SolverGraph, SolverGraphFinsler


class GeodesicSpline(torch.nn.Module):
    """
    BSplined parameterization of a geodesic curve between two points q0 and q1.

    Parameters
    ----------
    q0 : torch.Tensor (B, D)
        The starting point of the geodesic.
    q1 : torch.Tensor (B, D)
        The ending point of the geodesic.
    n_control_points : int
        The number of control points for the B-spline.
    degree : int
        The degree of the B-spline.
    control_points : torch.Tensor (B, n_control_points, D), optional
        The control points for the B-spline. Must include q0 and q1 as the first and last control points, respectively.
        If not None, will override the n_control_points parameter.
        If None, they will be initialized as a linear interpolation between q0 and q1.
    """

    def __init__(
        self,
        q0,
        q1,
        n_control_points,
        degree,
        control_points=None,
    ):
        super().__init__()

        self.register_buffer("q0", q0.clone().detach())
        self.register_buffer("q1", q1.clone().detach())

        if control_points is not None:
            n_control_points = control_points.shape[1]

        self.degree = degree
        self.n_control_points = n_control_points

        if control_points is None:
            t = torch.linspace(0, 1, n_control_points, device=q0.device, dtype=q0.dtype)
            control_points = q0[:, None, :] + t[None, :, None] * (q1 - q0)[:, None, :]

        # Only optimize interior control points
        self.inner = torch.nn.Parameter(
            control_points[:, 1:-1, :].clone().detach().requires_grad_(True)
        )

        self.knots = tcf.uniform_augmented_knots(
            n_control_points,
            degree,
        )

    @property
    def control_points(self):
        return torch.cat(
            [
                self.q0[:, None, :],
                self.inner,
                self.q1[:, None, :],
            ],
            dim=1,
        )

    def forward(self, T):
        sample_points = torch.linspace(-1, 1, T, device=self.q0.device, dtype=self.q0.dtype)
        spline_args = sample_points[:, None].expand(T, self.q0.shape[0])
        c_pts = self.control_points.to(self.q0.device)

        curve = tcf.bspline_curves(
            spline_args,
            c_pts,
            knots=self.knots.to(self.q0.device),
            degree=self.degree,
        )

        return rearrange(curve, "t b d -> b t d")


class GeodesicSolverSpline(GeodesicDistanceSolver):
    """
    Solver for computing geodesic distances using B-spline parameterization of the geodesic curve.
    The control points of the B-spline are optimized to minimize the energy of the curve according to the given cometric.

    Parameters
    ----------
    cometric : CoMetric
        The cometric to use for computing the energy of the curve.
    T : int
        The number of time steps to sample the curve.
    n_control_points : int
        The number of control points for the B-spline.
    degree : int
        The degree of the B-spline.
    lr : float, optional
        The learning rate for the optimizer. Default is 1e-2.
    n_iter : int, optional
        The number of iterations for the optimizer. Default is 1000.
    """

    def __init__(
        self,
        cometric: CoMetric,
        T: int = 100,
        n_control_points: int = 20,
        degree: int = 3,
        lr=1e-2,
        n_iter=1000,
    ):
        super().__init__()
        self.T = T
        self.cometric = cometric
        self.n_control_points = n_control_points
        self.degree = degree
        self.lr = lr
        self.n_iter = n_iter

        self.loss_list: list[float] = []

    def optimize_spline(self, spline: GeodesicSpline) -> None:
        """
        Optimizes the control points of the given spline to minimize the energy of the curve according to the cometric.

        Parameters
        ----------
        spline : GeodesicSpline
            The spline to optimize.
        """
        spline.train()
        loss_list = []
        optimizer = torch.optim.Adam(spline.parameters(), lr=self.lr)
        pbar = tqdm(range(self.n_iter), desc="Optimizing spline", leave=False)
        for i in pbar:
            optimizer.zero_grad()
            curves_points = spline(self.T)
            energy = self.compute_energy(curves_points)
            energy.sum().backward()
            optimizer.step()
            loss_list.append(energy.sum().item())
            pbar.set_postfix({"loss": energy.sum().item()})
        self.loss_list = loss_list

    def get_trajectories(self, q0, q1, init_control_points=None) -> torch.Tensor:
        """
        Computes the geodesic trajectory between two points q0 and q1 using B-spline parameterization.
        If init_control_points is provided, it will be used as the initial control points for the spline optimization.

        Parameters
        ----------
        q0 : torch.Tensor (B, D)
            The starting point of the geodesic.
        q1 : torch.Tensor (B, D)
            The ending point of the geodesic.
        init_control_points : torch.Tensor (B, n_control_points, D), optional
            The initial control points for the B-spline. Must include q0 and q1 as the first and last control points, respectively.
            If None, they will be initialized as a linear interpolation between q0 and q1.

        Returns
        -------
        torch.Tensor (B, T, D)
            The computed geodesic trajectory between q0 and q1.
        """
        geodesic_spline = GeodesicSpline(
            q0,
            q1,
            self.n_control_points,
            self.degree,
            control_points=init_control_points,
        )
        self.optimize_spline(geodesic_spline)
        curves_points = geodesic_spline(self.T)
        return curves_points

    def forward(self, q0, q1, init_control_points=None):
        """
        Computes the geodesic distance between two points q0 and q1 using B-spline parameterization.
        If init_control_points is provided, it will be used as the initial control points for the spline optimization.
        If None, they will be initialized as a linear interpolation between q0 and q1.

        Parameters
        ----------
        q0 : torch.Tensor (B, D)
            The starting point of the geodesic.
        q1 : torch.Tensor (B, D)
            The ending point of the geodesic.
        init_control_points : torch.Tensor (B, n_control_points, D), optional
            The initial control points for the B-spline. Must include q0 and q1 as the first and last control points, respectively.
            If None, they will be initialized as a linear interpolation between q0 and q1.

        Returns
        -------
        torch.Tensor (B,)
            The computed geodesic distance between q0 and q1.
        """
        geodesic_spline = GeodesicSpline(
            q0,
            q1,
            self.n_control_points,
            self.degree,
            control_points=init_control_points,
        )
        self.optimize_spline(geodesic_spline)
        curves_points = geodesic_spline(self.T)
        dst = self.compute_distance(curves_points)
        return dst


class GeodesicSolverSplineFinsler(GeodesicSolverSpline):
    """
    Solver for computing geodesic distances using B-spline parameterization of the geodesic curve.
    The control points of the B-spline are optimized to minimize the energy of the curve according to the given Finsler metric.
    """

    def __init__(
        self,
        finsler: FinslerMetric,
        T: int = 100,
        n_control_points: int = 20,
        degree: int = 3,
        lr=1e-2,
        n_iter=1000,
    ):
        super().__init__(
            cometric=finsler,
            T=T,
            n_control_points=n_control_points,
            degree=degree,
            lr=lr,
            n_iter=n_iter,
        )
        self.finsler = finsler

    def compute_energy(
        self, curves_points: torch.Tensor, use_midpoints: bool = True
    ) -> torch.Tensor:
        """
        Computes the energy of the given curves according to the Finsler metric.

        Parameters
        ----------
        curves_points : torch.Tensor (B, T, D)
            The points along the curves.
        use_midpoints : bool, optional
            Whether to use midpoints for the energy computation. Default is True.

        Returns
        -------
        torch.Tensor (B,)
            The computed energy for each curve.
        """
        dx = torch.diff(curves_points, dim=1)  # (B, T-1, D)
        if use_midpoints:
            pts = 0.5 * (curves_points[:, :-1, :] + curves_points[:, 1:, :])  # (B, T-1, D)
        else:
            pts = curves_points[:, :-1, :]  # (B, T-1, D)
        energy = [self.finsler.forward(pts_, dx_).pow(2) for pts_, dx_ in zip(pts, dx)]
        energy = torch.stack(energy, dim=0)  # (B,T-1)
        return energy.sum(dim=1)  # (B,)

    def compute_distance(
        self, curves_points: torch.Tensor, use_midpoints: bool = True
    ) -> torch.Tensor:
        """
        Computes the geodesic distance of the given curves according to the Finsler metric.

        Parameters
        ----------
        curves_points : torch.Tensor (B, T, D)
            The points along the curves.
        use_midpoints : bool, optional
            Whether to use midpoints for the distance computation. Default is True.

        Returns
        -------
        torch.Tensor (B,)
            The computed geodesic distance for each curve.
        """
        dx = torch.diff(curves_points, dim=1)  # (B, T-1, D)
        if use_midpoints:
            pts = 0.5 * (curves_points[:, :-1, :] + curves_points[:, 1:, :])  # (B, T-1, D)
        else:
            pts = curves_points[:, :-1, :]  # (B, T-1, D)
        distance = [self.finsler.forward(pts_, dx_) for pts_, dx_ in zip(pts, dx)]
        distance = torch.stack(distance, dim=0)  # (B,T-1)
        return distance.relu().sum(dim=1)  # (B,)


class SolverGraphSpline(GeodesicDistanceSolver):
    """
    Chained solver. First the initial trajectory are computed using
    a graph based approach. Then they are refined using the Spline solver.
    """

    def __init__(
        self,
        cometric: CoMetric,
        data: torch.Tensor,
        n_neighbors: int,
        batch_size: int = 64,
        T: int = 100,
        n_control_points: int = 20,
        degree: int = 3,
        lr=1e-2,
        n_iter=1000,
        max_data_count: None | int = None,
        smooth_curve: bool = True,
    ):
        super().__init__(cometric=cometric)
        self.graph_solver = SolverGraph(
            cometric=cometric,
            data=data,
            n_neighbors=n_neighbors,
            T=T,
            batch_size=batch_size,
            max_data_count=max_data_count,
            smooth_curve=smooth_curve,
        )
        self.spline_solver = GeodesicSolverSpline(
            cometric=cometric,
            T=T,
            n_control_points=n_control_points,
            degree=degree,
            lr=lr,
            n_iter=n_iter,
        )

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Parameters:
        ----------
        q0 : Tensor (b,d)
            The starting points of the geodesic.
        q1 : Tensor (b,d)
            The ending points of the geodesic.

        Returns:
        -------
        trajectories : Tensor (b,T,d)
            The geodesic trajectories between q0 and q1 (both included).
        """
        pts_on_traj_graph = self.graph_solver.get_trajectories(q0, q1, connect_euclidean=True)
        pts_on_traj_spline = self.spline_solver.get_trajectories(
            q0, q1, init_control_points=pts_on_traj_graph
        )

        final_traj = torch.zeros_like(pts_on_traj_spline)
        # Return the trajectory with the smallest distance
        # This prevent the case where Spline didn't converge
        B = q0.shape[0]
        for b in range(B):
            dst_graph = self.compute_distance(pts_on_traj_graph[b].unsqueeze(0))
            dst_spline = self.compute_distance(pts_on_traj_spline[b].unsqueeze(0))
            if dst_graph < dst_spline:
                final_traj[b] = pts_on_traj_graph[b]
            else:
                final_traj[b] = pts_on_traj_spline[b]
        return final_traj


class SolverGraphSplineFinsler(torch.nn.Module):
    """
    Chained solver. First the initial trajectory are computed using
    a graph based approach. Then they are refined using the Spline solver.
    """

    def __init__(
        self,
        finsler: FinslerMetric,
        data: torch.Tensor,
        n_neighbors: int,
        batch_size: int = 64,
        T: int = 100,
        n_control_points: int = 20,
        degree: int = 3,
        lr=1e-2,
        n_iter=1000,
        max_data_count: None | int = None,
        smooth_curve: bool = True,
    ):
        super().__init__()
        self.graph_solver = SolverGraphFinsler(
            finsler_metric=finsler,
            data=data,
            n_neighbors=n_neighbors,
            T=T,
            batch_size=batch_size,
            max_data_count=max_data_count,
            smooth_curve=smooth_curve,
        )
        self.spline_solver = GeodesicSolverSplineFinsler(
            finsler=finsler,
            T=T,
            n_control_points=n_control_points,
            degree=degree,
            lr=lr,
            n_iter=n_iter,
        )
        self.finsler = finsler

    def compute_distance(
        self, curves_points: torch.Tensor, use_midpoints: bool = True
    ) -> torch.Tensor:
        """
        Computes the geodesic distance of the given curves according to the Finsler metric.

        Parameters
        ----------
        curves_points : torch.Tensor (B, T, D)
            The points along the curves.
        use_midpoints : bool, optional
            Whether to use midpoints for the distance computation. Default is True.

        Returns
        -------
        torch.Tensor (B,)
            The computed geodesic distance for each curve.
        """
        dx = torch.diff(curves_points, dim=1)  # (B, T-1, D)
        if use_midpoints:
            pts = 0.5 * (curves_points[:, :-1, :] + curves_points[:, 1:, :])  # (B, T-1, D)
        else:
            pts = curves_points[:, :-1, :]  # (B, T-1, D)
        # distance = self.finsler.forward(pts, dx).relu().sum(dim=1)  # (B,)
        distance = [self.finsler.forward(pts_, dx_) for pts_, dx_ in zip(pts, dx)]
        distance = torch.stack(distance, dim=0)  # (B,T-1)
        distance = distance.relu().sum(dim=1)  # (B,)
        return distance

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Parameters:
        ----------
        q0 : Tensor (b,d)
            The starting points of the geodesic.
        q1 : Tensor (b,d)
            The ending points of the geodesic.

        Returns:
        -------
        trajectories : Tensor (b,T,d)
            The geodesic trajectories between q0 and q1 (both included).
        """
        pts_on_traj_graph = self.graph_solver.get_trajectories(q0, q1, connect_euclidean=True)
        pts_on_traj_spline = self.spline_solver.get_trajectories(
            q0, q1, init_control_points=pts_on_traj_graph
        )

        final_traj = torch.zeros_like(pts_on_traj_spline)
        # Return the trajectory with the smallest distance
        # This prevent the case where Spline didn't converge
        B = q0.shape[0]
        for b in range(B):
            dst_graph = self.compute_distance(pts_on_traj_graph[b].unsqueeze(0))
            dst_spline = self.compute_distance(pts_on_traj_spline[b].unsqueeze(0))
            if dst_graph < dst_spline:
                final_traj[b] = pts_on_traj_graph[b]
            else:
                final_traj[b] = pts_on_traj_spline[b]
        return final_traj
