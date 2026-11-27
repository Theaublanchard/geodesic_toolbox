"""
Cascaded / chained geodesic solvers.
"""

import numpy as np
import torch
from torch import Tensor

from ..cometric import CoMetric
from ..finsler import FinslerMetric
from .base import GeodesicDistanceSolver
from .bvp import BVP_ode
from .georce import GEORCE, GEORCEFinsler
from .graph import SolverGraph, SolverGraphFinsler
from .spline import GeodesicSolverSpline, GeodesicSolverSplineFinsler


class SolverGraphBVP(GeodesicDistanceSolver):
    """
    Chained solver. First the initial trajectory is computed using a graph based
    approach, then it is refined by solving the boundary-value ODE between the
    two endpoints.
    """

    def __init__(
        self, cometric: CoMetric, data: Tensor, n_neighbors: int, T: int, dim: int
    ) -> None:
        super().__init__(cometric)
        self.solver_init = SolverGraph(cometric, data, n_neighbors, T=T)
        self.T = self.solver_init.T
        self.dim = dim
        self.solver = BVP_ode(cometric, self.T, dim)

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
        """
        Compute the geodesic trajectories between two points.
        First find a suitable trajectory in the graph and then solve the BVP between the two points.

        Parameters:
        -----------
        q0 : Tensor (b,d)
            The starting points
        q1 : Tensor (b,d)
            The ending points

        Returns:
        --------
        pts_on_traj : Tensor (b,T,d)
            The points on the trajectory
        """
        pts_on_traj_knn = self.solver_init.get_trajectories(q0, q1)
        # pts_on_traj = self.solver.get_trajectories(q0, q1, init_traj=pts_on_traj_knn)

        # Solve the BVP between the two points, if failed return the graph based trajectory
        traj_q = []
        t = np.linspace(0, 1, self.T)
        for b in range(q0.shape[0]):
            init_traj_b = pts_on_traj_knn[b]
            state = self.solver.solve_equation(q0[b], q1[b], init_traj_b)
            if state.status != 0:
                traj = pts_on_traj_knn[b]
            else:
                traj = state.sol(t)[: self.dim].T
            traj_q.append(torch.from_numpy(traj))
        pts_on_traj = torch.stack(traj_q, dim=0)
        return pts_on_traj


class SolverGraphGEORCE(GeodesicDistanceSolver):
    """
    Chained solver. First the initial trajectory are computed using
    a graph based approach. Then they are refined using the GEORCE solver.
    """

    def __init__(
        self,
        cometric: CoMetric,
        data: torch.Tensor,
        n_neighbors: int,
        batch_size: int = 64,
        T: int = 100,
        max_iter=20,  # low value bc good initialization from graph
        tol=1e-10,
        rho=0.5,
        c=0.9,
        alpha_0: float = 1.0,
        pbar_georce: bool = False,
        max_data_count: None | int = None,
        smooth_curve: bool = True,
    ):
        super().__init__(cometric=cometric)
        # Public API: this chained solver returns exactly T points.
        # GEORCE now also interprets T as the number of points.
        self.graph_solver = SolverGraph(
            cometric=cometric,
            data=data,
            n_neighbors=n_neighbors,
            T=T,
            batch_size=batch_size,
            max_data_count=max_data_count,
            smooth_curve=smooth_curve,
        )
        self.georce_solver = GEORCE(
            cometric=cometric,
            T=T,
            max_iter=max_iter,
            tol=tol,
            rho=rho,
            c=c,
            alpha_0=alpha_0,
            pbar=pbar_georce,
        )

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Params:
        q0 : Tensor (b,d), start points.
        q1 : Tensor (b,d), end points

        Output:
        traj_q : Tensor (b,n_pts,dim), points on the trajectory
        """
        pts_on_traj_graph = self.graph_solver.get_trajectories(q0, q1, connect_euclidean=True)
        pts_on_traj_georce = self.georce_solver.get_trajectories(
            q0, q1, pts_on_traj_graph[:, 1:-1, :].clone().detach()
        )
        final_traj = torch.zeros_like(pts_on_traj_georce)
        # Return the trajectory with the smallest distance
        # This prevent the case where GEORCE didn't converge
        B = q0.shape[0]
        for b in range(B):
            dst_graph = self.compute_distance(pts_on_traj_graph[b].unsqueeze(0))
            dst_georce = self.compute_distance(pts_on_traj_georce[b].unsqueeze(0))
            if dst_graph < dst_georce:
                final_traj[b] = pts_on_traj_graph[b]
            else:
                final_traj[b] = pts_on_traj_georce[b]
        return final_traj


class SolverGraphGEORCEFinsler(GEORCEFinsler):
    """
    Chained solver. First the initial trajectory are computed using
    a graph based approach. Then they are refined using the GEORCE solver.
    """

    def __init__(
        self,
        finsler: FinslerMetric,
        data: torch.Tensor,
        n_neighbors: int,
        batch_size: int = 64,
        T: int = 100,
        max_iter=30,  # low value bc good initialization from graph
        tol=1e-10,
        rho=0.5,
        c=0.9,
        alpha_0: float = 1.0,
        pbar_georce: bool = False,
        max_data_count: None | int = None,
        smooth_curve: bool = True,
    ):
        super().__init__(
            finsler=finsler,
            T=T,
            max_iter=max_iter,
            tol=tol,
            rho=rho,
            c=c,
            alpha_0=alpha_0,
            pbar=pbar_georce,
        )
        self.graph_solver = SolverGraphFinsler(
            finsler_metric=finsler,
            data=data,
            n_neighbors=n_neighbors,
            T=T,
            batch_size=batch_size,
            max_data_count=max_data_count,
            smooth_curve=smooth_curve,
        )

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Params:
        q0 : Tensor (b,d), start points.
        q1 : Tensor (b,d), end points

        Output:
        traj_q : Tensor (b,n_pts,dim), points on the trajectory
        """
        pts_on_traj_graph = self.graph_solver.get_trajectories(q0, q1, connect_euclidean=True)
        pts_on_traj_georce = super().get_trajectories(
            q0, q1, pts_on_traj_graph[:, 1:-1, :].detach()
        )
        final_traj = torch.zeros_like(pts_on_traj_georce)
        # Return the trajectory with the smallest distance
        # This prevent the case where GEORCE didn't converge
        B = q0.shape[0]
        for b in range(B):
            traj_graph = pts_on_traj_graph[b, :, :]
            traj_georce = pts_on_traj_georce[b]
            dst_graph_ = self.dst_func(q0[b], q1[b], traj_graph[1:-1])
            dst_georce_ = self.dst_func(q0[b], q1[b], traj_georce[1:-1])
            if dst_graph_ < dst_georce_:
                final_traj[b] = traj_graph.detach().clone()
            else:
                final_traj[b] = pts_on_traj_georce[b].detach().clone()
        return final_traj


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

        Parameters:
        ----------
        curves_points : torch.Tensor (B, T, D)
            The points along the curves.
        use_midpoints : bool, optional
            Whether to use midpoints for the distance computation. Default is True.

        Returns:
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
