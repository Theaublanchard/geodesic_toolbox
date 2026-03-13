import numpy as np
import torch
from torch import Tensor
from math import ceil
from einops import rearrange
from collections.abc import Callable
from scipy.integrate import solve_bvp

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import CubicSpline
from networkx import Graph, DiGraph, is_connected, is_strongly_connected, is_weakly_connected
import networkx as nx

from .cometric import CoMetric, IdentityCoMetric, FinslerMetric
from .utils import (
    magnification_factor,
    # hamiltonian,
    cosine_time_scaling_schedule,
    scale_lr_magnification,
    vec,
    batched_kro,
)

from tqdm import tqdm
from torchdiffeq import odeint


def _get_padded_neighbor_indices(
    edge_idx: Tensor,
    batch_size: int,
    target_width: int | None = None,
    pad_value: int = 0,
) -> Tensor:
    """
    Pad a nonzero index list so each row has the same number of neighbors.

    Example with target_width=4 and pad_value=0:
    input edge_idx:
    tensor([[0, 1],
            [0, 12],
            [1, 4],
            [1, 5],
            [1, 65]])

    output padded_edge_idx:
    tensor([[0, 1],
            [0, 12],
            [0, 0],
            [0, 0],
            [1, 4],
            [1, 5],
            [1, 65],
            [1, 0]])

    Parameters:
    edge_idx : Tensor (n_edges, 2)
        Output of nonzero(as_tuple=False) on a batch of adjacency rows.
        Column 0 is the row id inside the batch, column 1 is the neighbor id.
    batch_size : int
        Number of rows in the adjacency batch.
    target_width : int, optional
        If provided, each row is padded to at least target_width entries.
        If some row has more neighbors than target_width, width is increased automatically.
    pad_value : int, optional
        Value used for padded neighbor ids.

    Returns:
    padded_edge_idx : Tensor (batch_size * width, 2)
        Padded pair representation [row_in_batch, neighbor_id].
        It can be reshaped back to (batch_size, width) by taking column 1.
    """
    if edge_idx.numel() == 0:
        counts = torch.zeros(batch_size, dtype=torch.long, device=edge_idx.device)
    else:
        counts = torch.bincount(edge_idx[:, 0], minlength=batch_size)

    width = int(counts.max().item()) if counts.numel() > 0 else 0
    if target_width is not None:
        width = max(width, target_width)

    padded_edge_idx = torch.full(
        (batch_size * width, 2),
        pad_value,
        device=edge_idx.device,
        dtype=torch.long,
    )

    row_template = (
        torch.arange(batch_size, device=edge_idx.device, dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, width)
        .reshape(-1)
    )
    padded_edge_idx[:, 0] = row_template

    if edge_idx.numel() == 0 or width == 0:
        return padded_edge_idx

    row_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)[:-1]])
    slot_idx = torch.arange(edge_idx.shape[0], device=edge_idx.device, dtype=torch.long)
    slot_idx = slot_idx - row_offsets.repeat_interleave(counts)

    flat_pos = edge_idx[:, 0] * width + slot_idx
    padded_edge_idx[flat_pos, 1] = edge_idx[:, 1]
    return padded_edge_idx


class GeodesicDistanceSolver(torch.nn.Module):
    """Base class for geodesic distance solvers."""

    def __init__(self, cometric: CoMetric = IdentityCoMetric()):
        super().__init__()
        self.cometric = cometric

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Params:
        q0 : Tensor (b,d), start points.
        q1 : Tensor (b,d), end points

        Output:
        traj_q : Tensor (b,n_pts,dim), points on the trajectory
        """
        pass

    def compute_distance(self, traj_q: Tensor) -> Tensor:
        """Given the trajectory, computes its length according to the metric

        Params :
        traj_q : Tensor (b,n_pts,d), points on the trajectories

        Output :
        distances : Tensor, (b,), distances of the trajectories
        """
        traj_front = traj_q[:, 1:, :]
        traj_back = traj_q[:, :-1, :]
        segments = traj_front - traj_back
        midpoints = (traj_front + traj_back) / 2

        # # This version uses way too much memory
        # segments = rearrange(segments, "b n d -> (b n) d")
        # midpoints = rearrange(midpoints, "b n d -> (b n) d")
        # distances = self.cometric.inverse_forward(midpoints, segments)  # (b*n, d)
        # distances = torch.einsum("B d, B d -> B", segments, distances)  # (b*n,)
        # distances = rearrange(distances, "(b n) -> b n", b=traj_q.shape[0])

        # # This version is more memory efficient but slower
        distances = [self.cometric.metric(m, seg) for m, seg in zip(midpoints, segments)]
        distances = torch.stack(distances)

        # Add a ReLU to avoid negative distances due to numerical errors
        distances = distances.relu().sqrt().sum(dim=1)
        return distances

    def forward(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given two batch of points q0 and q1, compute the geodesic distance between them.

        Parameters:
        -----------
        q0 : Tensor (B,d)
            The starting points
        q1 : Tensor (B,d)
            The ending points

        Returns:
        --------
        dst : Tensor (B,)
            The estimated geodesic distance between the points
        """
        traj_q = self.get_trajectories(q0, q1)
        dst = self.compute_distance(traj_q)
        return dst


class ShootingSolver(GeodesicDistanceSolver):
    """Compute the geodesic distance by shooting and integrating the hamiltonian equations.
    The integration method can be either Euler or Leapfrog.

    Params:
    cometric : CoMetric, function that outputs the inverse metric tensor as a (b,d,d) matrix
    lr : float, learning rate for the initial momentum optimisation
    n_step : int, number of optimisation step done
    dt : float, final integration step of the hamiltonian. It is higly recommended to choose a value such that dt* ceil(1/dt) = 1 for numerical stability
    method : str, type of integrator
    convergence_threshold : float, if not None, optimise until the distance between the endpoint and the target is below this threshold. Else, optimise for n_step iterations.
    time_scaling_schedule : Callable[float]->float, function from [0,1] to R+ that scales the time step of the integrator. It should be decreasing and such that time_scaling_schedule(1) = 1.
        It is used to decrease the time step as the optimisation converges. Only used when convergence_threshold is None.
    scale_lr : bool, whether to scale the learning rate with the magnification factor or not. This is to avoid shooting being stuck in high curvature region.

    Output:
    distances : Tensor (b,) geodesic distances
    """

    def __init__(
        self,
        cometric: CoMetric = IdentityCoMetric(),
        lr: float = 0.1,
        n_step: int = 100,
        dt: float = 0.01,
        method: str = "euler",
        convergence_threshold: float = None,
        time_scaling_schedule: Callable[[float], float] = cosine_time_scaling_schedule,
        scale_lr: bool = True,
        verbose: bool = False,
    ) -> None:

        super().__init__(cometric=cometric)
        self.dt = dt
        self.final_dt = dt
        self.n_pts = ceil(1 / self.dt)
        self.lr = lr
        self.n_step = n_step
        self.time_scaling_schedule = time_scaling_schedule
        self.scale_lr = scale_lr
        self.verbose = verbose

        self.last_loss = -1  # For tracing and debug purposes

        if method == "euler":
            self.integration_step = self.euler_step
        elif method == "leapfrog":
            self.integration_step = self.leapfrog_step
        else:
            raise ValueError(
                f"Unknown integration method got {method} ; expect one of [euler,leapfrog]"
            )

        if convergence_threshold is not None:
            self.optim_method = self._optim_until_convergence
            self.convergence_threshold = convergence_threshold
            self.n_step_done = 0
            self.n_step_max = n_step * 5
        else:
            self.optim_method = self._optim_until_step
            self.convergence_threshold = 1e-3

        self.dH_ = torch.func.jacrev(lambda p, q: self.H(p, q).sum(), argnums=(0, 1))

    def get_dp_dq(self, p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the partial derivatives of the Hamiltonian w.r.t. p and q.

        Params:
        p : Tensor, (b,d) momentum
        q : Tensor, (b,d) position

        Output:
        dH_dp : Tensor, (b,d) partial derivative of the Hamiltonian w.r.t. p
        dH_dq : Tensor, (b,d) partial derivative of the Hamiltonian w.r.t. q
        """
        dH_dp, dH_dq = self.dH_(p, q)
        return dH_dp, dH_dq

    def H(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Computes the Hamiltonian at point q for momentum p.
        H(p,q) = p^T G_inv(q) p

        Params:
        p : Tensor, (b,d) momentum
        q : Tensor, (b,d) position

        Output:
        res : Tensor, (b,) hamiltonian
        """
        return self.cometric.cometric(q, p)

    def euler_step(self, H: Callable, p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
        """
        Euler integrator step.

        Params:
        H : Callable, hamiltonian function
        p : Tensor, (b,d) momentum
        q : Tensor, (b,d) position

        Output:
        p : Tensor, (b,d) new momentum
        q : Tensor, (b,d) new position
        """
        with torch.enable_grad():
            dp, dq = self.get_dp_dq(p, q)
        p = p - self.dt * dq
        q = q + self.dt * dp

        return p, q

    def leapfrog_step(self, H: Callable, p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
        """
        Leapfrog integrator step.
        I don't know if this version is sympletic because here the hamiltonian is not separable.

        Params:
        H : Callable, hamiltonian function
        p : Tensor, (b,d) momentum
        q : Tensor, (b,d) position

        Output:
        p : Tensor, (b,d) new momentum
        q : Tensor, (b,d) new position
        """

        # Half step in momentum
        with torch.enable_grad():
            dq = self.get_dp_dq(p, q)[1]
        p_half = p - 0.5 * self.dt * dq

        # Full step in position
        with torch.enable_grad():
            dp_half = self.get_dp_dq(p_half, q)[0]
        q = q + self.dt * dp_half

        # Another half step in momentum
        with torch.enable_grad():
            dq = self.get_dp_dq(p_half, q)[1]
        p = p_half - 0.5 * self.dt * dq

        return p, q

    def shooting(
        self,
        p0: Tensor,
        q0: Tensor,
        return_traj: bool = False,
        return_p: bool = False,
    ) -> Tensor | tuple[list[Tensor], list[Tensor]] | list[Tensor]:
        """
        Integrate the hamiltonian equation given an initial velocity.

        Params:
        p0 : Tensor, (b,d) initial velocity
        q0 : Tensor, (b,d) initial position
        return_traj : bool, whether to return the trajectory or not
        return_p : bool, whether to return the final momentum or not

        Output:
        q : Tensor, (b,d) final point
        traj_q : list[Tensor], (n_pts,b,d) list of points on the trajectory
        traj_p : list[Tensor], (n_pts,b,d) list of momenta on the trajectory

        """
        q, p = q0.clone(), p0.clone()
        if return_traj:
            traj_q = [q]
        if return_p:
            traj_p = [p]
        # H = lambda p, q: hamiltonian(self.cometric(q), p)

        for _ in range(self.n_pts):
            p, q = self.integration_step(self.H, p, q)

            if return_traj:
                traj_q.append(q)
            if return_p:
                traj_p.append(p)

        if return_traj and return_p:
            return traj_q, traj_p
        elif return_traj:
            return traj_q
        elif return_p:
            return traj_q, traj_p
        else:
            return q

    def optimize_initial_velocity(self, q0: Tensor, q1: Tensor) -> Tensor:
        """
        Optimise the initial velocity to get the geodesic path between q0 and q1.

        Params:
        q0 : Tensor, (b,d) start point
        q1 : Tensor, (b,d) end point

        Output:
        p0 : Tensor, (b,d) initial velocity
        """
        q0_ = q0.detach().clone().requires_grad_()
        q1_ = q1.detach().clone().requires_grad_()
        # Initial guess corresponding to the analytical solution for euclidean space
        p0 = ((q1_ - q0_) / 2).detach().requires_grad_()
        # Initial guess corresponding to the analytical solution for constant metric
        # p0 = torch.zeros_like(q0, requires_grad=True, dtype=q0.dtype, device=q0.device)
        # p0.data = 0.5 * self.cometric.inverse_forward(q0_, q1_ - q0_).detach()

        # scale lr with magnification factor
        if self.scale_lr:
            mf_0 = magnification_factor(self.cometric, q0_).max()
            mf_1 = magnification_factor(self.cometric, q1_).max()
            scale = scale_lr_magnification(max(mf_0, mf_1), self.lr)
            new_lr = self.lr * scale
            optim = torch.optim.Adam([p0], lr=new_lr)
        else:
            optim = torch.optim.Adam([p0], lr=self.lr)

        # @TODO : compute the backward analytically
        p0 = self.optim_method(q0_, q1_, p0, optim)

        return p0.detach().requires_grad_()

    def _update_dt(self, curr_iteration: int) -> None:
        self.dt = self.final_dt * self.time_scaling_schedule(curr_iteration / self.n_step)
        self.n_pts = ceil(1 / self.dt)

    def _optim_until_step(
        self, q0_: Tensor, q1_: Tensor, p0: Tensor, optim: torch.optim.Optimizer
    ) -> Tensor:
        for it in range(self.n_step):
            self._update_dt(it)

            optim.zero_grad()
            q = self.shooting(p0, q0_)
            loss = torch.nn.functional.mse_loss(q, q1_)
            loss.backward()
            self.last_loss = torch.linalg.vector_norm(q - q1_, dim=1).detach().cpu().numpy()
            optim.step()

        if self.last_loss.mean() > self.convergence_threshold and self.verbose:
            print(
                f"Optimisation did not converge after {self.n_step} steps. MMSE shooting loss : {self.last_loss.mean()}\n last_loss = {self.last_loss}"
            )

        return p0

    def _optim_until_convergence(
        self, q0_: Tensor, q1_: Tensor, p0: Tensor, optim: torch.optim.Optimizer
    ) -> Tensor:
        loss = 100
        n_step_min = 5
        self.n_step_done = 0
        while (
            loss > self.convergence_threshold and self.n_step_done < self.n_step_max
        ) or self.n_step_done < n_step_min:
            optim.zero_grad()
            q = self.shooting(p0, q0_)
            loss = torch.nn.functional.mse_loss(q, q1_)
            loss.backward()
            self.last_loss = torch.linalg.vector_norm(q - q1_, dim=1).detach().cpu().numpy()
            optim.step()
            self.n_step_done += 1

        if self.n_step_done >= self.n_step_max and self.verbose:
            print(
                f"Optimisation did not converge after {self.n_step_max} steps. Last loss : {self.last_loss}"
            )

        return p0

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Params:
        q0 : Tensor (b,d), start points.
        q1 : Tensor (b,d), end points

        Output:
        traj_q : Tensor (b,n_pts,dim), points on the trajectory
        """
        with torch.enable_grad():
            q0 = q0.clone().requires_grad_()
            q1 = q1.clone().requires_grad_()
            initial_velocity = self.optimize_initial_velocity(q0, q1)
        traj_q = self.shooting(initial_velocity, q0, return_traj=True)

        # Detach the trajectory to prevent incorrect backpropagation and replace by the start and end points
        # This way the metric is only used in the distance computation
        # Also add the start and end points allows to propagate the gradients to them
        traj_q = [q.detach() for q in traj_q]
        traj_q[0] = q0
        traj_q[-1] = q1  # Not sure if good idea. If optim failed it can be bad

        traj_q = torch.stack(traj_q, dim=1)
        return traj_q

    def exp(self, p0: Tensor, q0: Tensor) -> Tensor:
        """
        Compute the exponential map of the metric.

        Params:
        p0 : Tensor, (b,d) initial velocity
        q0 : Tensor, (b,d) initial position

        Output:
        q : Tensor, (b,d) final point
        """
        return self.shooting(p0, q0)

    def log(self, q0: Tensor, q1: Tensor) -> Tensor:
        """
        Compute the log map of the metric.

        Params:
        q0 : Tensor, (b,d) start point
        q1 : Tensor, (b,d) end point

        Output:
        p0 : Tensor, (b,d) initial velocity
        """
        return self.optimize_initial_velocity(q0, q1)

    def angle(self, q0: Tensor, q1: Tensor, origin: Tensor = None) -> Tensor:
        """
        Compute the angle between two points according to the metric. The angle is here defined as
        the cosine similarity between the log of the two points relative to the origin point.
        Beware that the origin needs to be a point on the manifold that is connected to
        the two points.

        Params:
        q0 : Tensor, (b,d) start point
        q1 : Tensor, (b,d) end point
        origin : Tensor, (b,d) origin point. If None, the origin is set to zero.

        Output:
        angle : Tensor, (b,) angle between the two points
        """

        if origin is None:
            origin = torch.zeros_like(q0)

        p0 = self.log(origin, q0)
        p1 = self.log(origin, q1)

        angle = torch.nn.functional.cosine_similarity(p0, p1, dim=1)
        return angle

    def forward(self, q0: Tensor, q1: Tensor) -> Tensor:
        """
        Params :
        q0 : Tensor (b,d), start points. It should have requires_grad = True
        q1 : Tensor (b,d), end points

        Output :
        distances : Tensor (b,) geodesic distances
        """
        traj_q = self.get_trajectories(q0, q1)
        distances = self.compute_distance(traj_q)
        return distances


class BVP_wrapper(GeodesicDistanceSolver):
    """
    Wrapper class for scipy.integrate.solve_bvp to compute geodesic distances.
    Throughout the code, the position is always given by the fist dim elements of the state.

    Params:
    cometric : CoMetric
        cometric object
    T : int
        number of time steps
    dim : int
        dimension of the space
    verbose : int
        verbosity level of scipy.integrate.solve_bvp
    """

    def __init__(self, cometric: CoMetric, T: int = 100, dim: int = 2, verbose=0):
        super(BVP_wrapper, self).__init__(cometric)
        self.T = T
        self.dim = dim
        self.verbose = verbose

    def fun(self, t: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Computes the right hand side of whatever ODE we want to solve.

        Parameters
        ----------
        t : np.ndarray (m,)
            time steps
        state : np.ndarray (2*dim, m)
            state of the system

        Returns
        -------
        f_t : np.ndarray (2*dim, m)
            right-hand side of the ODE
        """
        pass

    def bc_(
        self,
        state_init: np.ndarray,
        state_final: np.ndarray,
        start_pts: np.ndarray,
        end_pts: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the boundary conditions of the system

        Parameters
        ----------
        state_init : np.ndarray (2*dim,)
            initial state.
        state_final : np.ndarray (2*dim,)
            final state
        start_pts : np.ndarray (dim,)
            initial point
        end_pts : np.ndarray (dim,)
            final point

        Returns
        -------
        bc_cond : np.ndarray (2*dim,)
            boundary conditions
        """

        q_0 = state_init[: self.dim]  # initial position (dim,)
        q_1 = state_final[: self.dim]  # final position (dim,)

        bc_cond = np.zeros_like(state_init)  # (2*dim,)
        bc_cond[: self.dim] = q_0 - start_pts
        bc_cond[self.dim :] = q_1 - end_pts

        return bc_cond

    def bc_fun(self, start_pts: np.ndarray, end_pts: np.ndarray) -> Callable:
        """
        Wrapper around bc_ to make it compatible with scipy.integrate.solve_bvp

        Parameters
        ----------
        start_pts : np.ndarray (dim,)
            initial point
        end_pts : np.ndarray (dim,)
            final point

        Returns
        -------
        bc : Callable
            boundary conditions function
        """

        def bc(state_init, state_final):
            return self.bc_(state_init, state_final, start_pts, end_pts)

        return bc

    def solve_equation(self, start_pts: Tensor, end_pts: Tensor, init_traj=None):
        """
        Solves the equation between two points

        Parameters
        ----------
        start_pts : Tensor(dim,)
            initial point
        end_pts : Tensor(dim,)
            final point
        init_traj : Tensor(T,dim), optional
            initial guess for the trajectory position. If None, the initial guess is a linear interpolation between the two points.

        Returns
        -------
        state : BVPResult
            solution of the BVP
        """
        pass

    def get_trajectories(self, start_pts: Tensor, end_pts: Tensor, init_traj=None) -> Tensor:
        """
        Computes the geodesic trajectories between two points

        Parameters
        ----------
        start_pts : Tensor(b,dim)
            initial point
        end_pts : Tensor(b,dim)
            final point
        init_traj : Tensor(b,T,dim), optional
            initial guess for the trajectory. If None, the initial guess is a linear interpolation between the two points.

        Output:
        traj_q : Tensor(b,T,dim)
            points on the trajectories
        """
        traj_q = []
        t = np.linspace(0, 1, self.T)
        for b in range(start_pts.shape[0]):
            init_traj_b = init_traj[b] if init_traj is not None else None
            state = self.solve_equation(start_pts[b], end_pts[b], init_traj_b)
            if state.status != 0 and self.verbose > 0:
                print(f"Failed to solve BVP for batch {b}. Got\n{state}")
            traj = state.sol(t)[: self.dim].T
            traj_q.append(torch.from_numpy(traj))
        return torch.stack(traj_q, dim=0)


class BVP_shooting(BVP_wrapper):
    """
    BVP solver for Hamiltonian system.
    It is usually pretty slow and not very stable.
    Typically when the metric explodes around manifold boundaries the algorithm fails.
    For example on the pointcarre metric, it fails when the points are near the boundary of the sphere.

    Params:
    cometric : CoMetric
        cometric object
    T : int
        number of time steps
    dim : int
        dimension of the space
    """

    def __init__(self, cometric: CoMetric, T: int = 100, dim: int = 2, verbose=0):
        super().__init__(cometric=cometric, dim=dim, T=T, verbose=verbose)
        self.get_dp_dq = torch.func.grad(
            lambda p, q: self.compute_hamiltonian(p, q).sum(), argnums=(0, 1)
        )
        if cometric.is_diag:
            raise NotImplementedError(
                "BVP_shooting not implemented/tested for diagonal cometrics"
            )

    def compute_hamiltonian(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Computes the Hamiltonian at point q for momentum p.

        Params:
        p : Tensor, (b,d) momentum
        q : Tensor, (b,d) position

        Output:
        res : Tensor, (b,) hamiltonian
        """
        return self.cometric.cometric(q, p).float()

    def compute_derivative(
        self, q: np.ndarray, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the derivative of the Hamiltonian w.r.t. q and p

        Parameters
        ----------
        q : np.ndarray (m,dim)
            position
        p : np.ndarray (m,dim)
            momentum

        Returns
        -------
        dp : np.ndarray (m,dim)
            derivative of the Hamiltonian w.r.t. p
        dq : np.ndarray (m,dim)
            derivative of the Hamiltonian w.r.t. q
        """
        p = torch.from_numpy(p).requires_grad_().float()
        q = torch.from_numpy(q).requires_grad_().float()

        with torch.enable_grad():
            dp, dq = self.get_dp_dq(p, q)
        dp = dp.detach().numpy()
        dq = dq.detach().numpy()
        return dp, dq

    def fun(self, t: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Computes the right-hand side of the Hamiltonian system

        Parameters
        ----------
        t : np.ndarray (m,)
            time steps
        state : np.ndarray (2*dim, m)
            state of the system

        Returns
        -------
        f_t : np.ndarray (2*dim, m)
            right-hand side of the Hamiltonian system
        """
        q = state[: self.dim].T  # (dim, m)
        p = state[self.dim :].T  # (dim, m)
        dH_dp, dH_dq = self.compute_derivative(q, p)

        f_t = np.zeros_like(state)
        f_t[: self.dim] = dH_dp.T
        f_t[self.dim :] = dH_dq.T
        return f_t

    def solve_equation(self, start_pts: Tensor, end_pts: Tensor, init_traj: Tensor = None):
        """
        Solves the Hamiltonian system between two points

        Parameters
        ----------
        start_pts : Tensor(dim,)
            initial point
        end_pts : Tensor(dim,)
            final point
        init_traj : Tensor(T,dim), optional
            initial guess for the trajectory. If None, the initial guess is a linear interpolation between the two points.

        Returns
        -------
        state : BVPResult
            solution of the BVP
        """
        start_pts = start_pts.detach().numpy()
        end_pts = end_pts.detach().numpy()

        t = np.linspace(0, 1, self.T)

        state_init = np.zeros((2 * self.dim, self.T))
        if init_traj is not None:
            traj_front = init_traj[1:, :]
            traj_back = init_traj[:-1, :]
            init_p = traj_front - traj_back  # (T-1,dim)
            init_p = torch.cat([init_p, init_p[-1].unsqueeze(0)], dim=0)  # (T,dim)
            state_init[: self.dim, :] = init_traj.detach().numpy().T
            state_init[self.dim :, :] = init_p.detach().numpy().T

        else:
            # Init position with linear interpolation
            init_q = t * start_pts[:, None] + (1 - t) * end_pts[:, None]
            # And velocity with difference
            init_p = (end_pts - start_pts)[:, None]
            state_init[: self.dim, :] = init_q
            state_init[self.dim :, :] = init_p
        bc = self.bc_fun(start_pts, end_pts)
        state = solve_bvp(
            self.fun, bc, t, state_init, max_nodes=5 * self.T, verbose=self.verbose
        )
        return state


class BVP_ode(BVP_wrapper):
    """
    BVP solver for the geodesic equation.
    Albeit slower than the shooting method, it is more stable and can handle more complex metrics.
    At least it seems to work on the pointcarre metric.

    Parameters
    ----------
    cometric : CoMetric
        cometric object
    T : int
        number of time steps
    dim : int
        dimension of the space
    """

    def __init__(self, cometric=CoMetric, T=100, dim=2, verbose=0):
        super().__init__(cometric=cometric, dim=dim, T=T, verbose=verbose)

        if cometric.is_diag:
            raise NotImplementedError("BVP_ode not implemented/tested for diagonal cometrics")

        if hasattr(self.cometric, "jacobian"):
            self.get_dVecM = lambda gamma: self.cometric.jacobian(gamma.squeeze(2))
        else:
            self.get_dVecM = self.compute_dVecM()

    def compute_dVecM(self) -> Callable:
        """
        Compute the derivative of the flatten metric tensor
        """
        eval_VecM = lambda gamma: vec(self.cometric.metric_tensor(gamma.squeeze(2)))
        jac_ = torch.func.jacrev(eval_VecM)
        dVecM = lambda gamma: torch.einsum("b D B d i -> b D d i", jac_(gamma))
        return dVecM

    def geodesic_equation(self, gamma: Tensor, gamma_dot: Tensor) -> Tensor:
        """Compute acceleration term of the geodesic equation

        gamma : Tensor
            (m,d,1) position
        gamma_dot : Tensor
            (m,d,1) velocity

        Returns
        -------
        Tensor
            (m,d,1) acceleration
        """
        m = gamma.shape[0]
        id = torch.eye(self.dim).repeat(m, 1, 1)
        kro_gammadot_id = batched_kro(gamma_dot.mT, id)
        kro_gammadot = batched_kro(gamma_dot, gamma_dot)

        M_inv = self.cometric(gamma.squeeze(2))
        dVecM = self.get_dVecM(gamma).squeeze(-1)

        a = 2 * kro_gammadot_id @ dVecM @ gamma_dot
        b = dVecM.mT @ kro_gammadot
        gamma_dotdot = -0.5 * M_inv @ (a - b)

        return gamma_dotdot

    def state_to_ode(self, state: np.ndarray) -> tuple[Tensor, Tensor]:
        """
        Unpack the state into position and velocity

        Parameters
        ----------
        state : array
            (2*d,m) state of the system

        Returns
        -------
        gamma : Tensor
            (m,d,1) position
        gamma_dot : Tensor
            (m,d,1) velocity
        """
        gamma = torch.from_numpy(state[: self.dim]).T
        gamma_dot = torch.from_numpy(state[self.dim :]).T

        gamma = gamma.unsqueeze(-1).float()
        gamma_dot = gamma_dot.unsqueeze(-1).float()

        return gamma, gamma_dot

    def ode_to_state(self, gamma: Tensor, gamma_dot: Tensor) -> np.ndarray:
        """
        Pack the position and velocity into a state

        Parameters
        ----------
        gamma : Tensor (m,d,1) position
        gamma_dot : Tensor (m,d,1) velocity

        Returns
        -------
        np.ndarray
            (2*d,m) state of the system
        """

        gamma = gamma.squeeze(2).detach().numpy()
        gamma_dot = gamma_dot.squeeze(2).detach().numpy()
        state = np.concatenate([gamma, gamma_dot], axis=1).T

        return state

    def fun(self, t: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Compute the right-hand side of the geodesic equation

        Parameters
        ----------
        t : array (m,) evaluation time
        state : array (2*dim, m) state of the system at all time steps t

        Returns
        -------
        array (2*dim, m)
            right-hand side of the geodesic equation
        """
        gamma, gamma_dot = self.state_to_ode(state)
        gamma_dotdot = self.geodesic_equation(gamma, gamma_dot)
        state = self.ode_to_state(gamma_dot, gamma_dotdot)
        return state

    def solve_equation(self, start_pts: Tensor, end_pts: Tensor, init_traj=None):
        """
        Solves the geodesic equation between two points

        Parameters
        ----------
        start_pts : Tensor(dim,)
            initial point
        end_pts : Tensor(dim,)
            final point
        init_traj : Tensor(T,dim), optional
            initial guess for the trajectory. If None, the initial guess is a linear interpolation between the two points.

        Returns
        -------
        state : BVPResult
            solution of the geodesic equation
        """
        start_pts = start_pts.detach().numpy()
        end_pts = end_pts.detach().numpy()

        t = np.linspace(0, 1, self.T)

        state_init = np.zeros((2 * self.dim, self.T))
        if init_traj is not None:
            traj_front = init_traj[1:, :]
            traj_back = init_traj[:-1, :]
            init_p = traj_front - traj_back  # (T-1,dim)
            init_p = torch.cat([init_p, init_p[-1].unsqueeze(0)], dim=0)  # (T,dim)
            state_init[: self.dim, :] = init_traj.detach().numpy().T
            state_init[self.dim :, :] = init_p.detach().numpy().T
        else:
            # Init the trajectory with a linear interpolation
            init_q = t * start_pts[:, None] + (1 - t) * end_pts[:, None]
            # And the velocity with the difference
            init_q_dot = (end_pts - start_pts)[:, None] / 2
            state_init[: self.dim, :] = init_q
            state_init[self.dim :, :] = init_q_dot

        bc = self.bc_fun(start_pts, end_pts)
        state = solve_bvp(
            self.fun, bc, t, state_init, max_nodes=5 * self.T, verbose=self.verbose
        )
        return state


class SolverGraph(GeodesicDistanceSolver):
    """Computes the geodesic distances between points using a KNN-graph

    Parameters
    ----------
    cometric : nn.Module
        The cometric to use to compute the geodesic distances
    data : torch.Tensor (N,D)
        The data points to use to compute the graph
    n_neighbors : int
        The number of neighbors to use
    dt : float
        The time step to use for the linear interpolation
    batch_size : int
        The size of the batch to use for the computation of the graph
    max_data_count : int
        The maximum number of data points to use for the graph. If None, all data points
        are used. If the number of data points is larger than this value, the graph is
        computed on a random subset of the data points to save memory.
    smooth_curve : bool
        If True, smooth the curve with a mean kernel before feeding it to the spline
        when reparametrizing the trajectory.
    """

    def __init__(
        self,
        cometric: CoMetric,
        data: torch.Tensor,
        n_neighbors: int,
        dt: float = 0.01,
        batch_size: int = 64,
        max_data_count: int | None = None,
        smooth_curve: bool = True,
    ) -> None:
        super().__init__(cometric)

        if max_data_count is not None and data.shape[0] > max_data_count:
            indices = torch.randperm(data.shape[0])[:max_data_count]
            data = data[indices]
        self.data = data
        self.n_neighbors = n_neighbors
        self.dt = dt
        self.T = int(1 / self.dt)
        self.b_size = batch_size
        self.smooth_curve = smooth_curve

        self._A, self.knn = self.compute_connectivity_matrix(data, n_neighbors)
        self.W = self.compute_weights(self._A, data, batch_size)

        self.predecessors = self.get_predecessors(self.W)

    def compute_connectivity_matrix(self, data: torch.Tensor, n_neighbors: int):
        """
        Compute the connectivity matrix from the graph.
        If the graph is not weakly connected, we connect the different components by adding edges between the closest points of the different components.

        Parameters:
        data : torch.Tensor (N,D)
            The data points to use to compute the graph
        n_neighbors : int
            The number of neighbors to use for the KNN graph

        Returns:
        A : torch.Tensor (N,N)
            The adjacency matrix of the graph
        knn : NearestNeighbors
            The KNN object used for inference
        """
        # Maybe radius graph is better ?
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree", n_jobs=-1)
        knn.fit(data.cpu())

        # Find the Euclidean kNN
        N_data = data.shape[0]
        _, indices = knn.kneighbors(data.cpu())
        indices = indices[:, 1:]  # Remove the point itself, (N_data, k)
        A = np.zeros((N_data, N_data), dtype=int)
        batch_idx = torch.arange(N_data).view(-1, 1)
        A[batch_idx, indices] = 1
        if not is_connected(Graph(A)):
            print("Adjacency Graph is not connected : connecting ")
            A = self.connect_graph(A)
        return torch.from_numpy(A), knn

    @property
    def A(self):
        return self._A

    def load_adjacency_matrix(self, A: torch.Tensor):
        """
        This can be usefull in a graph settings where the
        connectivity matrix comes from elsewhere and we just
        want to fill in the weights with our metric
        We expect the data to come from the same place with the same ordering
        """
        self._A = A
        self.weakly_connected = is_connected(Graph(A.cpu().numpy()))
        if not self.weakly_connected:
            self._A = self.connect_graph(A)
        self.W = self.compute_weights(self._A, self.data, self.b_size)
        self.predecessors = self.get_predecessors(self.W)

    @torch.no_grad()
    def compute_weights(
        self, A: torch.Tensor, data: torch.Tensor, b_size: int = 64
    ) -> torch.Tensor:
        """
        Given the adjecency matrix, fills it using the metric.

        Parameters:
        A : torch.Tensor (n,n)
            Connectivity matrix
        data : torch.Tensor (n,d)
            The data points
        b_size: int
            The batch size for parallel processing

        Returns
        W : torch.Tensor (n,n)
            The weighted adjacency matrix where W[i,j] approx dst(x_i,x_j)
        """
        t = torch.arange(0, 1, self.dt, device=data.device, dtype=data.dtype).view(1, 1, -1, 1)
        N_data = data.shape[0]
        W = torch.zeros((N_data, N_data), device=data.device, dtype=data.dtype)

        pbar = tqdm(
            range(0, N_data, b_size),
            desc="Initialize Graph",
            unit="batch",
        )
        for start in pbar:
            end = min(start + b_size, N_data)
            batch_idx = torch.arange(start, end)
            edge_idx = A[batch_idx].nonzero(as_tuple=False)
            # When using a loaded A matrix, there can be a different number of neighbors for each point,
            # so we need to pad the edge_idx to have a fixed number of neighbors for each point
            padded_edge_idx = _get_padded_neighbor_indices(
                edge_idx,
                batch_size=batch_idx.shape[0],
                target_width=self.n_neighbors,
                pad_value=0,
            )
            curr_idx = padded_edge_idx[:, 1].view(batch_idx.shape[0], -1)

            p_i = data[batch_idx][:, None, None, :]
            p_j = data[curr_idx][:, :, None, :]

            linear_traj = p_i + t * (p_j - p_i)  # (b_size, k, T, d)
            linear_traj = rearrange(linear_traj, "b k T d -> (b k) T d")
            curve_length = self.compute_distance(linear_traj)
            curve_length = rearrange(curve_length, "(b k) -> b k", b=batch_idx.shape[0])
            W[batch_idx.view(-1, 1), curr_idx] = curve_length
            W[curr_idx, batch_idx.view(-1, 1)] = curve_length

        # Remove padded assignments introduced by variable-degree batches.
        W = W * A.to(device=W.device, dtype=W.dtype)

        W = W.to(device="cpu")
        W = 1 / 2 * (W + W.T)  # Ensure symmetry

        assert is_connected(
            Graph(W.numpy())
        ), "The graph is not connected after computing the weights. This should not happen."
        return W

    def get_cc_connections_idx(self, A, X):
        """
        Given the adjacency matrix A and the data points X,
        returns the indices of the connections between the connected components.

        Parameters:
        A (torch.Tensor): Adjacency matrix of shape (n, n).
        X (torch.Tensor): Data points of shape (n, d).

        Returns:
        idx_correspondence (torch.Tensor) of shape (m, 2):
        Indices of the connections between connected components.
            where idx_correspondence[i, 0] is the index of the first point connecting the i-th component
            and idx_correspondence[i, 1] is the index of the second point connecting the i-th component to its
            closest component.
        """
        # Get connected components from the adjacency matrix W
        G = Graph(A.cpu().numpy())
        cc_list = []
        for c in nx.connected_components(G):
            cc_list.append(torch.tensor(list(c)))

        # minus one because the last cluster
        # will have been compared to all previous ones
        idx_correspondence = torch.zeros(len(cc_list) - 1, 2, dtype=torch.long)

        for i in range(len(cc_list) - 1):
            bst_dst_i = float("inf")
            idx_i, idx_j = None, None
            X_cc_i = X[cc_list[i]]
            for j in range(i + 1, len(cc_list)):
                X_cc_j = X[cc_list[j]]

                dst_ij = torch.cdist(X_cc_i, X_cc_j, p=2)
                bst_dst = torch.min(dst_ij)
                if bst_dst < bst_dst_i:
                    bst_dst_i = bst_dst
                    bst_pts = torch.argmin(dst_ij)
                    idx_i, idx_j = torch.unravel_index(bst_pts, dst_ij.shape)
                    # Map idx_i and idx_j to the original indices
                    idx_i = cc_list[i][idx_i]
                    idx_j = cc_list[j][idx_j]
            idx_correspondence[i, 0] = idx_i
            idx_correspondence[i, 1] = idx_j
        return idx_correspondence

    def connect_graph(self, A: torch.Tensor) -> torch.Tensor:
        """
        Connect the connected components of the graph by adding dummy edges.
        This is a workaround to ensure that the graph is connected.
        The cc are connected via their closest points according to euclidean distance.

        Parameters:
        A (torch.Tensor):
            Adjacency matrix of shape (n, n).

        Returns:
        A (torch.Tensor):
            Adjacency matrix of shape (n, n) with added edges to connect the connected components.
        """
        idx_correspondence = self.get_cc_connections_idx(A, self.data)
        # a = self.data[idx_correspondence]
        # t = torch.arange(0, 1, self.dt, device=self.data.device, dtype=W.dtype).view(1, -1, 1)

        # p_i = a[:, 0][:, None, :]
        # p_j = a[:, 1][:, None, :]
        # linear_traj = p_i + t * (p_j - p_i)  # (n_cc,T,d)
        # dst = self.compute_distance(linear_traj).to(W.dtype)
        # W[idx_correspondence[:, 0], idx_correspondence[:, 1]] = dst
        # W[idx_correspondence[:, 1], idx_correspondence[:, 0]] = dst
        A[idx_correspondence[:, 0], idx_correspondence[:, 1]] = 1
        A[idx_correspondence[:, 1], idx_correspondence[:, 0]] = 1
        return A
        # return W

    def get_similarity_matrix(self, W, sigma=1):
        """Compute the similarity matrix using the weight matrix W

        Parameters
        ----------
        W : torch.Tensor (N,N)
            The weight matrix
        sigma : float
            The sigma to use for the similarity matrix

        Returns
        -------
        S : torch.Tensor (N,N)
            The similarity matrix
        """
        S = torch.exp(-W / (2 * sigma**2))
        return S

    def get_predecessors(self, W) -> torch.Tensor:
        """Get the predecessors for the shortest path computation..."""
        print("Computing predecessors...")
        predecessors = shortest_path(
            csr_matrix(W.cpu().numpy()),
            directed=False,
            return_predecessors=True,
            overwrite=True,
        )[1]
        print("Done.")
        return torch.from_numpy(predecessors)

    ## This version is actually even slower than the one above fuck this
    # def get_predecessors(self,W,device,censor_value=-9999) -> torch.Tensor:
    #     """
    #     Constructs a predecessors matrix from the paths dictionary.
    #     The predecessors matrix is a square matrix where the entry (i, j) contains the predecessor
    #     of node j on the shortest path from node i. If there is no path from i to j, the entry is censor_value.

    #     Parameters:
    #     W (torch.Tensor): The weight matrix of shape (n, n).
    #     device (torch.device): The device on which the computation is performed.
    #     censor_value (int): The value to use for entries where there is no path from i to j.

    #     Returns:
    #     np.ndarray: A 2D numpy array of shape (n, n) containing the predecessors.
    #     """
    #     G = nx.from_numpy_array(W.cpu().numpy())

    #     if device.type == "cuda":
    #         with nx.config.backends.cugraph(n_jobs=4):
    #             paths = nx.all_pairs_dijkstra(G, weight="weight")
    #     elif device.type == "cpu":
    #         with nx.config.backends.parallel(n_jobs=4):
    #             paths = nx.all_pairs_dijkstra(G, weight="weight")

    #     n = G.number_of_nodes()
    #     pred = np.full((n, n), censor_value, dtype=int)
    #     for node_i, (dict_distance_i, dict_path_i) in paths:  # Fixed variable name
    #         for node_j, paths_ij in dict_path_i.items():
    #             if len(paths_ij) > 1:
    #                 pred[node_i, node_j] = paths_ij[-2]
    #     return torch.from_numpy(pred)

    def linear_interpolation(self, p_i, p_j, t):
        """
        Linear interpolation between two points

        Parameters:
        ----------
        p_i : Tensor (b,d)
            The starting point
        p_j : Tensor (b,d)
            The ending point
        t : Tensor (n_pts,)

        Returns:
        --------
        traj_q : Tensor (b,n_pts,d)
            The linear interpolation between the two points
        """
        return p_i[:, None, :] + t[None, :, None] * (p_j - p_i)[:, None, :]

    def connect_to_graph(
        self, q0: Tensor, q1: Tensor, euclidean_only: bool = True
    ) -> tuple[Tensor, Tensor]:
        """
        Given two batch of points q0 and q1, find their closest point in the graph
        The closest points are determined by the linear interpolation metric
        distance (not euclidean) between the points.

        Parameters:
        -----------
        q0 : Tensor (B,d)
            start point
        q1 : Tensor (B,d)
            end point
        euclidean_only : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        closest_q0 : Tensor (B,)
            Indicices of the closest point in self.data to q0
        closest_q1 : Tensor (B,)
            Indicices of the closest point in self.data to q1
        """
        B = q0.shape[0]

        # Get the indices of the closest points in the graph
        _, ind0 = self.knn.kneighbors(q0.detach().cpu())
        _, ind1 = self.knn.kneighbors(q1.detach().cpu())

        if euclidean_only:
            closest_q0 = torch.from_numpy(ind0[:, 0])
            closest_q1 = torch.from_numpy(ind1[:, 0])
            return closest_q0.int(), closest_q1.int()

        # Get rid of the last one that we added for initialisation
        ind0 = ind0[:, :-1]  # (B,K)
        ind1 = ind1[:, :-1]

        q0_neighbors = torch.zeros(
            B, self.n_neighbors, self.data.shape[1], device=q0.device, dtype=q0.dtype
        )  # (B,K,D)
        q1_neighbors = torch.zeros(
            B, self.n_neighbors, self.data.shape[1], device=q0.device, dtype=q0.dtype
        )

        for n in range(self.n_neighbors):
            q0_neighbors[:, n, :] = self.data[ind0[:, n]]
            q1_neighbors[:, n, :] = self.data[ind1[:, n]]

        # Find the points with the closest metric distance to q0 and q1
        t = torch.arange(0, 1, self.dt, device=q0.device, dtype=q0.dtype)  # (n_pts,)

        closest_q0 = torch.zeros(B).int()
        closest_q1 = torch.zeros(B).int()

        for b in range(B):
            q0b = q0[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q0b_neigh = q0_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q0b, q0b_neigh, t
            )  # (n_neighbors, n_pts, d)
            dst_to_q0b = self.compute_distance(linear_traj)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q0b)
            closest_q0[b] = ind0[b, argmin_length]

            q1b = q1[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q1b_neigh = q1_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q1b, q1b_neigh, t
            )  # (n_neighbors, n_pts, d)
            dst_to_q1b = self.compute_distance(linear_traj)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q1b)
            closest_q1[b] = ind1[b, argmin_length]
        return closest_q0, closest_q1

    def get_path_idx(self, start_idx: Tensor, end_idx: Tensor, max_path_length: int = 2000):
        """
        Given the start and end indices, retrieve the path in the graph.

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices
        end_idx : Tensor (B,)
            The ending indices
        max_path_length : int
            The maximum length of the path to retrieve for security reasons.

        Returns:
        --------
        path : Tensor (B,max_traj_length)
            The path in the graph. The value -9999 means the path is done.
        """
        B = start_idx.shape[0]
        temp_idx = end_idx
        path = [end_idx]

        iter = 0
        nb_pts_in_path = torch.zeros(B)
        # Simply properly retrieve the path
        while (nb_pts_in_path == 0).any() and iter < max_path_length:
            iter += 1
            pred_idx = self.predecessors[start_idx, temp_idx]
            if (pred_idx == -9999).any():
                # A path is done
                b_end_path = torch.argwhere(pred_idx == -9999).squeeze(1)
                b_end_path = torch.zeros(B, dtype=bool).scatter(0, b_end_path, True)
                lenght_to_update = (nb_pts_in_path == 0) & b_end_path
                nb_pts_in_path[lenght_to_update] = iter
                temp_idx[b_end_path] = start_idx[b_end_path]
                temp_idx[~b_end_path] = pred_idx[~b_end_path]
            else:
                temp_idx = pred_idx
            path.append(pred_idx.clone())

        path = torch.stack(path, dim=1)
        return path

    def get_pts_from_idx(self, start_idx: Tensor, path_idx: Tensor, censor_value: int = -9999):
        """From the indices, retrieve the points in the graph

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices of the path.
        path_idx : Tensor (B,max_traj_length)
            The indices of the points in the graph
        censor_value : int
            The value that indicates the end of the path.

        Returns:
        --------
        path : Tensor (B,max_traj_length,d)
            The points in the graph. The end of the path are padded to max_traj_length with the last valid point.
        """
        maks = path_idx == censor_value
        replacement_values = start_idx.unsqueeze(1).expand(-1, path_idx.shape[1])
        # We replace the censor_value by the starting point
        path_idx = torch.where(maks, replacement_values, path_idx)
        pts_on_traj = self.data[path_idx]
        return pts_on_traj

    def get_trajectories(self, q0: Tensor, q1: Tensor, connect_euclidean: bool = False):
        """
        Compute the geodesic trajectories between two points.
        This amounts to properly connecting the points to the KNN graph and then
        computing the shortest path between the two points.

        Parameters:
        -----------
        q0 : Tensor (b,d)
            The starting points
        q1 : Tensor (b,d)
            The ending points
        connect_euclidean : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        pts_on_traj : Tensor (b,T,d)
            The points on the trajectory
        """
        start_idx, end_idx = self.connect_to_graph(q0, q1, euclidean_only=connect_euclidean)
        path_idx = self.get_path_idx(start_idx, end_idx)

        pts_on_traj = torch.zeros(
            q0.shape[0], self.T, q0.shape[1], device=q0.device, dtype=q0.dtype
        )
        for b in range(q0.shape[0]):
            pts_on_traj[b] = self.reparametrize_curve(
                path_idx[b], q0[b], q1[b], smooth_curve=self.smooth_curve
            )

        # pts_on_traj = self.get_pts_from_idx(start_idx, path_idx)
        pts_on_traj = torch.cat([q1[:, None, :], pts_on_traj, q0[:, None, :]], dim=1)

        # Reverse end->start to start->end
        pts_on_traj = pts_on_traj.flip(1)
        return pts_on_traj

    def reparametrize_curve(
        self, idx_traj: Tensor, q0, q1, censor_value=-9999, smooth_curve=True
    ) -> Tensor:
        """
        Reparametrize a single curve to have a fixed number of points using a cubic spline.

        Parameters:
        -----------
        idx_traj : Tensor (n_pts)
            The indices of the points on the trajectory.
        q0 : Tensor (d,)
            The starting point
        q1 : Tensor (d,)
            The ending point
        censor_value : int
            The value that indicates the end of the path in idx_traj
        smooth_curve : bool
            If True, smooth the curve with a mean kernel before feeding it to the spline

        Returns:
        --------
        traj_q : Tensor (n_pts,d)
            The reparametrized trajectories
        """
        # Find the first occurence of the censor value
        mask = idx_traj == censor_value
        first_censor = torch.argmax(mask.int())
        # Retrieve the correct trajectory
        idx_traj = idx_traj[:first_censor]
        traj_q = self.data[idx_traj]
        traj_q = torch.cat([q1.unsqueeze(0), traj_q, q0.unsqueeze(0)], dim=0)

        if smooth_curve:
            # Smooth the curve with a mean kernel
            # Pad the curve with two  points at the start and end
            p0 = traj_q[0].unsqueeze(0)
            p1 = traj_q[-1].unsqueeze(0)
            traj_q = torch.cat([p0, p0, traj_q, p1, p1], dim=0)
            traj_q = traj_q.unfold(0, 3, 1).mean(dim=2)

        # Resample the curve
        cs = CubicSpline(np.linspace(0, 1, traj_q.shape[0]), traj_q.detach().cpu())
        traj_q = cs(np.arange(0, 1, self.dt))
        traj_q = torch.from_numpy(traj_q).to(q0.device).to(q0.dtype)
        return traj_q


class CascadeSolver(GeodesicDistanceSolver):
    def __init__(
        self, cometric: CoMetric, data: Tensor, n_neighbors: int, dt: float, dim: int
    ) -> None:
        super().__init__(cometric)
        self.solver_init = SolverGraph(cometric, data, n_neighbors, dt)
        self.T = self.solver_init.T
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


def dst_mat(a: Tensor, b: Tensor, dst_func: GeodesicDistanceSolver) -> Tensor:
    """
    Compute geodesic distances between the points a and b according to the metric g

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistanceSolver, function that computes the geodesic distance between two batch of points

    Output:
    dst : Tensor (b, b) matrix of geodesic distances where dst[i,j] = dst_func(a[i],b[j])
    """
    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    dst = torch.zeros(B, B, device=a.device)

    # row wise filling
    for i in tqdm(range(B), desc="Computing distance matrix", unit="row"):
        a_batch = a[i].unsqueeze(0).expand(B, -1)
        distances = dst_func(a_batch, b)
        dst[i] = distances

    return dst


def angle_mat(a: Tensor, b: Tensor, dst_func: GeodesicDistanceSolver) -> Tensor:
    """
    Compute geodesic angle between the points a and b according to the metric g

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistanceSolver, function that computes the geodesic distance between two batch of points

    Output:
    dst : Tensor (b, b) matrix of geodesic similarity where dst[i,j] = dst_func.angle(a[i],b[j])
    """
    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    dst = torch.zeros(B, B, device=a.device)

    # row wise filling
    for i in range(B):
        a_batch = a[i].unsqueeze(0).expand(B, -1)
        distances = dst_func.angle(a_batch, b)
        dst[i] = distances

    return dst


def dst_mat_naive(a: Tensor, b: Tensor, dst_func: GeodesicDistanceSolver) -> Tensor:
    """
    Compute geodesic distances between the points a and b according to the metric g

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistanceSolver, function that computes the geodesic distance between two batch of points

    Output:
    dst : Tensor (b, b) matrix of geodesic distances where dst[i,j] = dst_func(a[i],b[j])
    """
    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    dst = torch.zeros(B, B, device=a.device)

    for i in range(B):
        for j in range(B):
            distances = dst_func(a[i].unsqueeze(0), b[j].unsqueeze(0))
            dst[i, j] = distances

    return dst


def dst_mat_vectorized(
    a: Tensor,
    b: Tensor,
    dst_func: GeodesicDistanceSolver,
) -> Tensor:
    """
    Compute geodesic distances between points a and b efficiently.
    Vectorizes the computation to avoid explicit loops.
    Beware that this is memory intensive (risk of OOM) and might not be faster
    than the naive implementation for small batch sizes.

    Note that the result is not exactly the same as the naive implementation
    because of the optimisation process that optimize for the whole batch at once.

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistanceSolver, function that computes geodesic distances

    Output:
    dst : Tensor (b, b) matrix of geodesic distances where dst[i,j] = dst_func(a[i],b[j])
    """
    # @TODO add automatic routing of method based on batch size

    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    # Expand dimensions to create all pairs
    # Shape: (B, 1, dim) and (1, B, dim)
    a_expanded = a.unsqueeze(1)  # Shape: (B, 1, dim)
    b_expanded = b.unsqueeze(0)  # Shape: (1, B, dim)

    # Broadcast to (B, B, dim)
    a_tiled = a_expanded.expand(B, B, embed_dim)
    b_tiled = b_expanded.expand(B, B, embed_dim)

    # Reshape to (B*B, dim) for batch processing
    a_flat = a_tiled.reshape(-1, embed_dim)
    b_flat = b_tiled.reshape(-1, embed_dim)

    # Compute distances for all pairs at once
    distances_flat = dst_func(a_flat, b_flat)

    # Reshape back to (B, B)
    distances = distances_flat.reshape(B, B)

    return distances


class SolverGraphFinsler(torch.nn.Module):
    """Computes the geodesic distances between points using a KNN-graph
    for metrics from a Finsler space.

    Parameters
    ----------
    finsler_metric : FinslerMetrics
        The metric to use to compute the geodesic distances
    data : torch.Tensor (N,D)
        The data points used to compute the graph
    n_neighbors : int
        The number of neighbors to use
    dt : float
        The time step to use for the linear interpolation
    batch_size : int
        The size of the batch to use for the computation of the graph
    max_data_count : int
        The maximum number of data points to use for the graph. If None, all data points
        are used. If the number of data points is larger than this value, the graph is
        computed on a random subset of the data points to save memory.
    smooth_curve : bool
        If True, smooth the curve with a mean kernel before feeding it to the spline
        when reparametrizing the trajectory.
    """

    def __init__(
        self,
        finsler_metric: FinslerMetric,
        data: torch.Tensor,
        n_neighbors: int,
        dt: float = 0.01,
        batch_size: int = 64,
        max_data_count: int | None = None,
        smooth_curve: bool = True,
    ) -> None:
        super().__init__()
        self.finsler_metric = finsler_metric

        if max_data_count is not None and data.shape[0] > max_data_count:
            indices = torch.randperm(data.shape[0])[:max_data_count]
            data = data[indices]
        self.data = data
        self.n_neighbors = n_neighbors
        self.dt = dt
        self.T = int(1 / self.dt)
        self.b_size = batch_size
        self.smooth_curve = smooth_curve

        # self.W, self.knn = self.init_knn_graph(data, n_neighbors, batch_size)
        # self.predecessors = self.get_predecessors()

        self._A, self.knn = self.compute_connectivity_matrix(data, n_neighbors)
        self.W = self.compute_weights(self._A, data, batch_size)

        self.predecessors = self.get_predecessors(self.W)

    def compute_connectivity_matrix(self, data: torch.Tensor, n_neighbors: int):
        """
        Compute the connectivity matrix from the graph.
        If the graph is not weakly connected, we connect the different components by adding edges between the closest points of the different components.

        Parameters:
        data : torch.Tensor (N,D)
            The data points to use to compute the graph
        n_neighbors : int
            The number of neighbors to use for the KNN graph

        Returns:
        A : torch.Tensor (N,N)
            The adjacency matrix of the graph
        knn : NearestNeighbors
            The KNN object used for inference
        """
        # Maybe radius graph is better ?
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree", n_jobs=-1)
        knn.fit(data.cpu())

        # Find the Euclidean kNN
        N_data = data.shape[0]
        _, indices = knn.kneighbors(data.cpu())
        indices = indices[:, 1:]  # Remove the point itself, (N_data, k)
        A = np.zeros((N_data, N_data), dtype=int)
        batch_idx = torch.arange(N_data).view(-1, 1)
        A[batch_idx, indices] = 1
        if not is_connected(Graph(A)):
            print("Adjacency Graph is not connected : connecting ")
            A = self.connect_graph(A)
        return torch.from_numpy(A), knn

    @property
    def A(self):
        return self._A

    def load_adjacency_matrix(self, A: torch.Tensor):
        """Load an adjacency matrix from an external source and compute the weights and predecessors.
        This can be useful in a graph settings where the connectivity matrix comes from elsewhere and we just
        want to fill in the weights with our metric. We expect the data to come from the same place with the same ordering.

        Parameters:
        A : torch.Tensor (n,n)
            The adjacency matrix to load
        """
        print("Loading adjacency matrix...")
        self._A = A
        self.weakly_connected = is_connected(Graph(A.cpu().numpy()))
        if not self.weakly_connected:
            print("Reconnecting the new mat")
            self._A = self.connect_graph(A)
        print("Recomputing weights")
        self.W = self.compute_weights(self._A, self.data, self.b_size)
        print("Computing precedecessors")
        self.predecessors = self.get_predecessors(self.W)
        print("Over.")

    @torch.no_grad()
    def compute_weights(
        self, A: torch.Tensor, data: torch.Tensor, b_size: int = 64
    ) -> torch.Tensor:
        """
        Given the adjecency matrix, fills it using the metric.

        Parameters:
        A : torch.Tensor (n,n)
            Connectivity matrix
        data : torch.Tensor (n,d)
            The data points
        b_size: int
            The batch size for parallel processing

        Returns
        W : torch.Tensor (n,n)
            The weighted adjacency matrix where W[i,j] approx dst(x_i,x_j)
        """
        t = torch.arange(0, 1, self.dt, device=data.device, dtype=data.dtype).view(1, 1, -1, 1)
        N_data = data.shape[0]
        W = torch.zeros((N_data, N_data), device=data.device, dtype=data.dtype)

        pbar = tqdm(
            range(0, N_data, b_size),
            desc="Initialize Graph",
            unit="batch",
        )
        for start in pbar:
            end = min(start + b_size, N_data)
            batch_idx = torch.arange(start, end)            
            
            edge_idx = A[batch_idx].nonzero(as_tuple=False)
            # When using a loaded A matrix, there can be a different number of neighbors for each point,
            # so we need to pad the edge_idx to have a fixed number of neighbors for each point
            padded_edge_idx = _get_padded_neighbor_indices(
                edge_idx,
                batch_size=batch_idx.shape[0],
                target_width=self.n_neighbors,
                pad_value=0,
            )
            curr_idx = padded_edge_idx[:, 1].view(batch_idx.shape[0], -1)

            p_i = data[batch_idx][:, None, None, :] # (b_size, 1, 1, d)
            p_j = data[curr_idx][:, :, None, :] # (b_size, n_neighbors, 1, d)

            linear_traj = p_i + t * (p_j - p_i)  # (b_size, k, T, d)
            linear_traj = rearrange(linear_traj, "b k T d -> (b k) T d")
            curve_length = self.compute_distance(linear_traj)
            curve_length = rearrange(curve_length, "(b k) -> b k", b=batch_idx.shape[0])
            W[batch_idx.view(-1, 1), curr_idx] = curve_length

        # Remove padded assignments introduced by variable-degree batches.
        W = W * A.to(device=W.device, dtype=W.dtype)

        W = W.to(device="cpu")

        assert is_strongly_connected(
            DiGraph(W.numpy())
        ), "The graph is not connected after computing the weights. This should not happen."
        return W

    # @torch.no_grad()
    # def init_knn_graph(
    #     self, data: torch.Tensor, n_neighbors: int, b_size: int = 64
    # ) -> tuple[np.ndarray, NearestNeighbors]:
    #     """Initialize the graph using a KNN graph.

    #     Parameters
    #     ----------
    #     data : torch.Tensor (N,D)
    #         The data points
    #     n_neighbors : int
    #         The number of neighbors to use
    #     b_size : int
    #         The size of the batch to use for the computation

    #     Returns
    #     -------
    #     W : torch.Tensor (N,N)
    #         The weight matrix of the graph
    #     knn : NearestNeighbors
    #         The KNN object
    #     """
    #     # We add one to the number of neighbors to remove the point itself
    #     knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree")
    #     knn.fit(data.cpu())
    #     t = torch.arange(0, 1, self.dt, device=data.device, dtype=data.dtype).view(
    #         1, 1, -1, 1
    #     )  # (1,1,T,1)

    #     # Find the Euclidean kNN
    #     N_data = data.shape[0]
    #     _, indices = knn.kneighbors(data.cpu())
    #     indices = indices[:, 1:]  # Remove the point itself
    #     Weight_matrix = torch.zeros((N_data, N_data), device=data.device, dtype=data.dtype)

    #     pbar = tqdm(
    #         range(0, N_data, self.b_size),
    #         desc="Initialize Graph",
    #         unit="batch",
    #     )
    #     with torch.no_grad():
    #         for start in pbar:
    #             end = min(start + b_size, N_data)
    #             batch_idx = torch.arange(start, end)  # (b,)
    #             curr_idx = indices[batch_idx]  # (b,k)
    #             # Sometimes the batch indices amounts to only one
    #             # To ensure the dimension is correct, we unsqueeze the batch dimension in that case
    #             # This is just because numpy (or me idk) sucks and doesn't keep dim
    #             if curr_idx.ndim == 1:
    #                 curr_idx = curr_idx[None, :]  # (1,k)

    #             p_i = data[batch_idx][:, None, None, :]  # (b,1,1,d)
    #             p_j = data[curr_idx][:, :, None, :]  # (b,k,1,d)

    #             linear_traj = p_i + t * (p_j - p_i)  # (b,k,T,d)
    #             tangent_vectors = p_j - p_i
    #             tangent_vectors = tangent_vectors.expand(-1, -1, self.T, -1)
    #             linear_traj = rearrange(linear_traj, "b k T d -> (b k) T d")
    #             tangent_vectors = rearrange(tangent_vectors, "b k T d -> (b k) T d")
    #             curve_length = self.compute_distance(linear_traj, tangent_vectors)

    #             curve_length = rearrange(curve_length, "(b k) -> b k", b=batch_idx.shape[0])
    #             Weight_matrix[batch_idx.view(-1, 1), curr_idx] = curve_length

    #     if not is_strongly_connected(DiGraph(Weight_matrix.cpu().numpy())):
    #         Weight_matrix = self.connect_graph(Weight_matrix)
    #     return Weight_matrix.to("cpu"), knn

    def get_cc_connections_idx(self, A, X):
        """
        Given the adjacency matrix A and the data points X,
        returns the indices of the connections between the connected components.

        Parameters:
        A (torch.Tensor): Adjacency matrix of shape (n, n).
        X (torch.Tensor): Data points of shape (n, d).

        Returns:
        idx_correspondence (torch.Tensor) of shape (m, 2):
        Indices of the connections between connected components.
            where idx_correspondence[i, 0] is the index of the first point connecting the i-th component
            and idx_correspondence[i, 1] is the index of the second point connecting the i-th component to its
            closest component.
        """
        # Get connected components from the adjacency matrix A
        G = DiGraph(A.cpu().numpy())
        cc_list = []
        for c in nx.strongly_connected_components(G):
            cc_list.append(torch.tensor(list(c)))

        # minus one because the last cluster
        # will have been compared to all previous ones
        idx_correspondence = torch.zeros(len(cc_list) - 1, 2, dtype=torch.long)

        for i in range(len(cc_list) - 1):
            bst_dst_i = float("inf")
            idx_i, idx_j = None, None
            X_cc_i = X[cc_list[i]]
            for j in range(i + 1, len(cc_list)):
                X_cc_j = X[cc_list[j]]

                dst_ij = torch.cdist(X_cc_i, X_cc_j, p=2)
                bst_dst = torch.min(dst_ij)
                if bst_dst < bst_dst_i:
                    bst_dst_i = bst_dst
                    bst_pts = torch.argmin(dst_ij)
                    idx_i, idx_j = torch.unravel_index(bst_pts, dst_ij.shape)
                    # Map idx_i and idx_j to the original indices
                    idx_i = cc_list[i][idx_i]
                    idx_j = cc_list[j][idx_j]
            idx_correspondence[i, 0] = idx_i
            idx_correspondence[i, 1] = idx_j
        return idx_correspondence

    def connect_graph(self, A: torch.Tensor) -> torch.Tensor:
        """
        Connect the connected components of the graph by adding dummy edges.
        This is a workaround to ensure that the graph is connected.
        The cc are connected via their closest points according to euclidean distance.

        Parameters:
        A (torch.Tensor):
            Adjacency matrix of shape (n, n).

        Returns:
        A (torch.Tensor):
            Adjacency matrix of shape (n, n) with added edges to connect the connected components.
        """
        idx_correspondence = self.get_cc_connections_idx(A, self.data)
        A[idx_correspondence[:, 0], idx_correspondence[:, 1]] = 1
        A[idx_correspondence[:, 1], idx_correspondence[:, 0]] = 1
        return A

    def compute_distance(self, traj: torch.Tensor, tangent_vectors: torch.Tensor = None):
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
            [self.finsler_metric(m, seg) for m, seg in zip(traj, tangent_vectors)]
        )  # (B, T)
        distances = distances.relu().sum(dim=1)  # (B,)
        return distances

    def get_similarity_matrix(self, W, sigma=1):
        """Compute the similarity matrix using the weight matrix W

        Parameters
        ----------
        W : torch.Tensor (N,N)
            The weight matrix
        sigma : float
            The sigma to use for the similarity matrix

        Returns
        -------
        S : torch.Tensor (N,N)
            The similarity matrix
        """
        S = torch.exp(-W / (2 * sigma**2))
        return S

    def get_predecessors(self, W) -> torch.Tensor:
        """Get the predecessors for the shortest path computation..."""
        print("Computing predecessors...")
        dst_matrix, predecessors = shortest_path(
            csr_matrix(W.cpu().numpy()),
            directed=True,
            return_predecessors=True,
        )
        print("Done.")
        return torch.from_numpy(predecessors)

    def linear_interpolation(self, p_i, p_j, t):
        """
        Linear interpolation between two points

        Parameters:
        ----------
        p_i : Tensor (b,d)
            The starting point
        p_j : Tensor (b,d)
            The ending point
        t : Tensor (n_pts,)

        Returns:
        --------
        traj_q : Tensor (b,n_pts,d)
            The linear interpolation between the two points
        """
        return p_i[:, None, :] + t[None, :, None] * (p_j - p_i)[:, None, :]

    def connect_to_graph(
        self, q0: Tensor, q1: Tensor, euclidean_only: bool = True
    ) -> tuple[Tensor, Tensor]:
        """
        Given two batch of points q0 and q1, find their closest point in the graph
        The closest points are determined by the linear interpolation metric
        distance (not euclidean) between the points.

        Parameters:
        -----------
        q0 : Tensor (B,d)
            start point
        q1 : Tensor (B,d)
            end point
        euclidean_only : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        closest_q0 : Tensor (B,)
            Indicices of the closest point in self.data to q0
        closest_q1 : Tensor (B,)
            Indicices of the closest point in self.data to q1
        """
        B = q0.shape[0]

        # Get the indices of the closest points in the graph
        _, ind0 = self.knn.kneighbors(q0.detach().cpu())
        _, ind1 = self.knn.kneighbors(q1.detach().cpu())

        if euclidean_only:
            closest_q0 = torch.from_numpy(ind0[:, 0])
            closest_q1 = torch.from_numpy(ind1[:, 0])
            return closest_q0.int(), closest_q1.int()

        # Get rid of the last one that we added for initialisation
        ind0 = ind0[:, :-1]  # (B,K)
        ind1 = ind1[:, :-1]

        q0_neighbors = torch.zeros(
            B, self.n_neighbors, self.data.shape[1], device=q0.device, dtype=q0.dtype
        )  # (B,K,D)
        q1_neighbors = torch.zeros(
            B, self.n_neighbors, self.data.shape[1], device=q0.device, dtype=q0.dtype
        )

        for n in range(self.n_neighbors):
            q0_neighbors[:, n, :] = self.data[ind0[:, n]]
            q1_neighbors[:, n, :] = self.data[ind1[:, n]]

        # Find the points with the closest metric distance to q0 and q1
        t = torch.arange(0, 1, self.dt, device=q0.device, dtype=q0.dtype)

        closest_q0 = torch.zeros(B).int()
        closest_q1 = torch.zeros(B).int()

        for b in range(B):
            q0b = q0[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q0b_neigh = q0_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q0b, q0b_neigh, t
            )  # (n_neighbors, n_pts, d)
            tangent_vectors = (q0b_neigh - q0b)[:, None, :].repeat(
                1, self.T, 1
            )  # (n_neighbors, n_pts, d)
            dst_to_q0b = self.compute_distance(linear_traj, tangent_vectors)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q0b)
            closest_q0[b] = ind0[b, argmin_length]

            q1b = q1[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q1b_neigh = q1_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q1b, q1b_neigh, t
            )  # (n_neighbors, n_pts, d)
            tangent_vectors = (q1b_neigh - q1b)[:, None, :].repeat(
                1, self.T, 1
            )  # (n_neighbors, n_pts, d)
            dst_to_q1b = self.compute_distance(linear_traj, tangent_vectors)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q1b)
            closest_q1[b] = ind1[b, argmin_length]
        return closest_q0, closest_q1

    def get_path_idx(self, start_idx: Tensor, end_idx: Tensor, max_path_length: int = 2000):
        """
        Given the start and end indices, retrieve the path in the graph.

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices
        end_idx : Tensor (B,)
            The ending indices
        max_path_length : int
            The maximum length of the path. If the path is longer, it will be truncated.

        Returns:
        --------
        path : Tensor (B,max_traj_length)
            The path in the graph. The value -9999 means the path is done.
        """
        B = start_idx.shape[0]
        temp_idx = end_idx
        path = [end_idx]

        iter = 0
        nb_pts_in_path = torch.zeros(B)
        # Simply properly retrieve the path
        while (nb_pts_in_path == 0).any() and iter < max_path_length:
            iter += 1
            pred_idx = self.predecessors[start_idx, temp_idx]
            if (pred_idx == -9999).any():
                # A path is done
                b_end_path = torch.argwhere(pred_idx == -9999).squeeze(1)
                b_end_path = torch.zeros(B, dtype=bool).scatter(0, b_end_path, True)
                lenght_to_update = (nb_pts_in_path == 0) & b_end_path
                nb_pts_in_path[lenght_to_update] = iter
                temp_idx[b_end_path] = start_idx[b_end_path]
                temp_idx[~b_end_path] = pred_idx[~b_end_path]
            else:
                temp_idx = pred_idx
            path.append(pred_idx.clone())

            if iter >= max_path_length:
                print("Max path length reached")
                break

        path = torch.stack(path, dim=1)
        return path

    def get_pts_from_idx(self, start_idx: Tensor, path_idx: Tensor, censor_value: int = -9999):
        """From the indices, retrieve the points in the graph

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices of the path.
        path_idx : Tensor (B,max_traj_length)
            The indices of the points in the graph
        censor_value : int
            The value that indicates the end of the path.

        Returns:
        --------
        path : Tensor (B,max_traj_length,d)
            The points in the graph. The end of the path are padded to max_traj_length with the last valid point.
        """
        maks = path_idx == censor_value
        replacement_values = start_idx.unsqueeze(1).expand(-1, path_idx.shape[1])
        # We replace the censor_value by the starting point
        path_idx = torch.where(maks, replacement_values, path_idx)
        pts_on_traj = self.data[path_idx]
        return pts_on_traj

    def get_trajectories(self, q0: Tensor, q1: Tensor, connect_euclidean: bool = False):
        """
        Compute the geodesic trajectories between two points.
        This amounts to properly connecting the points to the KNN graph and then
        computing the shortest path between the two points.

        Parameters:
        -----------
        q0 : Tensor (b,d)
            The starting points
        q1 : Tensor (b,d)
            The ending points
        connect_euclidean : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        pts_on_traj : Tensor (b,T,d)
            The points on the trajectory
        """
        start_idx, end_idx = self.connect_to_graph(q0, q1, euclidean_only=connect_euclidean)
        path_idx = self.get_path_idx(start_idx, end_idx)

        pts_on_traj = torch.zeros(
            q0.shape[0], self.T, q0.shape[1], device=q0.device, dtype=q0.dtype
        )
        for b in range(q0.shape[0]):
            pts_on_traj[b] = self.reparametrize_curve(
                path_idx[b], q0[b], q1[b], smooth_curve=self.smooth_curve
            )

        # pts_on_traj = self.get_pts_from_idx(start_idx, path_idx)
        pts_on_traj = torch.cat([q1[:, None, :], pts_on_traj, q0[:, None, :]], dim=1)

        # Reverse end->start to start->end
        pts_on_traj = pts_on_traj.flip(1)
        return pts_on_traj

    def reparametrize_curve(
        self, idx_traj: Tensor, q0, q1, censor_value=-9999, smooth_curve=True
    ) -> Tensor:
        """
        Reparametrize a single curve to have a fixed number of points using a cubic spline.

        Parameters:
        -----------
        idx_traj : Tensor (n_pts)
            The indices of the points on the trajectory.
        q0 : Tensor (d,)
            The starting point
        q1 : Tensor (d,)
            The ending point
        censor_value : int
            The value that indicates the end of the path in idx_traj
        smooth_curve : bool
            If True, smooth the curve with a mean kernel before feeding it to the spline

        Returns:
        --------
        traj_q : Tensor (n_pts,d)
            The reparametrized trajectories
        """
        # Find the first occurence of the censor value
        mask = idx_traj == censor_value
        first_censor = torch.argmax(mask.int())
        # Retrieve the correct trajectory
        idx_traj = idx_traj[:first_censor]
        traj_q = self.data[idx_traj]
        traj_q = torch.cat([q1.unsqueeze(0), traj_q, q0.unsqueeze(0)], dim=0)

        if smooth_curve:
            # Smooth the curve with a mean kernel
            # Pad the curve with two  points at the start and end
            p0 = traj_q[0].unsqueeze(0)
            p1 = traj_q[-1].unsqueeze(0)
            traj_q = torch.cat([p0, p0, traj_q, p1, p1], dim=0)
            traj_q = traj_q.unfold(0, 3, 1).mean(dim=2)

        # Resample the curve
        cs = CubicSpline(np.linspace(0, 1, traj_q.shape[0]), traj_q.detach().cpu())
        traj_q = cs(np.arange(0, 1, self.dt))
        traj_q = torch.from_numpy(traj_q).float().to(q0.device)
        return traj_q

    def forward(self, q0: Tensor, q1: Tensor, connect_euclidean: bool = False):
        """
        Computes the geodesic distances between two points q0 and q1 using the KNN graph.

        Parameters:
        -----------
        q0 : Tensor (b,d)
            The starting points
        q1 : Tensor (b,d)
            The ending points
        connect_euclidean : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        dst : Tensor (b,)
            The geodesic distances between the two points
        """
        pts_on_traj = self.get_trajectories(q0, q1, connect_euclidean=connect_euclidean)
        traj_front = pts_on_traj[:, 1:, :]
        traj_back = pts_on_traj[:, :-1, :]
        segments = traj_front - traj_back
        midpoints = (traj_front + traj_back) / 2

        dst = self.compute_distance(midpoints, segments)

        return dst


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

    def compute_energy(self, z_t, z0, zT):
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
        dx = traj[1:] - traj[:-1]  # (T, d)
        energy = self.cometric.metric(traj[:-1], dx)
        # G = self.cometric.metric_tensor(traj[:-1])  # (T, d, d)
        # energy = torch.einsum("ti,tij,tj->t", dx, G, dx)  # (T,)
        return energy.sum()

    def grad_E(self, z_t, z0, zT):
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

    def get_mut_t(self, v_t, G_inv_t, diff):
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

    def line_search(self, x_0, x_T, u_t, u_t_i, x_t_i):
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

        while condition and curr_iter < self.max_iter:
            alpha *= self.rho  # Reduce step size
            curr_iter += 1

            with torch.no_grad():
                x_t_new = x_0 + torch.cumsum(
                    alpha * u_t[:-1] + (1 - alpha) * u_t_i[:-1], dim=0
                )
                E_new = self.compute_energy(x_t_new, x_0, x_T)
            val = E_0 + self.c * alpha * torch.dot(grad_E_0, p_k)
            condition = E_new > val

        if curr_iter == self.max_iter:
            print(f"Warning: Maximum iterations reached in line search. {alpha=}")

        return alpha

    def dst_func(self, x_0, x_T, x_t) -> Tensor:
        """
        Compute the geodesic distance between x_0 and x_T along the trajectory x_t.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        x_t: Tensor (T-1, d)
            The geodesic trajectory points. It does not include x_0 and x_T.

        Returns:
        --------
        distance: Tensor
            The geodesic distance between x_0 and x_T along the trajectory x_t.
        """
        full_traj = torch.cat([x_0[None, :], x_t, x_T[None, :]], dim=0)  # (T+1, d)
        dx = full_traj[1:] - full_traj[:-1]  # (T, d)
        distance = self.cometric.metric(full_traj[:-1], dx).sqrt()  # (T,)
        return distance.sum()

    def georce_solver(
        self,
        x_0: Tensor,
        x_T: Tensor,
        x_t_0: Tensor = None,
    ):
        """
        Solve the geodesic equation using the GEORCE algorithm.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        x_t_0: Tensor (T-1, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_T.
            It must not contain x_0 and x_T.

        Returns:
        -------
        x_final: Tensor (T+1, d)
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

        if x_t_0 is None:
            t = torch.linspace(0, 1, self.T + 1, device=x_0.device, dtype=x_0.dtype)
            x_t_0 = x_0[None, :] + t[1:-1, None] * (x_T - x_0)[None, :]  # (T-1, d)
        else:
            assert x_t_0.shape == (
                self.T - 1,
                d,
            ), f"x_t_0 must have shape {(self.T - 1, d)=} got {x_t_0.shape=}. But sure to exclude x_0 and x_T."

        # Precompute cometric at start without building graph
        with torch.no_grad():
            G_inv_0 = self.cometric.cometric_tensor(x_0[None, :]).squeeze(0)

        diff = x_T - x_0

        x_t_i = x_t_0.clone().requires_grad_(True)  # (T-1, d), not including x_0 and x_T
        u_t_i = (
            diff * torch.ones(self.T, d, device=x_0.device, dtype=x_0.dtype) / self.T
        )  # (T, d) Initial guess of the velocity, not including x_T

        # L4
        grad_E_t = self.grad_E(x_t_i, x_0, x_T)
        norm_grad_E_t = torch.linalg.vector_norm(grad_E_t.reshape(-1))
        norm_grad_E_t = norm_grad_E_t / (
            grad_E_t.shape[0] * grad_E_t.shape[1]
        )  # Normalize by the dimension

        with torch.no_grad():
            dst_list = [self.dst_func(x_0, x_T, x_t_i).item()]
        norm_gE_list = [norm_grad_E_t.item()]
        with torch.no_grad():
            E_list = [self.compute_energy(x_t_i, x_0, x_T).item()]
        alpha_list = [1.0]

        if self.pbar:
            pbar = tqdm(range(self.max_iter), total=self.max_iter, desc="Iterations")
        # for i in pbar:
        while (norm_grad_E_t > self.tol) & (i < self.max_iter):
            # L5
            G_inv_t = self.cometric.cometric_tensor(x_t_i)
            G_inv_t = torch.cat([G_inv_0[None, :], G_inv_t], dim=0)  # (T, d, d) | (T, d, d)
            if G_inv_t.ndim == 3:
                G_t = G_inv_t.inverse()  # (T, d, d)
            else:
                G_t = 1 / G_inv_t  # (T,d)

            # L6
            if G_inv_t.ndim == 3:
                v_t = torch.einsum(
                    "tj, tji, ti -> tj", u_t_i[1:], G_t[1:], u_t_i[1:]
                )  # (T-1, d)
            else:
                v_t = u_t_i[1:] * G_t[1:] * u_t_i[1:]  # (T-1, d)
            v_t = torch.autograd.grad(
                v_t.sum(),
                x_t_i,
                materialize_grads=True,
            )[0]

            # L7
            mu_t = self.get_mut_t(v_t, G_inv_t, diff)  # (T, d)

            # L8
            if G_inv_t.ndim == 3:
                u_t = -0.5 * torch.einsum("tij,tj->ti", G_inv_t, mu_t)  # (T, d)
            else:
                u_t = -0.5 * G_inv_t * mu_t  # (T, d)
            # L9/19
            alpha = self.line_search(x_0, x_T, u_t, u_t_i, x_t_i)
            # alpha = 0.1

            # L11/L12: update state without tracking to avoid graph growth
            with torch.no_grad():
                u_t_i = alpha * u_t + (1 - alpha) * u_t_i
                x_t_i = x_0 + torch.cumsum(u_t_i[:-1], dim=0)  # (T-1, d)
            # Re-enable grad only on trajectory points for next iteration
            x_t_i = x_t_i.detach().requires_grad_(True)

            # Prepare stop condition, ie L4
            grad_E_t = self.grad_E(x_t_i, x_0, x_T)
            norm_grad_E_t = torch.linalg.vector_norm(grad_E_t.reshape(-1))
            norm_grad_E_t = norm_grad_E_t / (
                grad_E_t.shape[0] * grad_E_t.shape[1]
            )  # Normalize by the dimension
            i += 1

            # Logging
            with torch.no_grad():
                dst = self.dst_func(x_0, x_T, x_t_i).item()
                E = self.compute_energy(x_t_i, x_0, x_T).item()
            dst_list.append(dst)
            norm_gE_list.append(norm_grad_E_t.item())
            E_list.append(E)
            alpha_list.append(alpha)

            if self.pbar:
                pbar.set_description(
                    f"{i=:0>3} |"
                    f" alpha: {alpha:.3E}, "
                    f"E = {E:.3E}, "
                    f"grad_E = {norm_grad_E_t.item():.3E}, "
                    f" dst = {dst:.3E}"
                )
                pbar.update(1)

        x_final = torch.cat([x_0[None, :], x_t_i, x_T[None, :]], dim=0)  # (T+1, d)
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
        x_t_0: Tensor (B,T-1, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_1.
            It must not contain x_0 and x_1.

        Returns:
        -------
        trajectories: Tensor (B, T+1, d)
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
        trajectories = torch.stack(trajectories, dim=0)  # (B, T+1, d)
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
        x_t_0: Tensor (B,T-1, d), optional
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
        self.tol = tol
        self.rho = rho
        self.c = c
        self.alpha_0 = alpha_0
        self.pbar = pbar

    def compute_distance(self, traj: torch.Tensor, tangent_vectors: torch.Tensor = None):
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

    def compute_energy(self, z_t, z0, zT, dx=None):
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
        dx: Tensor (T, d)
            The difference between consecutive points in the trajectory.
            If not provided, it will be computed as z_t[1:] - z_t[:-1].
        Returns:
        -------
        energy: Tensor
            The total energy of the geodesic trajectory.
        """
        traj = torch.cat([z0[None, :], z_t, zT[None, :]], dim=0)  # (T+1, d)
        if dx is None:
            dx = traj[1:] - traj[:-1]  # (T, d)
        G = self.finsler.fundamental_tensor(traj[:-1], dx)  # (T, d, d)
        energy = torch.einsum("ti,tij,tj->t", dx, G, dx)  # (T,)
        return energy.sum()

    def dot_sum(self, x_t, u_t):
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

    def dot_sum_u(self, x_t, u_t_0, u_t_1):
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

    def get_v_t(self, x_t_i, u_t_i):
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

    def get_zeta_t(self, x_t_i, u_t_i):
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

    def get_mut_t(self, v_t, zeta_t, G_inv_t, diff):
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

    def line_search(self, x_0, x_T, u_t, u_t_i, x_t_i):
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

        while condition and curr_iter < self.max_iter:
            alpha *= self.rho  # Reduce step size
            curr_iter += 1

            with torch.no_grad():
                x_t_new = x_0 + torch.cumsum(
                    alpha * u_t[:-1] + (1 - alpha) * u_t_i[:-1], dim=0
                )
                E_new = self.compute_energy(x_t_new, x_0, x_T)
            val = E_0 + self.c * alpha * torch.dot(grad_E_0, p_k)
            condition = E_new > val

        if curr_iter == self.max_iter:
            print(f"Warning: Maximum iterations reached in line search. {alpha=}")

        return alpha

    def dst_func(self, x_0, x_T, x_t) -> Tensor:
        """
        Compute the geodesic distance between x_0 and x_T along the trajectory x_t.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        x_t: Tensor (T-1, d)
            The geodesic trajectory points. It does not include x_0 and x_T.

        Returns:
        --------
        distance: Tensor
            The geodesic distance between x_0 and x_T along the trajectory x_t.
        """
        full_traj = torch.cat([x_0[None, :], x_t, x_T[None, :]], dim=0)  # (T+1, d)
        dx = full_traj[1:] - full_traj[:-1]  # (T, d)
        G = self.finsler.fundamental_tensor(full_traj[:-1], dx)  # (T, d, d)
        distance = torch.einsum("ti,tij,tj->t", dx, G, dx).abs().sqrt()  # (T,)
        return distance.sum()

    def georce_solver(
        self,
        x_0: Tensor,
        x_T: Tensor,
        x_t_0: Tensor = None,
    ):
        """
        Solve the geodesic equation using the GEORCE algorithm.

        Parameters:
        ----------
        x_0: Tensor (d,)
            The starting point of the geodesic.
        x_T: Tensor (d,)
            The ending point of the geodesic.
        x_t_0: Tensor (T-1, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_T.
            It must not contain x_0 and x_T.

        Returns:
        -------
        x_final: Tensor (T+1, d)
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

        if x_t_0 is None:
            t = torch.linspace(0, 1, self.T + 1, device=x_0.device, dtype=x_0.dtype)
            x_t_0 = x_0[None, :] + t[1:-1, None] * (x_T - x_0)[None, :]  # (T-1, d)
        else:
            assert x_t_0.shape == (
                self.T - 1,
                d,
            ), f"x_t_0 must have shape {(self.T - 1, d)=} got {x_t_0.shape=}. But sure to exclude x_0 and x_T."

        diff = x_T - x_0
        # (T-1, d), not including x_0 and x_T
        x_t_i = x_t_0.clone().requires_grad_(True)
        # (T, d) Initial guess of the velocity, not including x_T
        u_t_i = diff * torch.ones(self.T, d, device=x_0.device, dtype=x_0.dtype) / self.T

        # Precompute fundamental tensor at start without building graph
        with torch.no_grad():
            G_0 = self.finsler.fundamental_tensor(
                x_0[None, :], u_t_i[0, :].unsqueeze(0)
            ).squeeze(0)

        # L4
        grad_E_t = torch.autograd.grad(
            self.compute_energy(x_t_i, x_0, x_T), x_t_i, materialize_grads=True
        )[
            0
        ]  # (T-1, d)
        norm_grad_E_t = torch.linalg.vector_norm(grad_E_t.reshape(-1))
        norm_grad_E_t = norm_grad_E_t / (
            grad_E_t.shape[0] * grad_E_t.shape[1]
        )  # Normalize by the dimension

        with torch.no_grad():
            dst_list = [self.dst_func(x_0, x_T, x_t_i).item()]
        norm_gE_list = [norm_grad_E_t.item()]
        with torch.no_grad():
            E_list = [self.compute_energy(x_t_i, x_0, x_T, dx=u_t_i).item()]
        alpha_list = [1.0]

        if self.pbar:
            pbar = tqdm(range(self.max_iter), total=self.max_iter, desc="Iterations")
        # for i in pbar:
        while (norm_grad_E_t > self.tol) & (i < self.max_iter):
            # L5
            G_t = self.finsler.fundamental_tensor(x_t_i, u_t_i[1:])  # (T-1, d, d)
            G_t = torch.cat([G_0[None, :], G_t], dim=0)  # (T, d, d)
            G_inv_t = G_t.inverse()  # (T, d, d)

            # L6/7
            v_t = self.get_v_t(x_t_i, u_t_i[1:])  # (T-1, d)
            zeta_t = self.get_zeta_t(x_t_i, u_t_i[1:])  # (T-1, d)

            # L8
            mu_t = self.get_mut_t(v_t, zeta_t, G_inv_t, diff)

            # L9
            u_t = -0.5 * torch.einsum("tij,tj->ti", G_inv_t, mu_t)  # (T, d)

            # L10/11
            alpha = self.line_search(x_0, x_T, u_t, u_t_i, x_t_i)
            # alpha = 0.00001

            # L12/L13: update state without tracking to avoid graph growth
            with torch.no_grad():
                u_t_i = alpha * u_t + (1.0 - alpha) * u_t_i
                x_t_i = x_0 + torch.cumsum(u_t_i[:-1], dim=0)  # (T-1, d)
            # Re-enable grad only on trajectory points for next iteration
            x_t_i = x_t_i.detach().requires_grad_(True)

            # Prepare stop condition, ie L4
            grad_E_t = torch.autograd.grad(
                self.compute_energy(x_t_i, x_0, x_T), x_t_i, materialize_grads=True
            )[0]
            # (T-1, d)
            norm_grad_E_t = torch.linalg.vector_norm(grad_E_t.reshape(-1))
            norm_grad_E_t = norm_grad_E_t / (grad_E_t.shape[0] * grad_E_t.shape[1])
            i += 1

            # Logging
            with torch.no_grad():
                dst = self.dst_func(x_0, x_T, x_t_i).item()
                E = self.compute_energy(x_t_i, x_0, x_T).item()
            dst_list.append(dst)
            norm_gE_list.append(norm_grad_E_t.item())
            E_list.append(E)
            alpha_list.append(alpha)

            if self.pbar:
                pbar.set_description(
                    f"{i=:0>3} |"
                    f" alpha: {alpha:.3E}, "
                    f"E = {E:.3E}, "
                    f"grad_E = {norm_grad_E_t.item():.3E}, "
                    f" dst = {dst:.3E}"
                )
                pbar.update(1)

        if norm_grad_E_t.isnan():
            print(
                "Warning: Gradient of the energy is NaN. Stopping optimization. Return straight line."
            )
            t = torch.linspace(0, 1, self.T + 1, device=x_0.device, dtype=x_0.dtype)
            x_t_i = x_0[None, :] + t[1:-1, None] * (x_T - x_0)[None, :]  # (T-1, d)

        x_final = torch.cat([x_0[None, :], x_t_i, x_T[None, :]], dim=0)  # (T+1, d)
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
        x_t_0: Tensor (B,T-1, d), optional
            The initial guess of the geodesic trajectory points. If None, it will be computed
            as a linear interpolation between x_0 and x_1.
            It must not contain x_0 and x_1.

        Returns:
        -------
        trajectories: Tensor (B, T+1, d)
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
        trajectories = torch.stack(trajectories, dim=0)  # (B, T+1, d)
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
        x_t_0: Tensor (B,T-1, d), optional
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
        max_iter=100,
        tol=1e-10,
        rho=0.5,
        c=0.9,
        alpha_0: float = 1.0,
        pbar_georce: bool = False,
        max_data_count: None | int = None,
        smooth_curve: bool = True,
    ):
        super().__init__(cometric=cometric)
        dt = 1.0 / T
        self.graph_solver = SolverGraph(
            cometric=cometric,
            data=data,
            n_neighbors=n_neighbors,
            dt=dt,
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
            q0, q1, pts_on_traj_graph[:, 1:-2, :].clone().detach()
        )
        final_traj = torch.zeros_like(pts_on_traj_georce)
        # Return the trajectory with the smallest distance
        # This prevent the case where GEORCE didn't converge
        B = q0.shape[0]
        for b in range(B):
            dst_graph = self.compute_distance(pts_on_traj_graph[b].unsqueeze(0))
            dst_georce = self.compute_distance(pts_on_traj_georce[b].unsqueeze(0))
            if dst_graph < dst_georce:
                final_traj[b] = pts_on_traj_graph[b, :-1]
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
        max_iter=100,
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
        dt = 1.0 / T
        self.graph_solver = SolverGraphFinsler(
            finsler_metric=finsler,
            data=data,
            n_neighbors=n_neighbors,
            dt=dt,
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
            q0, q1, pts_on_traj_graph[:, 1:-2, :].detach()
        )
        final_traj = torch.zeros_like(pts_on_traj_georce)
        # Return the trajectory with the smallest distance
        # This prevent the case where GEORCE didn't converge
        B = q0.shape[0]
        for b in range(B):
            traj_graph = pts_on_traj_graph[b, :-1, :]
            traj_georce = pts_on_traj_georce[b]
            dst_graph_ = self.dst_func(q0[b], q1[b], traj_graph[1:-1])
            dst_georce_ = self.dst_func(q0[b], q1[b], traj_georce[1:-1])
            if dst_graph_ < dst_georce_:
                final_traj[b] = traj_graph.detach().clone()
            else:
                final_traj[b] = pts_on_traj_georce[b].detach().clone()
        return final_traj


class ExpMapFinsler(torch.nn.Module):
    """
    Computes the geodesic distances between points using the shooting method
    on a Finsler metric.

    Parameters:
    ----------
    finsler : FinslerMetric
        The Finsler metric to use for the geodesic distances.
    T : int
        The number of time steps to use for the geodesic trajectory.
    T_max : float
        The maximum time to use for the geodesic trajectory.
    method : str
        The method to use for the ODE solver. See torchdiffeq documentation for more details.
    boundary : float
        The boundary value to stop the ODE solver when the state diverges.
    resample : bool
        If True, the trajectory is resampled to have T points even if the ODE solver stops early.
    """

    def __init__(
        self,
        finsler: FinslerMetric,
        T_max=1.0,
        T=100,
        method="rk4",
        boundary: float = 1e6,
        resample: bool = True,
    ):
        super().__init__()
        self.finsler = finsler
        self.T = T
        self.T_max = T_max
        self.t = torch.linspace(0, T_max, T)
        self.method = method
        self.boundary = boundary
        self.resample = resample

    def get_dg_dq(self, q, p):
        """
        Computes the derivative of the fundamental tensor with respect to the position q

        Parameters
        ----------
        q : torch.Tensor (d,)
            Position
        p : torch.Tensor (d,)
            Velocity

        Returns
        -------
        dg_dq : torch.Tensor (d,d,d)
            Derivative of the fundamental tensor with respect to the position q
        """

        def g_no_b(q):
            return self.finsler.fundamental_tensor(q.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

        dg_dq = torch.autograd.functional.jacobian(g_no_b, q)
        return dg_dq

    def formal_christoffel_symbols(self, q, p):
        """
        Computes the formal christoffel symbols at position q and velocity p

        Parameters
        ----------
        q : torch.Tensor (d,)
            Position
        p : torch.Tensor (d,)
            Velocity

        Returns
        -------
        gamma : torch.Tensor (d,d,d)
            Christoffel symbols
        """
        g_inv = self.finsler.inverse_fundamental_tensor(
            q.unsqueeze(0), p.unsqueeze(0)
        ).squeeze(0)
        dg_dq = self.get_dg_dq(q, p)

        gamma = (
            torch.einsum("il,jlk->ijk", g_inv, dg_dq)
            + torch.einsum("il,kjl->ijk", g_inv, dg_dq)
            - torch.einsum("il,ljk->ijk", g_inv, dg_dq)
        )

        return 1 / 2 * gamma

    def geodesic_ode(self, t, state):
        """
        Geodesic ODE for the Randers metric

        Parameters
        ----------
        t : float
            Time, unused. For compatibility with torchdiffeq
        state : torch.Tensor (2d,)
            State (position and velocity)

        Returns
        -------
        dstate_dt : torch.Tensor (2d,)
            Derivative of the state with respect to time
        """
        d = state.shape[0] // 2
        q = state[:d]
        v = state[d:]

        gamma = self.formal_christoffel_symbols(q, v)

        v_dot = -torch.einsum("ijk,j,k->i", gamma, v, v)

        dstate_dt = torch.cat((v, v_dot))
        return dstate_dt

    def resample_state(self, state):
        """
        Resample the trajectory to have self.T points using cubic splines.

        Parameters
        ----------
        state : torch.Tensor (T',2d)
            State (position and momentum), T' <= T if the out of boundary event is triggered
        Returns
        -------
        state : torch.Tensor (T,2d)
            Resampled state (position and momentum)
        """
        x = state[:, : state.shape[1] // 2]
        v = state[:, state.shape[1] // 2 :]

        cs_x = CubicSpline(
            torch.linspace(0, self.T_max, x.shape[0]).cpu(),
            x.cpu(),
            axis=0,
        )
        new_x = torch.tensor(cs_x(self.t.cpu()), device=x.device)

        cs_v = CubicSpline(
            torch.linspace(0, self.T_max, v.shape[0]).cpu(),
            v.cpu(),
            axis=0,
        )
        new_v = torch.tensor(cs_v(self.t.cpu()), device=v.device)

        return torch.cat((new_x, new_v), dim=-1)

    def stop_when_diverge(self, state):
        """
        Filter the trajectory to keep only the points where the norm of the position is below the boundary.
        Stops the trajectory when the boundary is exceeded.

        Parameters
        ----------
        state : torch.Tensor (2d,)
            State (position and momentum)

        Returns
        -------
        state : torch.Tensor (T',2d)
            Filtered state (position and momentum), T' <= T if the out of boundary event is triggered
        """
        q = state[: state.shape[0] // 2]
        q_norm = torch.linalg.vector_norm(q, dim=-1)
        dq = q_norm - self.boundary
        outside_pts = torch.where(dq > 0)[0]
        if len(outside_pts) > 0:
            first_outside = outside_pts[0]
            state = state[:first_outside]

            if self.resample:
                state = self.resample_state(state)
        return state

    def geodesic_shooting(self, q0, v0):
        """
        Computes the geodesic trajectory starting from position q0 and initial velocity v0

        Parameters
        ----------
        q0 : torch.Tensor (d,)
            Initial position
        v0 : torch.Tensor (d,)
            Initial velocity

        Returns
        -------
        x_t : torch.Tensor (T',d)
            Geodesic trajectory, T' <= T if the out of boundary event is triggered
        v_t : torch.Tensor (T',d)
            Velocity along the geodesic trajectory, T' <= T if the out of boundary event is triggered
        """
        state_0 = torch.cat((q0, v0))
        state_T = odeint(
            func=self.geodesic_ode,
            y0=state_0,
            t=self.t.to(q0.device),
            method=self.method,
        )
        state_T = self.stop_when_diverge(state_T)
        x_t = state_T[:, : q0.shape[0]]
        v_t = state_T[:, q0.shape[0] :]
        return x_t, v_t

    def compute_distance(self, traj: torch.Tensor, tangent_vectors: torch.Tensor = None):
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

    def forward(self, x_0: Tensor, v_0: Tensor) -> Tensor:
        """Given the start and initial velocity points, compute the geodesic path
        by computing the exp map.

        Parameters:
        ----------
        x_0: Tensor (B, d)
            The starting points of the geodesic.
        v_0: Tensor (B, d)
            The initial velocities of the geodesic.

        Returns:
        -------
        trajectories: Tensor (B, T, d) or list of Tensors of shape (B, T_i, d)
            The geodesic trajectories starting at x_0 with initial velocity v_0
        """
        trajectories = []
        B, d = x_0.shape
        assert v_0.shape == (B, d), f"Both tensors must have the same shape {(B, d)=}"
        for b in range(B):
            x_t, v_t = self.geodesic_shooting(x_0[b], v_0[b])
            trajectories.append(x_t)
        if len(set([traj.shape[0] for traj in trajectories])) == 1:
            trajectories = torch.stack(trajectories, dim=0)  # (B, T, d)
        return trajectories


class ExpMapRiemann(torch.nn.Module):
    """
    Computes the geodesic distances between points using the shooting method
    on a Riemannian manifold. Here the geodesic is computed by
    integrating Hamilton's equations.

    Parameters:
    ----------
    cometric : CoMetric
        The Riemannian cometric to use for the geodesic distances.
    T : int
        The number of time steps to use for the geodesic trajectory.
    T_max : float
        The maximum time to use for the geodesic trajectory.
    method : str
        The method to use for the ODE solver. See torchdiffeq documentation for more details.
    boundary : float
        The boundary value to stop the ODE solver when the state diverges.
    resample : bool
        If True, the trajectory is resampled to have T points even if the ODE solver stops early.
    """

    def __init__(
        self,
        cometric: CoMetric,
        T_max=1.0,
        T=100,
        method="dopri5",
        boundary=1e5,
        resample=True,
    ):
        super().__init__()
        self.cometric = cometric
        self.T = T
        self.T_max = T_max
        self.t = torch.linspace(0, T_max, T)
        self.method = method
        self.boundary = boundary
        self.resample = resample

    def hamiltonian(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Hamiltonian function for the Riemannian manifold.

        Parameters
        ----------
        q : torch.Tensor (d,)
            Position
        p : torch.Tensor (d,)
            Momentum

        Returns
        -------
        H : torch.Tensor
            Hamiltonian value
        """
        H = 0.5 * self.cometric.cometric(q.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        return H

    def get_dH(self, q: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the derivatives of the Hamiltonian with respect to position and momentum.

        Parameters
        ----------
        q : torch.Tensor (d,)
            Position
        p : torch.Tensor (d,)
            Momentum

        Returns
        -------
        dH_dq : torch.Tensor (d,)
            Derivative of the Hamiltonian with respect to position
        dH_dp : torch.Tensor (d,)
            Derivative of the Hamiltonian with respect to momentum
        """
        H = self.hamiltonian(q, p)
        dH_dq, dH_qp = torch.autograd.grad(H, (q, p), create_graph=True)
        return dH_dq, dH_qp

    def geodesic_ode(self, t: float, state: Tensor) -> Tensor:
        """
        Computes the geodesic ODE using Hamilton's equations.

        Parameters
        ----------
        t : float
            Time, unused. For compatibility with torchdiffeq
        state : torch.Tensor (2d,)
            State (position and momentum)

        Returns
        -------
        dstate_dt : torch.Tensor (2d,)
            Derivative of the state with respect to time
        """
        d = state.shape[0] // 2
        q, p = state[:d], state[d:]
        dH_dq, dH_qp = self.get_dH(q, p)
        return torch.cat((dH_qp, -dH_dq))

    def resample_state(self, state):
        """
        Resample the trajectory to have self.T points using cubic splines.

        Parameters
        ----------
        state : torch.Tensor (T',2d)
            State (position and momentum), T' <= T if the out of boundary event is triggered
        Returns
        -------
        state : torch.Tensor (T,2d)
            Resampled state (position and momentum)
        """
        x = state[:, : state.shape[1] // 2]
        v = state[:, state.shape[1] // 2 :]

        cs_x = CubicSpline(
            torch.linspace(0, self.T_max, x.shape[0]).cpu(),
            x.cpu(),
            axis=0,
        )
        new_x = torch.tensor(cs_x(self.t.cpu()), device=x.device)

        cs_v = CubicSpline(
            torch.linspace(0, self.T_max, v.shape[0]).cpu(),
            v.cpu(),
            axis=0,
        )
        new_v = torch.tensor(cs_v(self.t.cpu()), device=v.device)

        return torch.cat((new_x, new_v), dim=-1)

    def stop_when_diverge(self, state):
        """
        Filter the trajectory to keep only the points where the norm of the position is below the boundary.
        Stops the trajectory when the boundary is exceeded.

        Parameters
        ----------
        state : torch.Tensor (2d,)
            State (position and momentum)

        Returns
        -------
        state : torch.Tensor (T',2d)
            Filtered state (position and momentum), T' <= T if the out of boundary event is triggered
        """
        q = state[: state.shape[0] // 2]
        q_norm = torch.linalg.vector_norm(q, dim=-1)
        dq = q_norm - self.boundary
        outside_pts = torch.where(dq > 0)[0]
        if len(outside_pts) > 0:
            first_outside = outside_pts[0]
            state = state[:first_outside]
            if self.resample:
                state = self.resample_state(state)
        return state

    def geodesic_shooting(self, q0: Tensor, p0: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the geodesic trajectory starting from position q0 and initial momentum p0

        Parameters
        ----------
        q0 : torch.Tensor (d,)
            Initial position
        p0 : torch.Tensor (d,)
            Initial momentum

        Returns
        -------
        q_t : torch.Tensor (T',d)
            Geodesic trajectory, T' <= T if the event is triggered
        p_t : torch.Tensor (T',d)
            Momentum along the geodesic trajectory, T' <= T if the event is triggered
        """
        state0 = torch.cat((q0, p0))
        state_T = odeint(
            func=self.geodesic_ode,
            y0=state0,
            t=self.t.to(q0.device),
            method=self.method,
        )
        state_T = self.stop_when_diverge(state_T)
        q_t = state_T[:, : q0.shape[0]]
        p_t = state_T[:, q0.shape[0] :]
        return q_t, p_t

    def compute_distance(self, traj: torch.Tensor, tangent_vectors: torch.Tensor = None):
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

    def forward(self, x_0: Tensor, v_0: Tensor, convert_to_moment: bool = False) -> Tensor:
        """Given the start and initial velocity points, compute the geodesic path
        by computing the exp map.

        Parameters:
        ----------
        x_0: Tensor (B, d)
            The starting points of the geodesic.
        v_0: Tensor (B, d)
            The initial velocities of the geodesic.
        convert_to_moment: bool
            If True, convert the initial velocity to momentum using the cometric.

        Returns:
        -------
        trajectories: Tensor (B, T, d) or list of Tensor or shape (B, T_i, d)
            The geodesic trajectories starting at x_0 with initial velocity v_0
        """
        if convert_to_moment:
            G_inv = self.cometric.cometric_tensor(x_0)
            if self.cometric.is_diag:
                v_0 = G_inv * v_0
            else:
                v_0 = torch.einsum("bij,bj->bi", G_inv, v_0)

        trajectories = []
        B, d = x_0.shape
        assert v_0.shape == (B, d), f"Both tensors must have the same shape {(B, d)=}"
        for b in range(B):
            x_t, v_t = self.geodesic_shooting(x_0[b], v_0[b])
            trajectories.append(x_t)
        if len(set([traj.shape[0] for traj in trajectories])) == 1:
            trajectories = torch.stack(trajectories, dim=0)  # (B, T, d)
        return trajectories
