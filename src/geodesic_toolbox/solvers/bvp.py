"""
Boundary value problem solvers for geodesics.
"""

from collections.abc import Callable

import numpy as np
import torch
from scipy.integrate import solve_bvp
from torch import Tensor

from ..cometric import CoMetric
from ..utils import batched_kro, vec
from .base import GeodesicDistanceSolver


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
        super().__init__(cometric)
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
