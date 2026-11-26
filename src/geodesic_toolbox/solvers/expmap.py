"""
Exponential map solvers (Randers, Riemann, Finsler).
"""

import torch
from scipy.interpolate import CubicSpline
from torch import Tensor
from torchdiffeq import odeint

from ..cometric import CoMetric
from ..finsler import DualRandersMetrics, FinslerMetric, RandersMetrics
from ..integrators import (
    ExplicitLeapfrogIntegrator,
    Hamiltonian,
    HamiltonianEulerIntegrator,
    ImplicitLeapfrogIntegrator,
)


class HamiltonianRanders(Hamiltonian):
    def __init__(self, randers: RandersMetrics):
        super().__init__()
        self.randers = randers
        self.dual_r = DualRandersMetrics(randers)

    def H(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Hamiltonian function for the Randers manifold.

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        p : torch.Tensor (b, d)
            Momentum

        Returns
        -------
        H : torch.Tensor (b,)
            Hamiltonian value
        """
        H = self.dual_r(q, p)
        return 0.5 * H**2

    def forward(self, q, p):
        return self.H(q, p)


class HamiltonianRiemann(Hamiltonian):
    def __init__(self, cometric: CoMetric):
        super().__init__()
        self.cometric = cometric

    def H(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Hamiltonian function for the Riemannian manifold.

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        p : torch.Tensor (b, d)
            Momentum

        Returns
        -------
        H : torch.Tensor (b,)
            Hamiltonian value
        """
        H = self.cometric.dual_energy(q, p)
        return 0.5 * H

    def forward(self, q, p):
        return self.H(q, p)


class ExpMapRanders(torch.nn.Module):
    """
    Exponential map for a Randers metric.
    Shoot a geodesic trajectory on a Randers metric using the shooting method.
    We use the hamiltonian formulation via the dual Randers metric on the cotangent space.

    Parameters:
    -----------
    randers : RandersMetric
        The base Randers metric to compute the geodesics
    T_max : float
        The maximum time to use for the geodesic trajectory.
    T: int
        The number of time points to compute along the trajectory. Step size of the integrator is computed as T_max / T.
    solver : str
        The solver to use for the shooting method. Must be one of 'leapfrog_explicit', 'leapfrog_implicit' or 'euler'. Default is 'leapfrog_implicit'.
    substep : int
        The number of substeps to use for the integrator. Default is 1 (no substeps).
    omega : float
        Coupling parameter for the explcit leapfrog integrator. Only used if solver is 'leapfrog_explicit'. Default is 10.0.
    n_fix_pts : int
        The number of fixed point iterations to use for the implicit leapfrog integrator. Only used if solver is 'leapfrog_implicit'. Default is 5.
    """

    def __init__(
        self,
        randers: RandersMetrics,
        T_max=1.0,
        T=10,
        solver: str = "leapfrog_implicit",
        substeps: int = 1,
        omega: float = 10.0,
        n_fix_pts: int = 5,
    ):
        super().__init__()
        self.randers = randers

        self.T = T
        self.T_max = T_max
        self.t = torch.linspace(0, T_max, T)
        self.dt = self.t[1] - self.t[0]

        self.substeps = substeps
        self.solver = solver
        self.omega = omega
        self.n_fix_pts = n_fix_pts

        self.H = HamiltonianRanders(randers)
        self.dual_r = self.H.dual_r

        if self.solver == "leapfrog_explicit":
            self.shooter = ExplicitLeapfrogIntegrator(
                H=self.H,
                gamma=self.dt,
                omega=self.omega,
                substeps=self.substeps,
            )
        elif self.solver == "leapfrog_implicit":
            self.shooter = ImplicitLeapfrogIntegrator(
                H=self.H,
                gamma=self.dt,
                substeps=self.substeps,
                n_fix_pts=self.n_fix_pts,
            )
        elif self.solver == "euler":
            self.shooter = HamiltonianEulerIntegrator(
                H=self.H,
                gamma=self.dt,
                substeps=self.substeps,
            )
        else:
            raise ValueError(
                f"Unknown solver {self.solver=}. Must be one of 'leapfrog_explicit', 'leapfrog_implicit' or 'euler'."
            )

    def legendre_transform(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Convert a speed vector to a momentum vector using the Legendre transform.
        Formula given by Lemma 1.1 (iii) of 10.48550/arXiv.0808.1166 Heat flow on Finsler manifolds

        p = g(x,v) v

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        v : torch.Tensor (b, d)
            Speed

        Returns
        -------
        p : torch.Tensor (b, d)
            Momentum
        """
        g = self.randers.fundamental_tensor(q, v)
        p = torch.einsum("bij,bj->bi", g, v)
        return p

    def inverse_legendre_transform(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Convert a momentum vector to a speed vector using the inverse Legendre transform.
        Formula given by Lemma 1.1 (iii) of 10.48550/arXiv.0808.1166 Heat flow on Finsler manifolds

        v =g^*(x,p) p

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        p : torch.Tensor (b, d)
            Momentum

        Returns
        -------
        v : torch.Tensor (b, d)
            Speed
        """
        g_inv = self.dual_r.fundamental_tensor(q, p)
        v = torch.einsum("bij,bj->bi", g_inv, p)
        return v

    def forward(self, q0: Tensor, v0: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the geodesic trajectory starting from position q0 and initial speed v0

        Parameters
        ----------
        q0 : torch.Tensor (b, d)
            Initial position
        v0 : torch.Tensor (b, d)
            Initial speed

        Returns
        -------
        q_t : torch.Tensor (b,T,d)
            Geodesic trajectory
        p_t : torch.Tensor (b,T,d)
            Momentum along the geodesic trajectory
        """
        p0 = self.legendre_transform(q0, v0)
        traj_q, traj_p, _ = self.shooter.forward(q0, p0, self.T, return_traj=True)  # (b,T,2d)
        return traj_q, traj_p


class ExpMapRiemann(torch.nn.Module):
    """
    Exponential map for a Riemannian metric.
    Shoot a geodesic trajectory on a Riemannain metric using the shooting method.
    We use the hamiltonian formulation.


    Parameters:
    -----------
    cometric : CoMetric
        The cometric to compute the geodesics
    T_max : float
        The maximum time to use for the geodesic trajectory.
    T: int
        The number of time points to compute along the trajectory. Step size of the integrator is computed as T_max / T.
    solver : str
        The solver to use for the shooting method. Must be one of 'leapfrog_explicit', 'leapfrog_implicit' or 'euler'. Default is 'leapfrog_implicit'.
    substep : int
        The number of substeps to use for the integrator. Default is 1 (no substeps).
    omega : float
        Coupling parameter for the explcit leapfrog integrator. Only used if solver is 'leapfrog_explicit'. Default is 10.0.
    n_fix_pts : int
        The number of fixed point iterations to use for the implicit leapfrog integrator. Only used if solver is 'leapfrog_implicit'. Default is 5.
    """

    def __init__(
        self,
        cometric: CoMetric,
        T_max=1.0,
        T=10,
        solver: str = "leapfrog_implicit",
        substeps: int = 1,
        omega: float = 10.0,
        n_fix_pts: int = 5,
    ):
        super().__init__()
        self.cometric = cometric

        self.T = T
        self.T_max = T_max
        self.t = torch.linspace(0, T_max, T)
        self.dt = self.t[1] - self.t[0]

        self.substeps = substeps
        self.solver = solver
        self.omega = omega
        self.n_fix_pts = n_fix_pts

        self.H = HamiltonianRiemann(cometric)

        if self.solver == "leapfrog_explicit":
            self.shooter = ExplicitLeapfrogIntegrator(
                H=self.H,
                gamma=self.dt,
                omega=self.omega,
                substeps=self.substeps,
            )
        elif self.solver == "leapfrog_implicit":
            self.shooter = ImplicitLeapfrogIntegrator(
                H=self.H,
                gamma=self.dt,
                substeps=self.substeps,
                n_fix_pts=self.n_fix_pts,
            )
        elif self.solver == "euler":
            self.shooter = HamiltonianEulerIntegrator(
                H=self.H,
                gamma=self.dt,
                substeps=self.substeps,
            )
        else:
            raise ValueError(
                f"Unknown solver {self.solver=}. Must be one of 'leapfrog_explicit', 'leapfrog_implicit' or 'euler'."
            )

    def legendre_transform(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Convert a speed vector to a momentum vector using the Legendre transform.

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        v : torch.Tensor (b, d)
            Speed

        Returns
        -------
        p : torch.Tensor (b, d)
            Momentum
        """
        G = self.cometric.metric_tensor(q)
        if self.cometric.is_diag:
            Gv = G * v
        else:
            Gv = torch.einsum("bij,bj->bi", G, v)
        return Gv

    def inverse_legendre_transform(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Convert a momentum vector to a speed vector using the inverse Legendre transform.

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        p : torch.Tensor (b, d)
            Momentum

        Returns
        -------
        v : torch.Tensor (b, d)
            Speed
        """
        G_inv = self.cometric.cometric_tensor(q)
        if self.cometric.is_diag:
            G_inv_p = G_inv * p
        else:
            G_inv_p = torch.einsum("bij,bj->bi", G_inv, p)
        return G_inv_p

    def forward(self, q0: Tensor, v0: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the geodesic trajectory starting from position q0 and initial speed v0

        Parameters
        ----------
        q0 : torch.Tensor (b, d)
            Initial position
        v0 : torch.Tensor (b, d)
            Initial speed

        Returns
        -------
        q_t : torch.Tensor (b,T,d)
            Geodesic trajectory
        p_t : torch.Tensor (b,T,d)
            Momentum along the geodesic trajectory
        """
        p0 = self.legendre_transform(q0, v0)
        traj_q, traj_p, _ = self.shooter.forward(q0, p0, self.T, return_traj=True)  # (b,T,2d)
        return traj_q, traj_p


class ExpMapFinsler(torch.nn.Module):
    """
    Exponential map for a Finsler metric.
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
            x_t, _v_t = self.geodesic_shooting(x_0[b], v_0[b])
            trajectories.append(x_t)
        if len({traj.shape[0] for traj in trajectories}) == 1:
            trajectories = torch.stack(trajectories, dim=0)  # (B, T, d)
        return trajectories
