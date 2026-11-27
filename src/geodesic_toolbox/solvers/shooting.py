"""
Shooting solver for geodesic distances.
"""

from collections.abc import Callable
from math import ceil, cos

import torch
from torch import Tensor

from ..cometric import CoMetric, IdentityCoMetric
from ..utils import magnification_factor, scale_lr_magnification
from .base import GeodesicDistanceSolver


def constant_time_scaling_schedule(x: float) -> float:
    """
    Constant time scaling schedule.
    Results in ceil(1/dt)*(n_step + 1) integration steps.

    Parameters:
    ----------
    x : float
        Current iteration progress (0 to 1)

    Returns:
    -------
    scaling : float
        Scaling factor
    """
    return 1


def linear_time_scaling_schedule(x: float, max_scaling: float = 5) -> float:
    """Linear ramp from max_scaling to 1.
    Results in ()*(n_step + 1) integration steps.

    Parameters:
    ----------
    x : float
        Current iteration progress (0 to 1)
    max_scaling : float
        Maximum scaling factor

    Returns:
    -------
    scaling : float
        Scaling factor
    """
    return 1 + (max_scaling - 1) * (1 - x)


def cosine_time_scaling_schedule(x: float, max_scaling: float = 5) -> float:
    """Cosine decay from max_scaling to 1

    Parameters:
    ----------
    x : float
        Current iteration progress (0 to 1)
    max_scaling : float
        Maximum scaling factor

    Returns:
    -------
    scaling : float
        Scaling factor
    """
    return 1 + (max_scaling - 1) * (1 + cos(x * 3.1415)) / 2


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
    time_scaling_schedule : str, name of the time scaling schedule to use. Choose one of ["constant", "linear", "cosine"].
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
        convergence_threshold: float | None = None,
        time_scaling_schedule: str = "cosine",
        scale_lr: bool = True,
        verbose: bool = False,
    ) -> None:
        SCHEDULES = {
            "constant": constant_time_scaling_schedule,
            "linear": linear_time_scaling_schedule,
            "cosine": cosine_time_scaling_schedule,
        }

        super().__init__(cometric=cometric)
        self.dt = dt
        self.final_dt = dt
        self.n_pts = ceil(1 / self.dt)
        self.lr = lr
        self.n_step = n_step
        assert time_scaling_schedule in SCHEDULES, (
            f"Unknown time scaling schedule {time_scaling_schedule}. Choose one of {list(SCHEDULES.keys())}"
        )
        self.time_scaling_schedule = SCHEDULES[time_scaling_schedule]

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
        return self.cometric.dual_energy(q, p)

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
