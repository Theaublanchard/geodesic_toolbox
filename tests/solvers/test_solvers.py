import torch

from geodesic_toolbox import cometric, finsler, solvers
from geodesic_toolbox.solvers.shooting import (
    constant_time_scaling_schedule,
    cosine_time_scaling_schedule,
    linear_time_scaling_schedule,
)


class ConstantOneForm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = 0.1 * torch.ones(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w.repeat(x.shape[0], 1)


def test_compute_distance_matches_straight_segments():
    solver = solvers.GeodesicDistanceSolver(cometric.IdentityCoMetric())
    traj = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])

    distance = solver.compute_distance(traj)
    assert torch.allclose(distance, torch.tensor([2.0]))


def test_hamiltonian_gradients_for_identity_metric():
    solver = solvers.ShootingSolver(cometric.IdentityCoMetric())
    p = torch.tensor([[1.0, -1.0]], requires_grad=True)
    q = torch.tensor([[0.0, 0.0]], requires_grad=True)

    dH_dp, dH_dq = solver.get_dp_dq(p, q)
    assert torch.allclose(dH_dp, 2 * p)
    assert torch.allclose(dH_dq, torch.zeros_like(q))


def test_euler_and_leapfrog_steps_constant_metric():
    solver = solvers.ShootingSolver(cometric.IdentityCoMetric(), dt=0.25)
    p = torch.tensor([[0.5, -0.5]], requires_grad=True)
    q = torch.tensor([[1.0, 1.0]], requires_grad=True)

    p_euler, q_euler = solver.euler_step(solver.H, p, q)
    assert torch.allclose(p_euler, p)
    assert torch.allclose(q_euler, q + 2 * solver.dt * p)

    p_lf, q_lf = solver.leapfrog_step(solver.H, p, q)
    assert torch.allclose(p_lf, p)
    assert torch.allclose(q_lf, q + 2 * solver.dt * p)


def test_shooting_returns_expected_endpoint():
    solver = solvers.ShootingSolver(cometric.IdentityCoMetric(), dt=0.25)
    p0 = torch.tensor([[0.5, 1.0]], requires_grad=True)
    q0 = torch.tensor([[0.0, 0.0]], requires_grad=True)

    final_q = solver.shooting(p0, q0)
    expected_q = q0 + 2 * p0
    assert torch.allclose(final_q, expected_q, atol=1e-4)


def test_time_scaling_schedules_bounds():
    # Time scaling schedules live in solvers.shooting (moved out of utils).
    assert constant_time_scaling_schedule(0.0) == 1
    assert linear_time_scaling_schedule(0.0) > linear_time_scaling_schedule(1.0)
    assert cosine_time_scaling_schedule(0.0) > cosine_time_scaling_schedule(1.0)


def test_georce_stats_classes_are_dataclasses():
    line_search = solvers.GEORCELineSearchStats(
        initial_energy=1.0, final_energy=0.5, num_iterations=3, accepted=True
    )
    iteration = solvers.GEORCEIterationStats(
        iteration=0,
        grad_norm=0.1,
        distance=1.0,
        energy=1.0,
        alpha=0.5,
        line_search=line_search,
    )
    run = solvers.GEORCERunStats(
        initial_distance=1.0,
        initial_energy=1.0,
        final_distance=0.5,
        final_energy=0.5,
        final_grad_norm=0.1,
        converged=True,
    )
    run.iterations.append(iteration)
    # default_factory must give each new instance its own list.
    fresh = solvers.GEORCERunStats(
        initial_distance=1.0,
        initial_energy=1.0,
        final_distance=0.5,
        final_energy=0.5,
        final_grad_norm=0.1,
        converged=True,
    )
    assert fresh.iterations == []
    assert fresh.line_search_warnings == 0
    assert fresh.n_iter == 0
    assert run.iterations[0].line_search is line_search


def test_georce_run_produces_stats():
    solver = solvers.GEORCE(cometric.IdentityCoMetric(), T=20, max_iter=50)
    dist = solver(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(dist, torch.tensor([2**0.5]), atol=1e-3)
    assert isinstance(solver.georce_stats, solvers.GEORCERunStats)
    assert solver.georce_stats.converged


def test_expmap_riemann_accepts_euler_integrator():
    solver = solvers.ExpMapRiemann(cometric.IdentityCoMetric(), T=5, solver="euler")

    traj_q, traj_p = solver(torch.zeros(1, 2), torch.tensor([[0.5, 0.5]]))

    assert traj_q.shape == (1, 5, 2)
    assert traj_p.shape == (1, 5, 2)
    assert torch.isfinite(traj_q).all()


def test_expmap_randers_accepts_euler_integrator():
    randers = finsler.RandersMetrics(
        base_cometric=cometric.IdentityCoMetric(),
        omega=ConstantOneForm(dim=2),
        beta=0.5,
    )
    solver = solvers.ExpMapRanders(randers, T=5, solver="euler")

    traj_q, traj_p = solver(torch.zeros(1, 2), torch.tensor([[0.5, 0.5]]))

    assert traj_q.shape == (1, 5, 2)
    assert traj_p.shape == (1, 5, 2)
    assert torch.isfinite(traj_q).all()


class DeterministicShootingSolver(solvers.ShootingSolver):
    def optimize_initial_velocity(self, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        return 0.5 * (q1 - q0)


def test_forward_distance_matches_euclidean_for_identity():
    solver = DeterministicShootingSolver(cometric.IdentityCoMetric(), dt=0.2, n_step=5)
    q0 = torch.tensor([[0.0, 0.0]])
    q1 = torch.tensor([[1.0, 1.0]])

    dist = solver(q0, q1)
    assert torch.allclose(dist, torch.norm(q1 - q0, dim=1), atol=1e-2)


def test_georce_constant_metric_with_non_optimal_initial_guess():
    # With a position-independent (constant) metric such as IdentityCoMetric, the
    # GEORCE "force" term v_t = d/dx (u^T G(x) u) is exactly zero, because G(x)
    # does not depend on x. The optimizer loop computes this via
    # torch.autograd.grad(v_t.sum(), x_t_i, ...). Once u_t_i is updated under
    # torch.no_grad(), v_t no longer requires grad at all, and torch.autograd.grad
    # raised "element 0 of tensors does not require grad and does not have a
    # grad_fn" (materialize_grads only materialises *unused inputs*, not an output
    # that does not require grad).
    #
    # A straight-line initial guess is already optimal for the identity metric, so
    # GEORCE would stop before reaching that step. We therefore seed a *curved*
    # (non-optimal) initial trajectory -- as SolverGraphGEORCE does from its KNN
    # path -- so the loop actually iterates into the previously failing branch.
    T = 12
    solver = solvers.GEORCE(
        cometric.IdentityCoMetric(is_diag=True), T=T, max_iter=50, tol=1e-8
    )

    x_0 = torch.tensor([0.2, 0.4])
    x_T = torch.tensor([0.8, 0.6])

    t = torch.linspace(0, 1, T)
    x_t_0 = x_0[None, :] + t[1:-1, None] * (x_T - x_0)[None, :]
    x_t_0 = x_t_0 + 0.1 * torch.sin(3 * torch.pi * t[1:-1, None])  # bend it
    # NOTE: endpoints deliberately do NOT require grad, so u_t_i never requires grad.
    # This forces the previously-failing branch (v_t not requiring grad) on the very
    # first iteration, independent of how fast the optimizer converges.
    expected = (x_T - x_0).norm()

    # (a) Non-grad endpoints: hits the failing branch immediately. Must not raise.
    _x_final, dst, _ngE, E, _alpha = solver.georce_solver(x_0, x_T, x_t_0)
    assert solver.georce_stats.n_iter >= 1
    # Optimal (straight-line) energy for the identity metric: 0.5 * |diff|^2 / (T-1).
    assert torch.allclose(E[-1], 0.5 * expected**2 / (T - 1), atol=1e-3)
    assert torch.allclose(dst[-1], expected, atol=1e-3)

    # (b) Grad-requiring endpoints (the notebook's exact scenario): must also work.
    solver2 = solvers.GEORCE(
        cometric.IdentityCoMetric(is_diag=True), T=T, max_iter=50, tol=1e-8
    )
    x_0g = x_0.clone().requires_grad_(True)
    x_Tg = x_T.clone().requires_grad_(True)
    _x_final2, dst2, _ngE2, _E2, _alpha2 = solver2.georce_solver(x_0g, x_Tg, x_t_0.clone())
    assert torch.allclose(dst2[-1], expected, atol=1e-3)


def test_georce_finsler_constant_like_metric():
    # Regression test: ensure the Finsler GEORCE does not hit the same
    # "does not require grad" failure when the Finsler metric is constant-like
    # (position-independent), e.g. a Randers metric with a constant base cometric
    # AND a constant 1-form.
    #
    # The Finsler solver computes its force/velocity terms via torch.func.grad
    # (get_v_t / get_zeta_t), which -- unlike torch.autograd.grad -- returns zero
    # for an argument the function does not depend on. This test locks that in,
    # so a future refactor back to torch.autograd.grad is caught here.
    T = 12
    randers = finsler.RandersMetrics(
        base_cometric=cometric.IdentityCoMetric(is_diag=True),
        omega=ConstantOneForm(dim=2),
        beta=0.5,
    )
    solver = solvers.GEORCEFinsler(finsler=randers, T=T, max_iter=50, tol=1e-8)

    x_0 = torch.tensor([0.2, 0.4])
    x_T = torch.tensor([0.8, 0.6])

    t = torch.linspace(0, 1, T)
    x_t_0 = x_0[None, :] + t[1:-1, None] * (x_T - x_0)[None, :]
    x_t_0 = x_t_0 + 0.1 * torch.sin(3 * torch.pi * t[1:-1, None])  # bend it

    # Must not raise, and must produce a finite, positive distance.
    _x_final, dst, _ngE, E, _alpha = solver.georce_solver(x_0, x_T, x_t_0)
    assert torch.isfinite(dst[-1])
    assert dst[-1] > 0
    assert torch.isfinite(E[-1])
