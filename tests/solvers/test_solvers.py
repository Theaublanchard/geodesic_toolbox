import torch

from geodesic_toolbox import cometric, finsler, solvers


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
