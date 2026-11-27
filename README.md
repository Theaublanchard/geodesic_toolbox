# Geodesic Toolbox: Riemannian and Finsler metric learning in PyTorch

[![Python 3.11+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)(LICENSE)

## Overview

**Geodesic Toolbox** is a PyTorch library for computing geodesic distances, trajectories and related differential-geometric quantities on Riemannian and Finsler manifolds of the form $(\mathbb{R}^d, g)$. It is built around modelling metrics through their **cometric** $g^{-1}$, and provides a differentiable framework for metric learning: solvers, samplers and integrators can all be optimized through.

### Key features

- **Geodesic solvers** for both Riemannian and Finsler manifolds: GEORCE, graph solvers, chained graph + refinement solvers, shooting, BVP, spline and exponential-map solvers
- **Rich metric support**: Euclidean, Poincaré, pullback, Fisher-Rao, centroid/LAND data-driven cometrics, parametric learnable cometrics, and Randers / Matsumoto / Slope Finsler metrics (including the dual Randers metric and the Binet-Legendre cometric)
- **Samplers and integrators**: MALA, a family of Riemannian HMC samplers, and Hamiltonian integrators (Euler, leapfrog, implicit midpoint)
- **Distance matrices**: batched pairwise geodesic distances via `dst_mat`
- **Fully differentiable**: everything runs in PyTorch, so metrics can be optimized end-to-end
- **GPU support**: native CUDA support for large-scale computations
- **Examples**: Jupyter notebooks in `examples/`

## Installation

We recommend [uv](https://docs.astral.sh/uv/). To use the library as a dependency:

```bash
uv add git+https://github.com/Theaublanchard/geodesic_toolbox
```

To develop against the repository:

```bash
git clone https://github.com/Theaublanchard/geodesic_toolbox.git
cd geodesic_toolbox
uv sync
```

To add it as an editable dependency of another project, put this in that project's `pyproject.toml`:

```toml
[tool.uv.sources]
geodesic-toolbox = { path = "<path>/geodesic_toolbox", editable = true }
```

## Quick start

### Geodesic distance on a Riemannian manifold

```python
import torch
from geodesic_toolbox import IdentityCoMetric, GEORCE

# Euclidean metric
cometric = IdentityCoMetric()
solver = GEORCE(cometric, T=100, max_iter=200)

q0 = torch.tensor([[0.0, 0.0]])
q1 = torch.tensor([[1.0, 1.0]])

distance = solver(q0, q1)
print(f"Geodesic distance: {distance.item():.4f}")
```

### Poincaré ball (hyperbolic) geometry

```python
from geodesic_toolbox import PointCarreCoMetric, GEORCE

cometric = PointCarreCoMetric()
solver = GEORCE(cometric, T=100, max_iter=200)

q0 = torch.tensor([[0.0, 0.0]])
q1 = torch.tensor([[0.5, 0.5]])  # inside the unit ball

distance = solver(q0, q1)
print(f"Hyperbolic distance: {distance.item():.4f}")
```

### Custom Riemannian metric via a neural network (pullback)

```python
import torch.nn as nn
from geodesic_toolbox import IdentityCoMetric, PullBackCometric, GEORCE


class MyDiffeomorphism(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


cometric = PullBackCometric(
    diffeo=MyDiffeomorphism(),
    base_cometric=IdentityCoMetric(is_diag=False),
    method="finite_difference",
)
solver = GEORCE(cometric, T=100, max_iter=200)
distance = solver(q0, q1)
```

### Randers (Finsler) metric

A Randers metric is $F(x, v) = |v|_{g(x)} + \beta\, \omega(x) \cdot v$ with a 1-form $\omega$ satisfying the Randers condition $\|\beta\, \omega(x)\|_{g(x)^{-1}} < 1$.

```python
import torch.nn as nn
from geodesic_toolbox import IdentityCoMetric, RandersMetrics, GEORCEFinsler


class Omega(nn.Module):
    """1-form omega(x) = rho(x) * dir(x), with a cometric-normalized direction."""

    def __init__(self, dim: int, base_cometric):
        super().__init__()
        self.direction = nn.Linear(dim, dim)
        self.amplitude = nn.Linear(dim, 1)
        self.base_cometric = base_cometric

    def forward(self, x):
        rho = self.amplitude(x).clip(0.0, 1.0)
        omega = self.direction(x)
        # base_cometric.cometric(x, omega) is the dual norm ||omega||_{g^-1}
        omega = omega / (self.base_cometric.cometric(x, omega)[:, None] + 1e-8)
        return rho * omega


base = IdentityCoMetric()
randers = RandersMetrics(base_cometric=base, omega=Omega(2, base), beta=0.5)
solver = GEORCEFinsler(randers, T=100, max_iter=200)
distance = solver(q0, q1)
```

Every solver in the library accepts batches of start/end points `(b, d)` and is differentiable in `x`, `v` and in the metric parameters.

## Solvers

All solvers implement the same interface: `solver(q0, q1)` returns the geodesic distance, and most expose `get_trajectories(q0, q1)`.

| Solver          | Class                                                      | Manifold   | Notes                                                       |
| --------------- | ---------------------------------------------------------- | ---------- | ----------------------------------------------------------- |
| GEORCE          | `GEORCE` / `GEORCEFinsler`                             | both       | Energy-minimizing trajectory; accurate, needs good init.    |
| Graph           | `SolverGraph` / `SolverGraphFinsler`                   | both       | Fast, data-driven (KNN graph), approximate.                 |
| Graph + GEORCE  | `SolverGraphGEORCE` / `SolverGraphGEORCEFinsler`       | both       | Graph initialisation, GEORCE refinement.                    |
| Graph + Spline  | `SolverGraphSpline` / `SolverGraphSplineFinsler`       | both       | Graph initialisation, spline refinement.                    |
| Graph + BVP     | `SolverGraphBVP`                                         | Riemannian | Graph initialisation, BVP refinement.                       |
| Shooting        | `ShootingSolver`                                         | Riemannian | Optimises initial momentum; fails on complex metrics.       |
| BVP             | `BVP_ode` / `BVP_shooting`                             | Riemannian | Solves the geodesic ODE as a BVP (non-diagonal metrics).    |
| Spline          | `GeodesicSolverSpline` / `GeodesicSolverSplineFinsler` | both       | Spline parameterisation of the curve.                       |
| Exponential map | `ExpMapRiemann` / `ExpMapRanders` / `ExpMapFinsler`  | both       | Integrates the (Hamiltonian) geodesic flow from `(q, v)`. |

### Graph-based solvers (fast, data-driven)

```python
import torch
from geodesic_toolbox import IdentityCoMetric, SolverGraph, SolverGraphFinsler

data = torch.randn(1000, 2)
cometric = IdentityCoMetric()

solver = SolverGraph(
    cometric=cometric,
    data=data,
    n_neighbors=10,
    T=100,          # trajectory discretisation
    batch_size=64,
    max_data_count=500,  # subsample for memory efficiency
)
distance = solver(q0, q1)
```

### Chained solvers (graph initialisation + refinement)

```python
from geodesic_toolbox import SolverGraphGEORCE, SolverGraphGEORCEFinsler

solver = SolverGraphGEORCE(
    cometric=cometric,
    data=data,
    n_neighbors=10,
    T=100,
    max_iter=100,
    tol=1e-5,
)
distance = solver(q0, q1)

# Same interface for the Finsler variant
solver = SolverGraphGEORCEFinsler(finsler=randers, data=data, n_neighbors=10)
```

### Exponential map (trajectory from an initial velocity)

```python
from geodesic_toolbox import ExpMapRanders

expmap = ExpMapRanders(randers=randers, T_max=10.0, T=500, substeps=4)
traj_q, traj_v = expmap(q0, v0)
```

## Supported metrics

### Riemannian cometrics

| Metric         | Class                                                                                      | Use case                                 |
| -------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------- |
| Euclidean      | `IdentityCoMetric`                                                                       | Baseline                                 |
| Poincaré ball | `PointCarreCoMetric`                                                                     | Hyperbolic geometry                      |
| Pullback       | `PullBackCometric`                                                                       | Neural-network-defined metrics           |
| Fisher-Rao     | `FisherRaoCometric`                                                                      | Information geometry                     |
| Lifted         | `LiftedCometric`                                                                         | Level-set constraints                    |
| Centroid RBF   | `CentroidsCometric`                                                                      | Gaussian interpolation at data centroids |
| LAND           | `LANDCometric`, `LANDRBFCometric`                                                      | Data-driven diagonal cometrics           |
| Soft-abs       | `SoftAbsCometric`                                                                        | Enforcing positive definiteness          |
| Parametric     | `DiagonalCometricModel`, `CometricModel`, `Cometric_MLP`, `SmallConvCometricModel` | Learnable metrics                        |

### Finsler metrics

| Metric         | Class                    | Use case                                    |
| -------------- | ------------------------ | ------------------------------------------- |
| Randers        | `RandersMetrics`       | Direction-dependent geometry                |
| Dual Randers   | `DualRandersMetrics`   | Dual (cotangent) Randers metric             |
| Binet-Legendre | `BinetLegendreRanders` | Binet-Legendre cometric of a Randers metric |
| Matsumoto      | `MatsumotoMetrics`     | Alternative Finsler metric                  |
| Slope          | `SlopeMetrics`         | Terrain-based metrics                       |

### Metric composition

Cometrics can be scaled, added, wrapped, and used anywhere a cometric is expected:

```python
from geodesic_toolbox import IdentityCoMetric, PointCarreCoMetric, SoftAbsCometric

scaled = 10.5 * IdentityCoMetric()
combined = IdentityCoMetric() + PointCarreCoMetric()

# Enforce positive-definiteness of any metric
safe = SoftAbsCometric(PointCarreCoMetric(), alpha=1e3)

solver = GEORCE(combined, T=100, max_iter=200)
```

## Samplers and integrators

For sampling and Hamiltonian dynamics on manifolds:

```python
from geodesic_toolbox import (
    IdentityCoMetric,
    ImplicitMidpointRHMCSampler,
    MALA,
    EulerIntegrator,
    ImplicitMidpointIntegrator,
    SeparableLeapfrogIntegrator,
)

# Riemannian HMC-style sampler on a cometric
sampler = ImplicitMidpointRHMCSampler(
    cometric=IdentityCoMetric(),
    num_integration_steps=10,
    fixed_point_iterations=100,
    step_size=0.1,
    log_target=lambda x: -0.5 * (x**2).sum(dim=-1),
)

# MALA given a log-target and its score
mala = MALA(
    log_target=lambda x: -0.5 * (x**2).sum(dim=-1),
    score_func=lambda x: -x,
    step_size=0.1,
)
```

Samplers and integrators are also available for Finsler manifolds (e.g. `ExplicitRHMCSampler`, `HamiltonianMonteCarlo`); see `geodesic_toolbox.samplers` and `geodesic_toolbox.integrators` for the full list.

## Advanced usage

### Differentiable optimisation through geodesics

Since all operations are differentiable, you can optimise metric parameters to shape geodesic distances:

```python
import torch
from geodesic_toolbox import DiagonalCometricModel, GEORCE

metric = DiagonalCometricModel(in_dim=2, hidden_dim=32, latent_dim=2, lbd=1e-3)
optimizer = torch.optim.Adam(metric.parameters(), lr=1e-3)
solver = GEORCE(metric, T=100, max_iter=50)

for _ in range(100):
    q0 = torch.randn(32, 2)
    q1 = torch.randn(32, 2)
    loss = solver(q0, q1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Visualising trajectories

```python
import matplotlib.pyplot as plt

traj = solver.get_trajectories(q0, q1)

plt.figure(figsize=(8, 8))
plt.plot(traj[0, :, 0].detach(), traj[0, :, 1].detach(), "b-", label="Geodesic")
plt.scatter([q0[0, 0].item()], [q0[0, 1].item()], c="green", s=100, label="Start")
plt.scatter([q1[0, 0].item()], [q1[0, 1].item()], c="red", s=100, label="End")
plt.legend()
plt.axis("equal")
plt.show()
```

## Examples

Complete examples are available in `examples/`:

- `explore_riemannian_metric.ipynb` — basic Riemannian geodesics
- `explore_riemannian_solvers.ipynb` — comparing different solvers
- `explore_randers_metric.ipynb` — building and inspecting a Randers metric
- `explore_randers_solvers.ipynb` — solvers on a Finsler manifold
- `explore_randers_hamiltonian.ipynb` — Hamiltonian formulation and exponential maps

## Performance considerations

- **Memory**: high-dimensional data can be memory-intensive. Prefer diagonal cometrics (`is_diag=True`) where possible, and use the graph solvers or `max_data_count` subsampling for large datasets.
- **Numerical stability**: wrap uncertain metrics in `SoftAbsCometric` and add regularisation (`lbd`) in parametric cometrics to keep them positive definite.
- **Finsler condition**: for `RandersMetrics`, keep $\|\beta\, \omega(x)\|_{g^{-1}} < 1$ everywhere — the tests and notebooks clamp the 1-form amplitude accordingly.

## Testing

```bash
uv run pytest tests/ -v

# with coverage
uv run pytest --cov=geodesic_toolbox --cov-report=term-missing
```

## Citation

If you use Geodesic Toolbox in your research, please cite:

```bibtex
@software{blanchard2025geodesic_toolbox,
  author = {Blanchard, Théau},
  title = {Geodesic Toolbox: Differentiable Geodesic Computation on Riemannian and Finsler Manifolds},
  year = {2025},
  url = {https://github.com/Theaublanchard/geodesic_toolbox},
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authors

- **Théau Blanchard** — initial development

## License

This project is licensed under the CC BY-NC-ND 4.0 License — see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions or suggestions:

- Open an [Issue](https://github.com/Theaublanchard/geodesic_toolbox/issues)
- Start a [Discussion](https://github.com/Theaublanchard/geodesic_toolbox/discussions)
