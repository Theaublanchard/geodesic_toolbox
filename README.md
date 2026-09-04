# Geodesic Toolbox: Riemannian metric and Finsler metric learning in PyTorch

<!-- [![PyPI version](https://img.shields.io/pypi/v/geodesic_toolbox.svg)](https://pypi.org/project/geodesic_toolbox/) -->

[![Python 3.11+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)(LICENSE)

## Overview

**Geodesic Toolbox** is a PyTorch-based library for computing geodesic distances, trajectories, and related differential geometric properties on Riemannian and Finsler manifolds. It provides a flexible, differentiable framework for working with user-defined metrics on manifolds represented as $(\mathbb{R}^d, g)$, where $g$ is a Riemannian metric. It aims to provide an easy to use library to solve tasks around metric learning.

The library is centered around the modeling of metrics via their cometric tensor $g^{-1}$, allowing for efficient computation of geodesics using various numerical solvers. It supports a wide range of built-in metrics, including Euclidean, Poincaré, Pullback metrics defined via neural networks, Fisher-Rao, and several Finsler metrics such as Randers and Matsumoto metrics. Users can also define custom metrics by subclassing the provided base classes.

### Key Features

- **Multiple Solver Implementations**: Shooting method, Boundary Value Problem (BVP) solvers, graph-based solvers, and the GEORCE algorithm
- **Rich Metric Support**:
  - Riemannian metrics (Identity, Poincaré, Pullback, Fisher-Rao, custom parametric)
  - Finsler metrics (Randers, Matsumoto, Slope metrics)
  - Metric composition and interpolation
- **Fully Differentiable**: All computations are implemented in PyTorch, enabling gradient-based optimization through geodesics
- **GPU Support**: Native CUDA support for large-scale computations
- **Flexible Architecture**: Easy to define custom metrics and solvers
- **Comprehensive Examples**: Jupyter notebooks demonstrating metric construction and geodesic computation

## Installation

We recommend using uv to install the project. To add the library as a dependency, run:

```bash
uv add git+https://github.com/Theaublanchard/geodesic_toolbox
```

To install the library in development mode, clone the repository and run:

```bash
git clone https://github.com/Theaublanchard/geodesic_toolbox.git
cd geodesic_toolbox
uv sync
```

And if you need it in another project, as editable, add the following to your library's `pyproject.toml`:

```toml
[tool.uv.sources]
geodesic-toolbox = { path = "geodesic_toolbox", editable = true }
```

## Quick Start

### Computing a Simple Geodesic Distance

```python
import torch
from geodesic_toolbox import ShootingSolver, IdentityCoMetric

# Define the metric (Euclidean space)
cometric = IdentityCoMetric()
solver = ShootingSolver(
    cometric=cometric,
    lr=0.1,
    n_step=100,
    dt=0.01
)

# Two points on the manifold
q0 = torch.tensor([[0.0, 0.0]])
q1 = torch.tensor([[1.0, 1.0]])

# Compute geodesic distance
distance = solver(q0, q1)
print(f"Geodesic distance: {distance.item():.4f}")
```

### Using the Poincaré Ball Metric

```python
from geodesic_toolbox import PointCarreCoMetric

# Hyperbolic geometry on the Poincaré ball
cometric = PointCarreCoMetric()
solver = ShootingSolver(cometric=cometric, n_step=200)

# Points inside the unit ball
q0 = torch.tensor([[0.0, 0.0]])
q1 = torch.tensor([[0.5, 0.5]])

# Hyperbolic distance
distance = solver(q0, q1)
print(f"Hyperbolic distance: {distance.item():.4f}")
```

### Custom Riemannian Metrics

```python
from geodesic_toolbox import PullBackCometric
import torch.nn as nn

# Define a diffeomorphism (e.g., neural network)
class MyDiffeomorphism(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output dimension
        )
  
    def forward(self, x):
        return self.net(x)

# Create pullback metric
diffeo = MyDiffeomorphism()
cometric = PullBackCometric(
    diffeo=diffeo,
    base_cometric=IdentityCoMetric(is_diag=False),
    method="finite_difference"
)

solver = ShootingSolver(cometric=cometric, n_step=100)
distance = solver(q0, q1)
```

## Solver Methods

The library provides multiple algorithms for geodesic computation, each with different trade-offs:

### 1. **Graph-Based Solver** (recommended for large datasets)

Uses KNN graphs for fast approximate geodesics on data manifolds.

```python
from geodesic_toolbox import SolverGraph

# Requires a reference dataset
data = torch.randn(1000, 2)

solver = SolverGraph(
    cometric=cometric,
    data=data,
    n_neighbors=10,
    dt=0.01,           # Trajectory discretization
    batch_size=64,
    max_data_count=500 # Subsample for memory efficiency
)

distance = solver(q0, q1)
```

**Advantages**: Very fast, handles non-convex regions
**Disadvantages**: Requires training data, approximate distances

### 4. **GEORCE Solver**

Control-theoretic approach optimizing geodesic energy.

```python
from geodesic_toolbox import GEORCE

solver = GEORCE(
    cometric=cometric,
    T=100,           # Trajectory discretization
    max_iter=200,    # Maximum iterations
    tol=1e-6,        # Convergence tolerance
    c=0.9, rho=0.5   # Line search parameters
)

distance = solver(q0, q1)
```

**Advantages**: Highly accurate
**Disadvantages**: Slower, requires careful tuning or good initialisation

### 3. **Hybrid Solver (Graph-Based + GEORCE)**

Combines graph-based initialization with GEORCE refinement.

```python
from geodesic_toolbox import SolverGraphGEORCE

solver = SolverGraphGEORCE(
    cometric=cometric,
    data=data,
    n_neighbors=10,
    T=100,
    max_iter=100,
    tol=1e-6
)
distance = solver(q0, q1)
```

**Advantages**: Balances speed and accuracy
**Disadvantages**: Requires to have training data. Longer runtime than pure graph-based solver.

## 4. **Shooting Solver** (not recommended for complex metrics)

Uses Hamiltonian mechanics to find geodesics by optimizing initial momentum.

```python
from geodesic_toolbox import ShootingSolver

solver = ShootingSolver(
    cometric=cometric,
    lr=0.1,                    # Learning rate for optimization
    n_step=100,                # Number of optimization steps
    dt=0.01,                   # Time discretization step
    method="euler",            # Integrator: "euler" or "leapfrog"
    convergence_threshold=1e-3 # Stop when endpoint error below threshold
)

distance = solver(q0, q1)
trajectory = solver.get_trajectories(q0, q1)  # Get full path
```

**Advantages**: Accurate
**Disadvantages**: Fails on complex metrics with singularities, expensive optimisation, doesn't work on Finsler metrics

### 5. **Boundary Value Problem (BVP) Solvers**

Directly solves the geodesic equation as a boundary value problem.

```python
from geodesic_toolbox import BVP_ode

solver = BVP_ode(
    cometric=cometric,
    T=100,        # Number of discretization points
    dim=2,        # Manifold dimension
    verbose=0     # Verbosity of scipy.integrate.solve_bvp
)

distance = solver(q0, q1)
```

**Advantages**: Accurate for smooth metrics
**Disadvantages**: Slow, high memory usage, may fail on complex metrics

## Supported Metrics

### Riemannian Metrics

| Metric          | Class                         | Use Case                       |
| --------------- | ----------------------------- | ------------------------------ |
| Euclidean       | `IdentityCoMetric`          | Baseline                       |
| Poincaré Ball  | `PointCarreCoMetric`        | Hyperbolic geometry            |
| Pullback Metric | `PullBackCometric`          | Neural network-defined metrics |
| Fisher-Rao      | `FisherRaoCometric`         | Information geometry           |
| Lifted Metric   | `LiftedCometric`            | Level set constraints          |
| Interpolated    | `CentroidsCometric`         | Data-driven metrics            |
| Parametric      | `DiagonalCometricModel` ... | Learnable metrics              |

### Finsler Metrics

| Metric    | Class                | Use Case                       |
| --------- | -------------------- | ------------------------------ |
| Randers   | `RandersMetrics`   | Directional-dependent geometry |
| Matsumoto | `MatsumotoMetrics` | Alternative Finsler metric     |
| Slope     | `SlopeMetrics`     | Terrain-based metrics          |

### Metric Composition

```python
from geodesic_toolbox import IdentityCoMetric, ScaledCometric, SumOfCometric

# Scale a metric
scaled = 10.5 * IdentityCoMetric()

# Sum two metrics
combined = IdentityCoMetric() + PointCarreCoMetric()

# Use in solver
solver = ShootingSolver(cometric=combined+scaled)
```

## Advanced Usage

### Differentiable Optimization Through Geodesics

Since all operations are differentiable, you can optimize parameters to minimize geodesic distances:

```python
import torch
import torch.nn as nn
from geodesic_toolbox import ShootingSolver, DiagonalCometricModel

# Learnable metric
metric = DiagonalCometricModel(
    in_dim=2,
    hidden_dim=32,
    latent_dim=2
)
optimizer = torch.optim.Adam(metric.parameters(), lr=1e-3)
solver = ShootingSolver(cometric=metric, n_step=50)

# Training loop
for epoch in range(100):
    q0 = torch.randn(32, 2, requires_grad=True)
    q1 = torch.randn(32, 2, requires_grad=True)
  
    distances = solver(q0, q1)
    loss = distances.mean()
  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Visualization

```python
import matplotlib.pyplot as plt
from geodesic_toolbox import ShootingSolver

solver = ShootingSolver(cometric=cometric, n_step=100)
traj = solver.get_trajectories(q0, q1)

# Plot trajectory
plt.figure(figsize=(8, 8))
plt.plot(traj[0, :, 0].detach(), traj[0, :, 1].detach(), 'b-', label='Geodesic')
plt.scatter([q0[0, 0]], [q0[0, 1]], c='green', s=100, label='Start')
plt.scatter([q1[0, 0]], [q1[0, 1]], c='red', s=100, label='End')
plt.legend()
plt.axis('equal')
plt.show()
```

## Examples

Complete examples are available in the `examples/` directory:

- `explore_riemannian_metric.ipynb` - Basic Riemannian geodesics
- `explore_riemannian_solvers.ipynb` - Comparing different solvers
- `explore_randers_metric.ipynb` - Finsler metric computation
- `explore_randers_solvers.ipynb` - Advanced Finsler examples

Run them with:

```bash
jupyter notebook examples/
```

## Performance Considerations

### Memory Usage

The library can be memory-intensive for high-dimensional data. It is recommended to use diagonal metrics (CoMetric.is_diag=True) when possible. Some approxiation strategies used are : batch processing, finite difference Jacobian computation, and graph solvers...

### Numerical Stability

```python
# Ensure positive-definite metrics
from geodesic_toolbox import SoftAbsCometric

safe_cometric = SoftAbsCometric(cometric, alpha=1e3)

# Add regularization
cometric = CometricModel(..., lbd=1e-3)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Coverage report:

```bash
pytest tests/ --cov=geodesic_toolbox
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
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

<!-- ## Roadmap

- [ ] Higher-order derivatives (for Christoffel symbols)
- [ ] Improved memory efficiency for high-dimensional jacobians
- [ ] Additional Finsler metric types
- [ ] Batch geodesic distance matrices
- [ ] Riemannian optimization utilities -->

<!-- ## Documentation

Comprehensive documentation is available at [docs/](docs/):

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Algorithms](docs/algorithms.md)
- [Theory](docs/theory.md) -->

## Authors

- **Théau Blanchard** - Initial development

## License

This project is licensed under the CC BY-NC-ND 4.0 License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or suggestions:

- Open an [Issue](https://github.com/Theaublanchard/geodesic_toolbox/issues)
- Start a [Discussion](https://github.com/Theaublanchard/geodesic_toolbox/discussions)
- Check [Discussions](docs/FAQ.md) for FAQs
