"""
Pin the public API surface.

Every name declared in a sub-package's ``__all__`` must be accessible from
that sub-package and from the top-level ``geodesic_toolbox`` package (the
top-level ``__init__`` star-imports each sub-package).
"""

import importlib

import pytest

import geodesic_toolbox

ALL_PACKAGES = ["cometric", "finsler", "integrators", "samplers", "solvers", "utils"]

# A representative subset of the public names each sub-package must expose.
# The full ``__all__`` check below covers the rest.
KEY_NAMES = {
    "cometric": [
        "CoMetric",
        "IdentityCoMetric",
        "CentroidsCometric",
        "LANDCometric",
        "PointCarreCoMetric",
        "SoftAbsCometric",
        "mat_sqrt",
        "safe_eigh",
    ],
    "finsler": [
        "FinslerMetric",
        "RandersMetrics",
        "DualRandersMetrics",
        "MatsumotoMetrics",
        "SlopeMetrics",
        "ToyFinslerMetric",
        "BinetLegendreRanders",
    ],
    "integrators": [
        "Integrator",
        "Hamiltonian",
        "EulerIntegrator",
        "ImplicitMidpointIntegrator",
        "SeparableLeapfrogIntegrator",
        "ImplicitLeapfrogIntegrator",
        "ExplicitLeapfrogIntegrator",
        "HamiltonianImplicitMidpointIntegrator",
    ],
    "samplers": [
        "Sampler",
        "MALA",
        "HamiltonianMonteCarlo",
        "ImplicitMidpointRHMCSampler",
        "ImplicitLeapfrogRHMCSampler",
        "ExplicitRHMCSampler",
        "ImplicitMidpointRHMCRFSampler",
        "ImplicitLeapfrogRHMCRFSampler",
        "ExplicitRHMCRFSampler",
    ],
    "solvers": [
        "GeodesicDistanceSolver",
        "ShootingSolver",
        "BVP_shooting",
        "BVP_ode",
        "SolverGraph",
        "SolverGraphFinsler",
        "GEORCE",
        "GEORCEFinsler",
        "SolverGraphBVP",
        "GeodesicSolverSpline",
        "ExpMapRanders",
        "ExpMapRiemann",
        "ExpMapFinsler",
        "dst_mat",
    ],
    "utils": [
        "magnification_factor",
        "scale_lr_magnification",
        "batched_kro",
        "vec",
        "sample_cometric_normal",
        "empirical_cov_mat",
    ],
}


def test_subpackages_declare_all():
    for package_name in ALL_PACKAGES:
        module = importlib.import_module(f"geodesic_toolbox.{package_name}")
        assert hasattr(module, "__all__"), (
            f"geodesic_toolbox.{package_name} does not define __all__"
        )
        assert module.__all__, f"geodesic_toolbox.{package_name}.__all__ is empty"


@pytest.mark.parametrize("package_name", ALL_PACKAGES)
def test_subpackage_exposes_declared_names(package_name):
    module = importlib.import_module(f"geodesic_toolbox.{package_name}")
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing, f"geodesic_toolbox.{package_name} is missing: {missing}"


@pytest.mark.parametrize("package_name, names", list(KEY_NAMES.items()))
def test_key_names_accessible_from_subpackage(package_name, names):
    module = importlib.import_module(f"geodesic_toolbox.{package_name}")
    missing = [name for name in names if not hasattr(module, name)]
    assert not missing, f"geodesic_toolbox.{package_name} is missing: {missing}"


@pytest.mark.parametrize(
    "name",
    sorted({name for names in KEY_NAMES.values() for name in names}),
)
def test_key_names_importable_from_top_level(name):
    assert hasattr(geodesic_toolbox, name), f"geodesic_toolbox.{name} is missing"


def test_all_declared_names_importable_from_top_level():
    for package_name in ALL_PACKAGES:
        module = importlib.import_module(f"geodesic_toolbox.{package_name}")
        missing = [name for name in module.__all__ if not hasattr(geodesic_toolbox, name)]
        assert not missing, f"top level is missing {package_name} names: {missing}"
