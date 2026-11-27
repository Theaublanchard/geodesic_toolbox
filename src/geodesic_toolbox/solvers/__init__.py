"""
Geodesic solvers: shooting, BVP, graph, GEORCE, chained (graph + refinement),
spline, distance matrix, and exponential map solvers.
"""

from .base import GeodesicDistanceSolver
from .bvp import BVP_ode, BVP_shooting, BVP_wrapper
from .chained import (
    SolverGraphBVP,
    SolverGraphGEORCE,
    SolverGraphGEORCEFinsler,
    SolverGraphSpline,
    SolverGraphSplineFinsler,
)
from .distance_matrix import angle_mat, dst_mat, dst_mat_naive, dst_mat_vectorized
from .expmap import (
    ExpMapFinsler,
    ExpMapRanders,
    ExpMapRiemann,
    HamiltonianRanders,
    HamiltonianRiemann,
)
from .georce import (
    GEORCE,
    GEORCEFinsler,
    GEORCEIterationStats,
    GEORCELineSearchStats,
    GEORCERunStats,
)
from .graph import SolverGraph, SolverGraphFinsler
from .shooting import ShootingSolver
from .spline import (
    GeodesicSolverSpline,
    GeodesicSolverSplineFinsler,
    GeodesicSpline,
)

__all__ = [
    "GEORCE",
    "BVP_ode",
    "BVP_shooting",
    "BVP_wrapper",
    "ExpMapFinsler",
    "ExpMapRanders",
    "ExpMapRiemann",
    "GEORCEFinsler",
    "GEORCEIterationStats",
    "GEORCELineSearchStats",
    "GEORCERunStats",
    "GeodesicDistanceSolver",
    "GeodesicSolverSpline",
    "GeodesicSolverSplineFinsler",
    "GeodesicSpline",
    "HamiltonianRanders",
    "HamiltonianRiemann",
    "ShootingSolver",
    "SolverGraph",
    "SolverGraphBVP",
    "SolverGraphFinsler",
    "SolverGraphGEORCE",
    "SolverGraphGEORCEFinsler",
    "SolverGraphSpline",
    "SolverGraphSplineFinsler",
    "angle_mat",
    "dst_mat",
    "dst_mat_naive",
    "dst_mat_vectorized",
]
