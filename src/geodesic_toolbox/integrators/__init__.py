"""
Script containing integrators.
We first introduce the general operator for ODE of the sort
    dx/dt = f(x)
Then we focus on the case of Hamiltonian dynamics, where
    f(x) = J * grad H(x)
with J the symplectic matrix and H the Hamiltonian function.
"""

from .base import BatchedFunction, Integrator
from .euler import EulerIntegrator, HamiltonianEulerIntegrator
from .hamiltonian import Hamiltonian, HamiltonianIntegrator, HamiltonianToBatchedFunction
from .implicit_midpoint import (
    HamiltonianImplicitMidpointIntegrator,
    ImplicitMidpointIntegrator,
)
from .leapfrog import (
    ExplicitLeapfrogIntegrator,
    ImplicitLeapfrogIntegrator,
    SeparableLeapfrogIntegrator,
)

__all__ = [
    "BatchedFunction",
    "EulerIntegrator",
    "ExplicitLeapfrogIntegrator",
    "Hamiltonian",
    "HamiltonianEulerIntegrator",
    "HamiltonianImplicitMidpointIntegrator",
    "HamiltonianIntegrator",
    "HamiltonianToBatchedFunction",
    "ImplicitLeapfrogIntegrator",
    "ImplicitMidpointIntegrator",
    "Integrator",
    "SeparableLeapfrogIntegrator",
]
