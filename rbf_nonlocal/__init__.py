"""
rbf-nonlocal-balance: A Meshless RBF Solver for Nonlocal Balance Equations.

This package implements the radial-basis-function collocation scheme of

    Y. Melouani, A. Bouhamidi, I. El Harraki,
    "A Meshless Radial Basis Function Method for Nonlocal Balance Equations,"
    preprint, 2025,

with high-order BDF time stepping and an analytical-Jacobian Newton-Krylov
solver. It targets the system

.. math::
    \\partial_t u + \\nabla \\cdot (V(\\mathcal{W}_u) \\otimes u)
    + \\mathcal{L} u = \\mathcal{N}(u) + f

on irregular two-dimensional domains, where :math:`\\mathcal{L} u`
is either :math:`-\\nu\\,\\Delta u` (advection-diffusion) or
:math:`\\varepsilon\\,\\Delta^2 u` (pure nonlocal convection with
hyper-viscosity).

Quick start
-----------
The high-level API is :class:`NonlocalBalanceSolver`::

    from rbf_nonlocal import NonlocalBalanceSolver

    solver = NonlocalBalanceSolver(
        domain="flower",
        velocity="exp_density",
        regularization="laplacian",
        mode="validation",
        n_interior=80, n_boundary=80,
        nu=[[0.01, 0.0], [0.0, 0.01]],
        dt=0.01, t_final=5.0,
        polydeg=4, tau=10.0,
    )
    result = solver.run()
    solver.plot_solution(result, save_dir="figures/run1")

Public surface
--------------
The two tiers below are imported and re-exported here. The top tier is
what the README's quick-start uses; the lower tier is for users who want
to swap individual components or build their own solver.

* High-level: :class:`NonlocalBalanceSolver`, :class:`SolverResult`,
  :func:`make_domain`, :func:`make_velocity`, :func:`make_reaction`.

* Domain & basis: :class:`Domain`, :class:`FlowerDomain`,
  :class:`GearDomain`.

* Physics: :class:`Velocity`, :class:`VelocityExpDensity`,
  :class:`Reaction`, :class:`DefaultReaction`,
  :class:`Kernel`, :class:`GaussianKernel`.

* Manufactured solution: :class:`ManufacturedSolution`,
  :class:`DefaultManufacturedSolution`.

* Numerics: :func:`build_matrices`, :func:`compute_basis_laplacians`,
  :func:`compute_basis_bilaplacians`, :func:`compute_basis_gradients`,
  :func:`compute_nonlocal_operators`, :func:`setup_quadrature`,
  :class:`RegularizationOperator`, :func:`integrate`,
  :func:`newton_krylov_solve`, :data:`BDF_COEFFS`.

* Plotting: :func:`plot_contours`, :func:`plot_surfaces_3d`,
  :func:`plot_error_history`, :func:`plot_iteration_diagnostics`,
  :func:`plot_solution`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# High-level API (the top tier)
# ---------------------------------------------------------------------------

from .solver import NonlocalBalanceSolver, SolverResult

from .domains  import Domain, FlowerDomain, GearDomain, make_domain
from .velocity import Velocity, VelocityExpDensity, make_velocity
from .reaction import Reaction, DefaultReaction, make_reaction


# ---------------------------------------------------------------------------
# Lower-level building blocks
# ---------------------------------------------------------------------------

# Basis kernels
from . import basis  # noqa: F401  (full module exposed for advanced use)

# Interpolation matrices
from .matrices import (
    RBFInterpolationMatrices,
    build_matrices,
    compute_basis_laplacians,
    compute_basis_bilaplacians,
    compute_basis_gradients,
    evaluate_basis_functions,
)

# Nonlocal operator assembly
from .nonlocal_ops import (
    Kernel,
    GaussianKernel,
    NonlocalOperators,
    setup_quadrature,
    compute_nonlocal_operators,
)

# Manufactured solution
from .manufactured import (
    ManufacturedSolution,
    DefaultManufacturedSolution,
    precompute_source_terms,
)

# Right-hand side
from .rhs import (
    RegularizationOperator,
    compute_rhs,
    frechet_derivative,
    make_frechet_operator,
)

# Time integration
from .time_integration import (
    BDF_COEFFS,
    NewtonReport,
    StepReport,
    IntegrationReport,
    bdf_step,
    integrate,
    newton_krylov_solve,
)

from .plotting import (
    plot_collocation_points,
    plot_solution_field,
    plot_relative_error_field,
    plot_solution_3d,
    plot_error_history,
    plot_iteration_diagnostics,
    plot_solution,
)


# ---------------------------------------------------------------------------
# Public symbol list
# ---------------------------------------------------------------------------
#
# Order matters here: high-level first, lower-level after. ``from
# rbf_nonlocal import *`` is discouraged in production code (better to be
# explicit), but for the README's quick-start it should pull in the
# sensible defaults without polluting the user's namespace with
# ``np``, ``Path``, etc.

__all__ = [
    # version
    "__version__",
    # ---- top tier ----
    "NonlocalBalanceSolver",
    "SolverResult",
    "make_domain",
    "make_velocity",
    "make_reaction",
    # ---- domain ----
    "Domain",
    "FlowerDomain",
    "GearDomain",
    # ---- physics ----
    "Velocity",
    "VelocityExpDensity",
    "Reaction",
    "DefaultReaction",
    "Kernel",
    "GaussianKernel",
    # ---- manufactured solution ----
    "ManufacturedSolution",
    "DefaultManufacturedSolution",
    "precompute_source_terms",
    # ---- interpolation matrices ----
    "RBFInterpolationMatrices",
    "build_matrices",
    "compute_basis_laplacians",
    "compute_basis_bilaplacians",
    "compute_basis_gradients",
    "evaluate_basis_functions",
    # ---- nonlocal operators ----
    "NonlocalOperators",
    "setup_quadrature",
    "compute_nonlocal_operators",
    # ---- right-hand side ----
    "RegularizationOperator",
    "compute_rhs",
    "frechet_derivative",
    "make_frechet_operator",
    # ---- time integration ----
    "BDF_COEFFS",
    "NewtonReport",
    "StepReport",
    "IntegrationReport",
    "bdf_step",
    "integrate",
    "newton_krylov_solve",
    # ---- plotting ----
    "plot_contours",
    "plot_surfaces_3d",
    "plot_error_history",
    "plot_iteration_diagnostics",
    "plot_solution",
]