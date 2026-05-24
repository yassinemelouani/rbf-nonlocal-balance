"""
High-level solver API.

This module is the front door of the package. It wires the geometry,
basis, nonlocal operators, velocity, reaction, regulariser, and time
integrator together behind a single class, :class:`NonlocalBalanceSolver`,
that supports both validation against a manufactured solution and free
simulation of a user-supplied problem.

Typical usage
-------------

>>> from rbf_nonlocal import NonlocalBalanceSolver
>>> solver = NonlocalBalanceSolver(
...     domain="flower",
...     velocity="exp_density",
...     regularization="laplacian",
...     mode="validation",
...     n_interior=80, n_boundary=80,
...     nu=[[0.01, 0.0], [0.0, 0.01]],
...     dt=0.01, t_final=5.0,
...     polydeg=4, tau=10.0,
... )
>>> result = solver.run()
>>> print(result.err_inf[-1], result.err_l2[-1])

In simulation mode, ``mode="simulation"``, the user must supply an
``initial_condition`` callable (and may supply ``boundary_condition`` and
``source_term``). Errors are not computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .domains import Domain, make_domain
from .manufactured import (
    DefaultManufacturedSolution,
    ManufacturedSolution,
    precompute_source_terms,
)
from .matrices import (
    RBFInterpolationMatrices,
    build_matrices,
    compute_basis_bilaplacians,
    compute_basis_laplacians,
)
from .nonlocal_ops import (
    GaussianKernel,
    Kernel,
    NonlocalOperators,
    compute_nonlocal_operators,
    setup_quadrature,
)
from .reaction import DefaultReaction, Reaction, make_reaction
from .rhs import RegularizationOperator
from .time_integration import IntegrationReport, integrate
from .velocity import Velocity, make_velocity

__all__ = [
    "SolverResult",
    "NonlocalBalanceSolver",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """
    Everything the user might want after a run.

    Attributes
    ----------
    times : np.ndarray, shape ``(n_steps + 1,)``
        Time grid.
    history : np.ndarray, shape ``(n_steps + 1, 2, n)``
        Interior coefficients at each time step.
    boundary_history : np.ndarray, shape ``(n_steps + 1, 2, n_prime)``
        Dirichlet data at each time step.
    bdf_orders : np.ndarray, shape ``(n_steps,)``
        BDF order actually used at each step (ramps from 1 to ``bdf_max_order``).
    newton_iters : np.ndarray, shape ``(n_steps,)``
    gmres_iters : np.ndarray, shape ``(n_steps,)``
    err_inf, err_l2 : np.ndarray or None, shape ``(n_steps + 1,)``
        Relative :math:`L^\\infty` and :math:`L^2` errors, only populated
        in validation mode (otherwise ``None``).
    """
    times: np.ndarray
    history: np.ndarray
    boundary_history: np.ndarray
    bdf_orders: np.ndarray
    newton_iters: np.ndarray
    gmres_iters: np.ndarray
    err_inf: Optional[np.ndarray] = None
    err_l2: Optional[np.ndarray] = None

    @property
    def n_steps(self) -> int:
        return len(self.times) - 1

    @property
    def final_err_inf(self) -> Optional[float]:
        return None if self.err_inf is None else float(self.err_inf[-1])

    @property
    def final_err_l2(self) -> Optional[float]:
        return None if self.err_l2 is None else float(self.err_l2[-1])


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

InitFn  = Callable[[np.ndarray, np.ndarray], np.ndarray]            # (x, y) -> (n, 2)
BdyFn   = Callable[[float, np.ndarray, np.ndarray], np.ndarray]     # (t, x, y) -> (n', 2)
SrcFn   = Callable[[float, np.ndarray, np.ndarray], np.ndarray]     # (t, x, y) -> (n, 2)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class NonlocalBalanceSolver:
    """
    Configurable solver for the system

    .. math::
        \\partial_t u + \\nabla \\cdot (V(\\mathcal{W}_u) \\otimes u)
        + \\mathcal{L} u = \\mathcal{N}(u) + f.

    Parameters
    ----------
    domain : str or Domain
        ``"flower"``, ``"gear"``, or a :class:`Domain` instance.
    velocity : str or Velocity
        ``"exp_density"`` or a :class:`Velocity`
        instance.
    regularization : str
        ``"laplacian"`` (with diffusion ``nu``) or ``"bilaplacian"``
        (with hyper-viscosity ``epsilon``).
    mode : str
        ``"validation"`` — runs against the manufactured solution from
        :class:`rbf_nonlocal.manufactured.DefaultManufacturedSolution`
        and reports relative errors.
        ``"simulation"`` — runs from a user-supplied initial condition,
        with optional boundary and forcing terms; no error is computed.

    n_interior, n_boundary : int
        Number of interior and boundary collocation points.
    polydeg : int
        Degree of the polynomial tail (default 4).
    tau : float
        Tension parameter of the RBF (default 10.0). Ignored when
        ``regularization='bilaplacian'`` (which uses the thin-plate spline).

    nu : array-like (2, 2), optional
        Diffusion matrix; required when ``regularization='laplacian'``.
    epsilon : float, optional
        Hyper-viscosity coefficient; required when
        ``regularization='bilaplacian'``.

    dt : float
    t_final : float
    bdf_max_order : int

    reaction : str or Reaction, optional
        Defaults to :class:`rbf_nonlocal.reaction.DefaultReaction`.
    kernel : Kernel, optional
        Defaults to ``GaussianKernel(alpha=2.0)``.
    quad_order : int
        Per-axis Gauss-Legendre order for the nonlocal integral. Default 30.

    initial_condition : callable, optional
        ``(x, y) -> np.ndarray of shape (n, 2)``. **Required** in
        simulation mode; ignored in validation mode (where it's read from
        the manufactured solution).
    boundary_condition : callable, optional
        ``(t, x, y) -> np.ndarray of shape (n', 2)``. Defaults to
        zero in simulation mode and to the manufactured solution's trace
        in validation mode.
    source_term : callable, optional
        ``(t, x, y) -> np.ndarray of shape (n, 2)``. Defaults to zero
        in simulation mode and to the analytically derived forcing in
        validation mode.

    domain_kwargs : dict, optional
        Forwarded to :func:`make_domain`.
    velocity_kwargs : dict, optional
        Forwarded to :func:`make_velocity`.
    newton_kwargs : dict, optional
        Forwarded to :func:`rbf_nonlocal.time_integration.newton_krylov_solve`.
    n_jobs : int
        Joblib parallelism for source-term precomputation (validation
        mode only). Default ``-1`` (all cores).
    progress : bool
        If ``True``, print a one-line progress message every 100 steps.
    """

    def __init__(
        self,
        *,
        # Geometry / physics
        domain: Union[str, Domain],
        velocity: Union[str, Velocity],
        regularization: str,
        mode: str,
        # Discretisation
        n_interior: int,
        n_boundary: int,
        polydeg: int = 4,
        tau: float = 10.0,
        rbf_kind: Optional[str] = None,
        # Regularisation coefficients
        nu: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None,
        # Time
        dt: float,
        t_final: float,
        bdf_max_order: int = 6,
        # Optional pieces
        reaction: Union[str, Reaction] = "default",
        kernel: Optional[Kernel] = None,
        quad_order: int = 30,
        # User callbacks (simulation mode)
        initial_condition: Optional[InitFn] = None,
        boundary_condition: Optional[BdyFn] = None,
        source_term: Optional[SrcFn] = None,
        # Constructor side-channels
        domain_kwargs: Optional[dict] = None,
        velocity_kwargs: Optional[dict] = None,
        newton_kwargs: Optional[dict] = None,
        n_jobs: int = -1,
        progress: bool = False,
    ) -> None:
        # --- store / validate scalars ---
        if mode not in ("validation", "simulation"):
            raise ValueError(
                f"mode must be 'validation' or 'simulation' (got {mode!r})."
            )
        if regularization not in ("laplacian", "bilaplacian"):
            raise ValueError(
                f"regularization must be 'laplacian' or 'bilaplacian' "
                f"(got {regularization!r})."
            )
        if regularization == "laplacian" and nu is None:
            raise ValueError("regularization='laplacian' requires nu (a 2x2 array).")
        if regularization == "bilaplacian" and epsilon is None:
            raise ValueError("regularization='bilaplacian' requires epsilon (float).")
        if mode == "simulation" and initial_condition is None:
            raise ValueError("mode='simulation' requires an initial_condition callable.")

        self.mode = mode
        self.regularization = regularization
        self.dt = float(dt)
        self.t_final = float(t_final)
        self.bdf_max_order = int(bdf_max_order)
        self.n_jobs = n_jobs
        self.progress = progress
        self.newton_kwargs = newton_kwargs or {}

        # --- domain ---
        if isinstance(domain, Domain):
            self.domain = domain
        else:
            self.domain = make_domain(domain, **(domain_kwargs or {}))

        # --- velocity ---
        if isinstance(velocity, Velocity):
            self.velocity = velocity
        else:
            self.velocity = make_velocity(velocity, **(velocity_kwargs or {}))

        # --- reaction ---
        if isinstance(reaction, Reaction):
            self.reaction = reaction
        else:
            self.reaction = make_reaction(reaction)

        # --- kernel ---
        self.kernel = kernel if kernel is not None else GaussianKernel(alpha=2.0)

        # --- regularisation coefficients ---
        if nu is not None:
            self.nu = np.asarray(nu, dtype=float)
        else:
            self.nu = None
        self.epsilon = float(epsilon) if epsilon is not None else None

        # --- choose RBF kind based on regularisation, unless overridden ---
        if rbf_kind is None:
            self.rbf_kind = "tension" if regularization == "laplacian" else "thin_plate"
        else:
            if rbf_kind not in ("tension", "thin_plate"):
                raise ValueError(
                    f"rbf_kind must be 'tension' or 'thin_plate' (got {rbf_kind!r})."
                )
            self.rbf_kind = rbf_kind
        self.tau = float(tau)
        self.polydeg = int(polydeg)

        # --- discretisation sizes ---
        self.n_interior = int(n_interior)
        self.n_boundary = int(n_boundary)
        self.quad_order = int(quad_order)

        # --- user callables ---
        self.initial_condition_user = initial_condition
        self.boundary_condition_user = boundary_condition
        self.source_term_user = source_term

        # --- placeholders filled by .run() ---
        self.mats: Optional[RBFInterpolationMatrices] = None
        self.nl_ops: Optional[NonlocalOperators] = None
        self.reg: Optional[RegularizationOperator] = None
        self.exact: Optional[ManufacturedSolution] = None
        self._setup_done = False

    # ----------------------------------------------------------------- setup

    def setup(self) -> None:
        """
        Build geometry, matrices, and operators. Idempotent.

        Calling :meth:`run` calls :meth:`setup` automatically; calling
        ``setup`` directly is only useful when you want to inspect the
        precomputed matrices before time-stepping.
        """
        if self._setup_done:
            return

        # 1. Points.
        interior, boundary = self.domain.generate_points(
            n_interior=self.n_interior,
            n_boundary=self.n_boundary,
        )
        self.interior_points = interior
        self.boundary_points = boundary

        # 2. RBF matrices.
        self.mats = build_matrices(
            interior_points=interior,
            boundary_points=boundary,
            kind=self.rbf_kind,
            tau=self.tau,
            polydeg=self.polydeg,
        )

        # 3. Higher-order operator on the basis.
        if self.regularization == "laplacian":
            Xa, Xb = compute_basis_laplacians(self.mats)
            self.reg = RegularizationOperator(
                mode="laplacian", coeff=self.nu, Xa=Xa, Xb=Xb,
            )
        else:
            Xa, Xb = compute_basis_bilaplacians(self.mats)
            self.reg = RegularizationOperator(
                mode="bilaplacian", coeff=self.epsilon, Xa=Xa, Xb=Xb,
            )

        # 4. Quadrature and nonlocal operators (kernel is time-independent).
        quad_points, quad_weights = setup_quadrature(order=self.quad_order)
        self.quad_points  = quad_points
        self.quad_weights = quad_weights
        self.nl_ops = compute_nonlocal_operators(
            mats=self.mats,
            quad_points=quad_points,
            quad_weights=quad_weights,
            domain=self.domain,
            kernel=self.kernel,
        )

        self._setup_done = True

    # -------------------------------------------------------------- run loop

    def run(self) -> SolverResult:
        """Set up (if needed), build IC/boundary/source data, and time-step."""
        self.setup()

        # Time grid (closed, includes t_final).
        n_steps = int(round(self.t_final / self.dt))
        times = np.linspace(0.0, n_steps * self.dt, n_steps + 1)

        # IC, boundary, source — branched by mode.
        X0, boundary_history, F = self._setup_data(times)

        # Drivers for time_integration.integrate.
        boundary_fn = lambda t, _bh=boundary_history, _ts=times: \
            _interpolate_boundary(t, _ts, _bh)

        source_fn   = lambda step, _F=F: _F[step]  # source array indexed by step

        # Optional progress callback.
        progress_callback = self._make_progress_callback(n_steps) if self.progress else None

        report = integrate(
            X0=X0,
            times=times,
            boundary_fn=boundary_fn,
            source_fn=source_fn,
            mats=self.mats,
            nl_ops=self.nl_ops,
            reg=self.reg,
            velocity=self.velocity,
            reaction=self.reaction,
            bdf_max_order=self.bdf_max_order,
            newton_kwargs=self.newton_kwargs,
            progress_callback=progress_callback,
        )

        # Pack into SolverResult.
        bdf_orders   = np.array([s.bdf_order      for s in report.steps], dtype=int)
        newton_iters = np.array([s.newton.iters   for s in report.steps], dtype=int)
        gmres_iters  = np.array([s.newton.gmres_iters_total for s in report.steps], dtype=int)

        err_inf, err_l2 = (None, None)
        if self.mode == "validation":
            err_inf, err_l2 = self._compute_errors(times, report.history, boundary_history)

        return SolverResult(
            times=times,
            history=report.history,
            boundary_history=boundary_history,
            bdf_orders=bdf_orders,
            newton_iters=newton_iters,
            gmres_iters=gmres_iters,
            err_inf=err_inf,
            err_l2=err_l2,
        )

    # ----------------------------------------------------------- helpers

    def _setup_data(
        self, times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build initial coefficients X0, boundary trace history, and source
        tensor F[step, n, 2].

        Returns
        -------
        X0 : (2, n) initial interior coefficients.
        boundary_history : (n_steps+1, 2, n_prime).
        F : (n_steps+1, n, 2) source-term tensor.
        """
        n        = self.mats.n
        n_prime  = self.mats.n_prime
        n_steps  = len(times) - 1

        xi = self.interior_points[:, 0]
        yi = self.interior_points[:, 1]
        xb = self.boundary_points[:, 0]
        yb = self.boundary_points[:, 1]

        if self.mode == "validation":
            # Source: precomputed analytically with joblib parallelism.
            F, exact = precompute_source_terms(
                interior_points=self.interior_points,
                domain=self.domain,
                velocity=self.velocity,
                reaction=self.reaction,
                times=times,
                regularization=self.regularization,
                nu=self.nu,
                epsilon=self.epsilon,
                quad_points=self.quad_points,
                quad_weights=self.quad_weights,
                kernel=self.kernel,
                exact=DefaultManufacturedSolution(),
                n_jobs=self.n_jobs,
            )
            self.exact = exact

            # IC: u_h(0, x_i) = X0[d, i] = u_d(0, x_i).
            u0_int = exact.u(0.0, xi, yi)                       # (n, 2)
            X0 = u0_int.T.copy()                                # (2, n)

            # Boundary history: (n_steps+1, 2, n_prime).
            boundary_history = np.empty((n_steps + 1, 2, n_prime))
            for k, t in enumerate(times):
                u_b = exact.u(float(t), xb, yb)                 # (n_prime, 2)
                boundary_history[k] = u_b.T

        else:
            # --- Simulation mode ---
            # Source: zero unless user provided one.
            F = np.zeros((n_steps + 1, n, 2), dtype=float)
            if self.source_term_user is not None:
                for k, t in enumerate(times):
                    F[k] = np.asarray(
                        self.source_term_user(float(t), xi, yi), dtype=float,
                    )

            # IC.
            u0_int = np.asarray(self.initial_condition_user(xi, yi), dtype=float)
            if u0_int.shape != (n, 2):
                raise ValueError(
                    f"initial_condition must return shape (n, 2) = ({n}, 2); "
                    f"got {u0_int.shape}."
                )
            X0 = u0_int.T.copy()

            # Boundary history.
            boundary_history = np.empty((n_steps + 1, 2, n_prime))
            if self.boundary_condition_user is None:
                boundary_history.fill(0.0)
            else:
                for k, t in enumerate(times):
                    g = np.asarray(
                        self.boundary_condition_user(float(t), xb, yb), dtype=float,
                    )
                    if g.shape != (n_prime, 2):
                        raise ValueError(
                            f"boundary_condition must return shape (n_prime, 2) "
                            f"= ({n_prime}, 2); got {g.shape}."
                        )
                    boundary_history[k] = g.T

        return X0, boundary_history, F

    def _compute_errors(
        self,
        times: np.ndarray,
        history: np.ndarray,           # (n_steps+1, 2, n)
        boundary_history: np.ndarray,  # (n_steps+1, 2, n_prime)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Relative L^inf and L^2 errors at the interior nodes for every time
        step. The boundary nodes carry exact data so they don't enter the
        relative-error calculation.
        """
        if self.exact is None:
            raise RuntimeError("exact solution not set; this is a bug.")

        xi = self.interior_points[:, 0]
        yi = self.interior_points[:, 1]

        n_t = len(times)
        err_inf = np.empty(n_t, dtype=float)
        err_l2  = np.empty(n_t, dtype=float)

        for k, t in enumerate(times):
            u_exact = self.exact.u(float(t), xi, yi)        # (n, 2)
            u_num   = history[k].T                          # (n, 2)
            diff    = u_num - u_exact

            # Norm pooled across components and nodes; divide by exact-norm
            # to give relative error.
            denom_inf = max(float(np.max(np.abs(u_exact))), 1.0e-14)
            denom_l2  = max(float(np.linalg.norm(u_exact)), 1.0e-14)

            err_inf[k] = float(np.max(np.abs(diff))) / denom_inf
            err_l2[k]  = float(np.linalg.norm(diff)) / denom_l2

        return err_inf, err_l2

    def _make_progress_callback(self, n_steps: int):
        every = max(1, n_steps // 50)

        def cb(step, t, newton_report):
            if step % every == 0 or step == n_steps:
                print(
                    f"[step {step:>5d}/{n_steps:>5d}  t={t:8.4f}]  "
                    f"newton_iters={newton_report.iters:2d}  "
                    f"gmres_iters={newton_report.gmres_iters_total:3d}  "
                    f"residual={newton_report.final_residual:.2e}"
                )

        return cb

    # -------------------------------------------------------------- plotting

    def plot_solution(
        self,
        result: SolverResult,
        save_dir: Optional[str] = None,
        which: str = "final",
    ):
        """
        Plot contours and 3-D surfaces of the numerical solution (and exact
        solution and pointwise error, in validation mode). Imports
        matplotlib lazily so that ``import rbf_nonlocal`` stays cheap.

        Parameters
        ----------
        result : SolverResult
        save_dir : str, optional
            If given, figures are written here. If ``None``, ``plt.show()``
            is called instead.
        which : str
            ``"final"`` or ``"all"``; the latter saves a frame per time
            step (slow, useful for movies).

        Notes
        -----
        Implementation lives in :mod:`rbf_nonlocal.plotting`.
        """
        from . import plotting
        plotting.plot_solution(self, result, save_dir=save_dir, which=which)


# ---------------------------------------------------------------------------
# Boundary interpolation utility
# ---------------------------------------------------------------------------

def _interpolate_boundary(
    t: float, times: np.ndarray, history: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate the boundary trace history at time ``t``.

    The integrator may evaluate ``boundary_fn(t)`` at exactly one of the
    pre-tabulated times, in which case this is exact; otherwise (for
    sub-step evaluations or rounding) it linearly interpolates the two
    nearest pre-tabulated traces. With the default uniform time grid this
    branch never fires.
    """
    if t <= times[0]:
        return history[0]
    if t >= times[-1]:
        return history[-1]

    idx = int(np.searchsorted(times, t))
    t0, t1 = times[idx - 1], times[idx]
    w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return (1.0 - w) * history[idx - 1] + w * history[idx]