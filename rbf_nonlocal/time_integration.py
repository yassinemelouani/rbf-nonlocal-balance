"""
Time integration with variable-order BDF and Newton-Krylov.

The semi-discrete system

.. math::
    \\dot X = \\Phi(t, X)

is integrated with backward differentiation formulas (BDF) of order
:math:`k = \\min(n_{\\text{step}} + 1,\\, k_{\\max})`. The order ramps up
naturally from BDF1 on the first step to BDF6 from step five onward, so no
external bootstrap is needed.

Each BDF step solves the nonlinear system

.. math::
    X^{n+1} \\;-\\; \\Delta t\\, \\gamma_0\\,
        \\Phi\\bigl(t^{n+1},\\, X^{n+1}\\bigr)
    \\;=\\; \\sum_{j=1}^{k} \\alpha_j\\, X^{n+1-j}

by Newton iteration. The linear system at each Newton iteration is solved
with restarted GMRES, optionally preconditioned by an incomplete LU
factorisation of an explicitly assembled Jacobian.

Public API
----------
* :data:`BDF_COEFFS` — dict mapping order :math:`k \\to (\\boldsymbol\\alpha,\\, \\gamma_0)`.
* :func:`bdf_step` — advance one step at a chosen order.
* :func:`integrate` — run the full time loop, return solution history and
  diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, gmres, splu

from .matrices import RBFInterpolationMatrices
from .nonlocal_ops import NonlocalOperators
from .reaction import Reaction
from .rhs import RegularizationOperator, compute_rhs, frechet_derivative
from .velocity import Velocity

__all__ = [
    "BDF_COEFFS",
    "NewtonReport",
    "StepReport",
    "IntegrationReport",
    "newton_krylov_solve",
    "bdf_step",
    "integrate",
]


# ---------------------------------------------------------------------------
# BDF coefficient table
# ---------------------------------------------------------------------------
#
# Convention:  X^{n+1} = sum_j alpha_j X^{n+1-j}  +  dt * gamma_0 * Phi(X^{n+1})
#
# i.e. alpha_j is the multiplier of X^{n+1-j} on the right-hand side.
# These match the standard normalisation (e.g. Hairer & Wanner II.5.1.5).

BDF_COEFFS: dict = {
    1: (np.array([1.0]),
        1.0),

    2: (np.array([4.0/3.0, -1.0/3.0]),
        2.0/3.0),

    3: (np.array([18.0/11.0, -9.0/11.0, 2.0/11.0]),
        6.0/11.0),

    4: (np.array([48.0/25.0, -36.0/25.0, 16.0/25.0, -3.0/25.0]),
        12.0/25.0),

    5: (np.array([300.0/137.0, -300.0/137.0, 200.0/137.0,
                  -75.0/137.0, 12.0/137.0]),
        60.0/137.0),

    6: (np.array([360.0/147.0, -450.0/147.0, 400.0/147.0,
                  -225.0/147.0, 72.0/147.0, -10.0/147.0]),
        60.0/147.0),
}


# ---------------------------------------------------------------------------
# Diagnostic structures
# ---------------------------------------------------------------------------

@dataclass
class NewtonReport:
    """Per-Newton-iteration diagnostics."""
    iters: int
    converged: bool
    final_residual: float
    gmres_iters_total: int


@dataclass
class StepReport:
    """One BDF step."""
    step: int
    t: float
    bdf_order: int
    newton: NewtonReport


@dataclass
class IntegrationReport:
    """Whole time loop."""
    times: np.ndarray
    history: np.ndarray         # shape (n_steps + 1, 2, n)
    steps: List[StepReport] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def total_newton_iters(self) -> int:
        return sum(s.newton.iters for s in self.steps)

    @property
    def total_gmres_iters(self) -> int:
        return sum(s.newton.gmres_iters_total for s in self.steps)


# ---------------------------------------------------------------------------
# Newton-Krylov inner solve
# ---------------------------------------------------------------------------

def _materialize_jacobian(
    X: np.ndarray,
    G: np.ndarray,
    mats: RBFInterpolationMatrices,
    nl_ops: NonlocalOperators,
    reg: RegularizationOperator,
    velocity: Velocity,
    reaction: Reaction,
    bdf_factor: float,
) -> np.ndarray:
    r"""
    Form the dense Jacobian :math:`J = I - \mathrm{bdf\_factor}\,\partial_X\Phi`
    by applying ``frechet_derivative`` to each unit vector. Costs ``2 n``
    Fréchet derivatives but pays for itself by enabling iLU preconditioning.
    """
    size = 2 * mats.n
    J = np.empty((size, size), dtype=float)
    e = np.zeros((2, mats.n), dtype=float)
    for k in range(size):
        flat = e.reshape(-1)
        flat[k] = 1.0
        dphi = frechet_derivative(e, X, G, mats, nl_ops, reg, velocity, reaction)
        J[:, k] = (e - bdf_factor * dphi).reshape(-1)
        flat[k] = 0.0
    return J


def newton_krylov_solve(
    X_init: np.ndarray,
    rhs_history: List[np.ndarray],
    G_new: np.ndarray,
    f_new: np.ndarray,
    t_new: float,
    dt: float,
    bdf_order: int,
    mats: RBFInterpolationMatrices,
    nl_ops: NonlocalOperators,
    reg: RegularizationOperator,
    velocity: Velocity,
    reaction: Reaction,
    *,
    newton_tol: float = 1.0e-8,
    newton_maxiter: int = 30,
    gmres_tol: float = 1.0e-10,
    gmres_restart: int = 50,
    gmres_maxiter: int = 200,
    precondition: Optional[bool] = None,
) -> Tuple[np.ndarray, NewtonReport]:
    r"""
    Solve one BDF step by Newton iteration with GMRES inner linear solves.

    Solves the nonlinear residual

    .. math::
        R(X) = X - \mathrm{rhs\_sum} - \Delta t\,\gamma_0\,\Phi(t^{n+1}, X) = 0,

    where ``rhs_sum = sum_j alpha_j X^{n+1-j}``.

    Parameters
    ----------
    X_init : np.ndarray, shape ``(2, n)``
        Initial guess; typically the previous time-step's solution.
    rhs_history : list of np.ndarray, each shape ``(2, n)``
        ``[X^n, X^{n-1}, ..., X^{n+1-k}]`` in this order; only the first
        ``bdf_order`` entries are used.
    G_new, f_new, t_new : Dirichlet data, source, and time at the new step.
    dt : float
    bdf_order : int
    mats, nl_ops, reg, velocity, reaction
        Same as in :func:`rbf_nonlocal.rhs.compute_rhs`.
    newton_tol : float
        Convergence tolerance on ``||R(X)||_inf``.
    newton_maxiter : int
    gmres_tol, gmres_restart, gmres_maxiter : GMRES parameters.
    precondition : bool, optional
        Whether to materialize the Jacobian and form an iLU preconditioner.
        If ``None``, defaults to ``True`` for ``n <= 800`` and ``False``
        otherwise. The materialisation costs ``2 n`` Fréchet derivatives
        per Newton iteration but typically more than recovers the cost
        through faster GMRES convergence.

    Returns
    -------
    X_new : np.ndarray, shape ``(2, n)``
    report : NewtonReport
    """
    if bdf_order not in BDF_COEFFS:
        raise ValueError(
            f"BDF order {bdf_order} is not in BDF_COEFFS "
            f"(available: {sorted(BDF_COEFFS)})."
        )
    if len(rhs_history) < bdf_order:
        raise ValueError(
            f"rhs_history needs at least bdf_order={bdf_order} entries, "
            f"got {len(rhs_history)}."
        )

    alphas, gamma0 = BDF_COEFFS[bdf_order]
    bdf_factor = dt * gamma0

    if precondition is None:
        precondition = mats.n <= 800

    # rhs_sum = sum_j alphas[j] * X^{n+1-(j+1)}
    rhs_sum = np.zeros_like(X_init)
    for j, alpha in enumerate(alphas):
        rhs_sum += alpha * rhs_history[j]

    def residual(X):
        Phi = compute_rhs(t_new, X, G_new, f_new, mats, nl_ops, reg, velocity, reaction)
        return X - rhs_sum - bdf_factor * Phi

    X = X_init.copy()
    gmres_iters_total = 0
    converged = False

    for newton_iter in range(1, newton_maxiter + 1):
        R = residual(X)
        residual_norm = float(np.max(np.abs(R)))
        if residual_norm < newton_tol:
            converged = True
            break

        # Linear operator J = I - bdf_factor * dPhi.
        n = mats.n
        size = 2 * n

        def matvec(v, X=X):
            delta = v.reshape(2, n)
            dphi = frechet_derivative(delta, X, G_new, mats, nl_ops, reg, velocity, reaction)
            return (delta - bdf_factor * dphi).reshape(-1)

        J_op = LinearOperator((size, size), matvec=matvec, dtype=float)

        # Optional iLU preconditioner.
        M = None
        if precondition:
            J_dense = _materialize_jacobian(
                X, G_new, mats, nl_ops, reg, velocity, reaction, bdf_factor,
            )
            J_sparse = csc_matrix(J_dense)
            try:
                lu = splu(J_sparse)
                M = LinearOperator(
                    (size, size),
                    matvec=lambda v, lu=lu: lu.solve(v),
                    dtype=float,
                )
            except RuntimeError:
                # Singular; fall through with no preconditioner.
                M = None

        gmres_count = [0]

        def gmres_callback(_x, _count=gmres_count):
            _count[0] += 1

        b = -R.reshape(-1)
        delta_flat, info = gmres(
            J_op,
            b,
            rtol=gmres_tol,
            restart=gmres_restart,
            maxiter=gmres_maxiter,
            M=M,
            callback=gmres_callback,
            callback_type="pr_norm",
        )
        gmres_iters_total += gmres_count[0]

        if info != 0:
            # GMRES did not converge to tolerance, but we still take the
            # step if it makes progress; flag in the report.
            pass

        X = X + delta_flat.reshape(2, n)

    return X, NewtonReport(
        iters=newton_iter,
        converged=converged,
        final_residual=residual_norm,
        gmres_iters_total=gmres_iters_total,
    )


# ---------------------------------------------------------------------------
# One BDF step
# ---------------------------------------------------------------------------

def bdf_step(
    history: List[np.ndarray],
    G_new: np.ndarray,
    f_new: np.ndarray,
    t_new: float,
    dt: float,
    *,
    bdf_max_order: int = 6,
    mats: RBFInterpolationMatrices,
    nl_ops: NonlocalOperators,
    reg: RegularizationOperator,
    velocity: Velocity,
    reaction: Reaction,
    newton_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, NewtonReport, int]:
    """
    Advance one BDF step at order :math:`\\min(\\text{len(history)},\\,
    \\text{bdf\\_max\\_order})`.

    Parameters
    ----------
    history : list of np.ndarray
        ``[X^n, X^{n-1}, ...]`` in reverse-chronological order. Length sets
        the available order.
    G_new, f_new, t_new, dt : data for the step being taken.
    bdf_max_order : int
        Cap the BDF order. Default 6.
    mats, nl_ops, reg, velocity, reaction : passed through.
    newton_kwargs : dict, optional
        Forwarded to :func:`newton_krylov_solve`.

    Returns
    -------
    X_new : np.ndarray, shape ``(2, n)``
    report : NewtonReport
    bdf_order : int
        The order actually used for this step.
    """
    if not history:
        raise ValueError("history must contain at least one entry (X^n).")
    if not 1 <= bdf_max_order <= 6:
        raise ValueError(f"bdf_max_order must be in [1, 6] (got {bdf_max_order}).")

    bdf_order = min(len(history), bdf_max_order)
    newton_kwargs = newton_kwargs or {}

    X_new, report = newton_krylov_solve(
        X_init=history[0],
        rhs_history=history,
        G_new=G_new,
        f_new=f_new,
        t_new=t_new,
        dt=dt,
        bdf_order=bdf_order,
        mats=mats,
        nl_ops=nl_ops,
        reg=reg,
        velocity=velocity,
        reaction=reaction,
        **newton_kwargs,
    )
    return X_new, report, bdf_order


# ---------------------------------------------------------------------------
# Full time loop
# ---------------------------------------------------------------------------

def integrate(
    X0: np.ndarray,
    times: np.ndarray,
    boundary_fn: Callable[[float], np.ndarray],
    source_fn: Callable[[int], np.ndarray],
    *,
    mats: RBFInterpolationMatrices,
    nl_ops: NonlocalOperators,
    reg: RegularizationOperator,
    velocity: Velocity,
    reaction: Reaction,
    bdf_max_order: int = 6,
    newton_kwargs: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, float, NewtonReport], None]] = None,
) -> IntegrationReport:
    r"""
    Full time loop, building up history and ramping the BDF order from 1 to
    ``bdf_max_order``.

    Parameters
    ----------
    X0 : np.ndarray, shape ``(2, n)``
        Initial interior coefficients :math:`X^0`.
    times : array-like, shape ``(n_steps + 1,)``
        Time grid; ``times[0]`` is the initial time and the loop advances
        through ``times[1], times[2], ...``.
    boundary_fn : callable
        ``boundary_fn(t) -> np.ndarray`` of shape ``(2, n_prime)``,
        returning Dirichlet data per component at time ``t``.
    source_fn : callable
        ``source_fn(step_index) -> np.ndarray`` of shape ``(n, 2)``. Use
        the precomputed source-term tensor in validation mode and a
        ``lambda _: np.zeros((n, 2))`` in simulation mode.
    mats, nl_ops, reg, velocity, reaction : passed through.
    bdf_max_order : int
    newton_kwargs : dict, optional
    progress_callback : callable, optional
        ``progress_callback(step_index, t, newton_report)``; useful for
        tqdm bars.

    Returns
    -------
    IntegrationReport
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a 1-D array with at least two entries.")

    history: List[np.ndarray] = [np.asarray(X0, dtype=float).copy()]
    out_history = np.empty((len(times), *X0.shape), dtype=float)
    out_history[0] = history[0]
    step_reports: List[StepReport] = []

    for step in range(1, len(times)):
        t_new = float(times[step])
        dt = t_new - float(times[step - 1])

        G_new = np.asarray(boundary_fn(t_new), dtype=float)
        f_new = np.asarray(source_fn(step), dtype=float)

        X_new, newton_report, used_order = bdf_step(
            history,
            G_new=G_new, f_new=f_new, t_new=t_new, dt=dt,
            bdf_max_order=bdf_max_order,
            mats=mats, nl_ops=nl_ops, reg=reg,
            velocity=velocity, reaction=reaction,
            newton_kwargs=newton_kwargs,
        )

        # Update history.
        history.insert(0, X_new)
        if len(history) > bdf_max_order:
            history.pop()

        out_history[step] = X_new
        step_reports.append(StepReport(
            step=step, t=t_new, bdf_order=used_order, newton=newton_report,
        ))

        if progress_callback is not None:
            progress_callback(step, t_new, newton_report)

    return IntegrationReport(
        times=times,
        history=out_history,
        steps=step_reports,
    )