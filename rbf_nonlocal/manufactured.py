"""
Manufactured solution and analytical source terms.

This module supplies the manufactured solution used in the validation
experiments of Section 6 of the paper,

.. math::
    u_1(t, x, y) &= \\bigl(1 + \\tanh(0.5\\,t)\\bigr)\\,\\sin(\\pi x)\\,\\sin(\\pi y),\\\\
    u_2(t, x, y) &= \\bigl(1 + 0.5\\,\\tanh(0.3\\,t)\\bigr)\\,\\cos(\\pi x)\\,\\cos(\\pi y),

together with all the derivatives (in :math:`t`, in :math:`x`, the Laplacian,
and the bi-Laplacian) needed to assemble the source term :math:`f` that
makes ``u`` an exact solution of the system.

The high-level entry point is :func:`precompute_source_terms`, which loops
over the time grid and returns an array ``F[k, i, d]`` of shape
``(n_steps + 1, n_interior, 2)`` such that

.. math::
    \\partial_t u + \\nabla \\cdot (V(\\mathcal{W}_u) \\otimes u)
    + \\mathcal{L} u - \\mathcal{N}(u) = f,

with :math:`\\mathcal{L} = -\\nu\\,\\Delta` (``regularization='laplacian'``)
or :math:`\\mathcal{L} = \\varepsilon\\,\\Delta^2`
(``regularization='bilaplacian'``).

Per-step work is parallelised with joblib; pass ``n_jobs=1`` to run serially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from .domains import Domain
from .nonlocal_ops import GaussianKernel, Kernel
from .reaction import Reaction
from .velocity import Velocity

__all__ = [
    "ManufacturedSolution",
    "DefaultManufacturedSolution",
    "precompute_source_terms",
]


# ---------------------------------------------------------------------------
# Abstract manufactured solution
# ---------------------------------------------------------------------------

class ManufacturedSolution:
    """
    Abstract base class for a smooth manufactured solution
    :math:`u(t, x, y) \\in \\mathbb{R}^2`.

    Concrete subclasses implement :meth:`u`, :meth:`dudt`, :meth:`gradient`,
    :meth:`laplacian`, and :meth:`bilaplacian`. All routines are vectorised
    over the spatial axis.
    """

    name: str = "abstract"

    def u(self, t, x, y):                # pragma: no cover - abstract
        raise NotImplementedError

    def dudt(self, t, x, y):             # pragma: no cover - abstract
        raise NotImplementedError

    def gradient(self, t, x, y):         # pragma: no cover - abstract
        raise NotImplementedError

    def laplacian(self, t, x, y):        # pragma: no cover - abstract
        raise NotImplementedError

    def bilaplacian(self, t, x, y):      # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The manufactured solution from the paper
# ---------------------------------------------------------------------------

@dataclass
class DefaultManufacturedSolution(ManufacturedSolution):
    r"""
    The manufactured solution used in Section 6 of the paper,

    .. math::
        u_1(t, x, y) &= \bigl(1 + \tanh(0.5\,t)\bigr)\,\sin(\pi x)\,\sin(\pi y),\\
        u_2(t, x, y) &= \bigl(1 + 0.5\,\tanh(0.3\,t)\bigr)\,\cos(\pi x)\,\cos(\pi y).

    The space and time factors are separable, so all spatial derivatives
    inherit the :math:`\sin` / :math:`\cos` structure and the eigenvalues
    :math:`-\pi^2` (Laplacian, single component) and :math:`+\pi^4` per
    factor (bi-Laplacian).
    """

    name: str = "default"

    # Time-dependent envelopes
    def _T1(self, t): return 1.0 + np.tanh(0.5 * t)
    def _T2(self, t): return 1.0 + 0.5 * np.tanh(0.3 * t)
    def _T1_dot(self, t): return 0.5 * (1.0 - np.tanh(0.5 * t) ** 2)
    def _T2_dot(self, t): return 0.15 * (1.0 - np.tanh(0.3 * t) ** 2)

    # ---- value u(t, x, y) ----
    def u(self, t, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        s = np.sin(np.pi * x) * np.sin(np.pi * y)
        c = np.cos(np.pi * x) * np.cos(np.pi * y)
        out = np.empty(x.shape + (2,))
        out[..., 0] = self._T1(t) * s
        out[..., 1] = self._T2(t) * c
        return out

    # ---- time derivative ∂_t u ----
    def dudt(self, t, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        s = np.sin(np.pi * x) * np.sin(np.pi * y)
        c = np.cos(np.pi * x) * np.cos(np.pi * y)
        out = np.empty(x.shape + (2,))
        out[..., 0] = self._T1_dot(t) * s
        out[..., 1] = self._T2_dot(t) * c
        return out

    # ---- spatial gradient ∇u, shape (..., 2_components, 2_xy) ----
    def gradient(self, t, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pi = np.pi
        sx, cx = np.sin(pi * x), np.cos(pi * x)
        sy, cy = np.sin(pi * y), np.cos(pi * y)
        T1, T2 = self._T1(t), self._T2(t)

        grad = np.empty(x.shape + (2, 2))
        # u_1 = T1 sin(pi x) sin(pi y)
        grad[..., 0, 0] = T1 * pi * cx * sy        # ∂x u_1
        grad[..., 0, 1] = T1 * pi * sx * cy        # ∂y u_1
        # u_2 = T2 cos(pi x) cos(pi y)
        grad[..., 1, 0] = -T2 * pi * sx * cy       # ∂x u_2
        grad[..., 1, 1] = -T2 * pi * cx * sy       # ∂y u_2
        return grad

    # ---- Laplacian Δu, shape (..., 2) ----
    def laplacian(self, t, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        s = np.sin(np.pi * x) * np.sin(np.pi * y)
        c = np.cos(np.pi * x) * np.cos(np.pi * y)
        # Δ[sin(pi x) sin(pi y)] = -2 pi^2 sin(pi x) sin(pi y), and likewise for cos.
        out = np.empty(x.shape + (2,))
        out[..., 0] = -2.0 * np.pi**2 * self._T1(t) * s
        out[..., 1] = -2.0 * np.pi**2 * self._T2(t) * c
        return out

    # ---- bi-Laplacian Δ²u, shape (..., 2) ----
    def bilaplacian(self, t, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        s = np.sin(np.pi * x) * np.sin(np.pi * y)
        c = np.cos(np.pi * x) * np.cos(np.pi * y)
        # Δ² = (Δ)² applied to a -2 pi^2 eigenfunction gives +4 pi^4.
        out = np.empty(x.shape + (2,))
        out[..., 0] = 4.0 * np.pi**4 * self._T1(t) * s
        out[..., 1] = 4.0 * np.pi**4 * self._T2(t) * c
        return out


# ---------------------------------------------------------------------------
# Single-step source-term assembly
# ---------------------------------------------------------------------------

def _source_at_time(
    t: float,
    interior_points: np.ndarray,
    quad_points: np.ndarray,
    quad_weights_eff: np.ndarray,
    domain: Domain,
    velocity: Velocity,
    reaction: Reaction,
    kernel: Kernel,
    nu: Optional[np.ndarray],
    epsilon: Optional[float],
    regularization: str,
    exact: ManufacturedSolution,
) -> np.ndarray:
    r"""
    Compute :math:`f(t, x_i)` for a single instant. This is what the joblib
    worker calls.

    The convective divergence is computed in product form,

    .. math::
        \nabla \cdot (V \otimes u) = (V \cdot \nabla u)_d + u_d (\nabla \cdot V),

    where :math:`\nabla V_d = J_V(\mathcal{W}_u) \cdot \nabla \mathcal{W}_u`
    via the chain rule.
    """
    xi = interior_points[:, 0]
    yi = interior_points[:, 1]

    # 1. Manufactured u, ∂_t u, ∇u, and the regulariser at the interior nodes.
    u_int    = exact.u(t, xi, yi)                  # (n, 2)
    dudt_int = exact.dudt(t, xi, yi)               # (n, 2)
    grad_u   = exact.gradient(t, xi, yi)           # (n, 2, 2): [..., d, dim]

    if regularization == "laplacian":
        Lop_u = exact.laplacian(t, xi, yi)         # (n, 2)
    elif regularization == "bilaplacian":
        Lop_u = exact.bilaplacian(t, xi, yi)       # (n, 2)
    else:
        raise ValueError(
            f"regularization must be 'laplacian' or 'bilaplacian' "
            f"(got {regularization!r})."
        )

    # 2. Nonlocal density Wu and its spatial gradient at the interior nodes.
    #    Wu_i,d  = ∫ k(x_i, y) u_d(y) dy
    #    ∇Wu_i,d = ∫ ∇_x k(x_i, y) u_d(y) dy
    u_quad = exact.u(t, quad_points[:, 0], quad_points[:, 1])    # (Q, 2)
    diff   = interior_points[:, None, :] - quad_points[None, :, :]   # (n, Q, 2)
    k_vals = kernel(t, diff)                                     # (n, Q)
    grad_k = kernel.gradient_x(t, diff)                          # (n, Q, 2)

    weighted_u = u_quad * quad_weights_eff[:, None]              # (Q, 2)
    Wu = k_vals @ weighted_u                                     # (n, 2)
    grad_Wu = np.einsum("iqd,qc->icd", grad_k, weighted_u)       # (n, 2, 2)

    # 3. Velocity, its Jacobian, and ∇·V via the chain rule.
    V    = velocity.field(Wu)                                    # (n, 2)
    JV   = velocity.jacobian(Wu)                                 # (n, 2, 2): [..., d, c]
    # ∇V[i, d, dim] = JV[i, d, c] * ∇Wu[i, c, dim]
    grad_V = np.einsum("idc,icj->idj", JV, grad_Wu)              # (n, 2, 2)
    div_V  = grad_V[..., 0, 0] + grad_V[..., 1, 1]               # (n,)

    # 4. Convective term ∇·(V ⊗ u) = (V·∇u) + u (∇·V).
    Vdotgrad_u = np.einsum("ij,idj->id", V, grad_u)              # (n, 2)
    conv = Vdotgrad_u + u_int * div_V[:, None]                   # (n, 2)

    # 5. Reaction.
    Nval = reaction.field(u_int)                                 # (n, 2)

    # 6. Assemble f.
    f = dudt_int + conv - Nval
    if regularization == "laplacian":
        # ∂_t u - ν Δu + ∇·(V⊗u) = N + f  →  f = ∂_t u + ∇·(V⊗u) - N - ν Δu
        f -= np.einsum("dc,ic->id", np.asarray(nu, dtype=float), Lop_u)
    else:  # bilaplacian
        f += float(epsilon) * Lop_u

    return f


# ---------------------------------------------------------------------------
# Public driver — loops over time and parallelises
# ---------------------------------------------------------------------------

def precompute_source_terms(
    interior_points: np.ndarray,
    domain: Domain,
    velocity: Velocity,
    reaction: Reaction,
    times: np.ndarray,
    *,
    regularization: str,
    nu: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
    quad_points: Optional[np.ndarray] = None,
    quad_weights: Optional[np.ndarray] = None,
    kernel: Optional[Kernel] = None,
    exact: Optional[ManufacturedSolution] = None,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, ManufacturedSolution]:
    """
    Compute the analytical source term :math:`f(t_k, x_i)` at every node and
    every time step on the grid.

    Parameters
    ----------
    interior_points : np.ndarray, shape ``(n, 2)``
    domain : Domain
        Geometry whose indicator restricts the nonlocal integral.
    velocity : Velocity
    reaction : Reaction
    times : array-like, shape ``(n_steps + 1,)``
        Time grid (typically ``np.arange(0, t_final + dt, dt)``).
    regularization : ``"laplacian"`` or ``"bilaplacian"``
        Selects which higher-order operator is in the PDE and therefore
        which derivative of the manufactured solution to subtract from ``f``.
    nu : array-like, shape ``(2, 2)``, required if ``regularization='laplacian'``
        The diffusion matrix.
    epsilon : float, required if ``regularization='bilaplacian'``
        The hyper-viscosity coefficient.
    quad_points, quad_weights : np.ndarray, optional
        If not provided, a default 30 x 30 Gauss-Legendre rule on
        ``[0, 1]^2`` is used (matching the paper).
    kernel : Kernel, optional
        Defaults to ``GaussianKernel(alpha=2.0)``.
    exact : ManufacturedSolution, optional
        Defaults to :class:`DefaultManufacturedSolution`.
    n_jobs : int
        Number of parallel jobs (joblib). ``-1`` uses all cores; ``1``
        runs sequentially.

    Returns
    -------
    F : np.ndarray, shape ``(n_steps + 1, n, 2)``
    exact : ManufacturedSolution
        The same manufactured solution object, returned for use in error
        analysis (so callers don't have to instantiate it twice).
    """
    if regularization == "laplacian" and nu is None:
        raise ValueError("regularization='laplacian' requires nu (a 2x2 array).")
    if regularization == "bilaplacian" and epsilon is None:
        raise ValueError("regularization='bilaplacian' requires epsilon (float).")

    interior_points = np.asarray(interior_points, dtype=float)
    times = np.asarray(times, dtype=float)
    if nu is not None:
        nu = np.asarray(nu, dtype=float)

    if quad_points is None or quad_weights is None:
        from .nonlocal_ops import setup_quadrature
        quad_points, quad_weights = setup_quadrature(order=30)

    if kernel is None:
        kernel = GaussianKernel(alpha=2.0)
    if exact is None:
        exact = DefaultManufacturedSolution()

    # Premultiply weights by the indicator: a vector of length Q that is
    # zero outside the domain and (weight) inside it.
    indicator = domain.contains(quad_points[:, 0], quad_points[:, 1]).astype(float)
    quad_weights_eff = np.asarray(quad_weights, dtype=float) * indicator

    F_list = Parallel(n_jobs=n_jobs)(
        delayed(_source_at_time)(
            float(t),
            interior_points, quad_points, quad_weights_eff,
            domain, velocity, reaction, kernel,
            nu, epsilon, regularization, exact,
        )
        for t in times
    )
    F = np.stack(F_list, axis=0)
    return F, exact