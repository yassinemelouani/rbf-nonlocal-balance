"""
Nonlocal operator matrices for the RBF collocation scheme.

For the RBF interpolant :math:`u_h = a^T X + b^T \\tilde G`, the nonlocal term

.. math::
    \\mathcal{W}_{u_h}(t, x) = \\int_\\Omega k(t, x, y)\\, u_h(t, y)\\, dy

evaluated at the interior collocation nodes admits the matrix representation

.. math::
    \\mathcal{W}_{u_h}(t, x_i) \\;=\\; (K_a)_{ij}\\, X_j + (K_b)_{ij}\\, \\tilde G_j ,

with companion matrices ``dKa_dx, dKa_dy, dKb_dx, dKb_dy`` for the two
spatial gradient components.

This module assembles all six matrices in one pass via tensor-product
Gauss-Legendre quadrature on a bounding box, restricted to the domain
through its indicator function.

The interaction kernel ``k(t, x, y)`` is supplied as a :class:`Kernel`
instance. A :class:`GaussianKernel` is provided that reproduces the choice
:math:`k(x, y) = \\exp(-\\alpha\\|x-y\\|^2)` used throughout the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from .domains import Domain
from .matrices import RBFInterpolationMatrices, evaluate_basis_functions

__all__ = [
    "Kernel",
    "GaussianKernel",
    "NonlocalOperators",
    "setup_quadrature",
    "compute_nonlocal_operators",
]


# ---------------------------------------------------------------------------
# Kernel abstraction
# ---------------------------------------------------------------------------

class Kernel:
    """
    Base class for nonlocal interaction kernels :math:`k(t, x, y)`.

    Subclasses implement two methods:

    * :meth:`__call__(t, diff)` — kernel value, with ``diff = x - y`` an
      array of arbitrary shape ending in ``(..., 2)``. Must return an array
      of shape ``diff.shape[:-1]``.
    * :meth:`gradient_x(t, diff)` — gradient of :math:`k` with respect to
      its first spatial argument :math:`x`, of shape ``diff.shape``.

    Both methods must be vectorised in their leading axes; no Python loop
    over ``i`` or ``q`` is acceptable.
    """

    def __call__(self, t: float, diff: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def gradient_x(self, t: float, diff: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class GaussianKernel(Kernel):
    r"""
    Stationary Gaussian kernel :math:`k(x, y) = \exp(-\alpha \|x - y\|^2)`.

    This is the kernel used in all numerical experiments of the paper, with
    the default :math:`\alpha = 2`. The kernel is symmetric, positive, and
    decays rapidly: setting ``alpha`` larger makes the interaction more
    local.
    """

    alpha: float = 2.0

    def __call__(self, t: float, diff: np.ndarray) -> np.ndarray:
        diff = np.asarray(diff, dtype=float)
        dist_sq = np.sum(diff**2, axis=-1)
        return np.exp(-self.alpha * dist_sq)

    def gradient_x(self, t: float, diff: np.ndarray) -> np.ndarray:
        # nabla_x exp(-alpha |x - y|^2) = -2 alpha (x - y) k(x, y)
        diff = np.asarray(diff, dtype=float)
        k_vals = self(t, diff)
        return -2.0 * self.alpha * diff * k_vals[..., None]


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

def setup_quadrature(
    order: int = 30,
    lower: float = 0.0,
    upper: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tensor-product 2-D Gauss-Legendre quadrature on the box
    ``[lower, upper] x [lower, upper]``.

    Parameters
    ----------
    order : int
        Number of 1-D Gauss-Legendre nodes per axis; the 2-D rule has
        ``order**2`` points. Defaults to 30 (see the paper, Section 6).
    lower, upper : float
        Box bounds. The default ``[0, 1]^2`` contains all geometries used
        by :class:`rbf_nonlocal.domains.FlowerDomain` and
        :class:`rbf_nonlocal.domains.GearDomain`.

    Returns
    -------
    quad_points : np.ndarray, shape ``(order**2, 2)``
    quad_weights : np.ndarray, shape ``(order**2,)``
    """
    if order <= 0:
        raise ValueError(f"order must be a positive integer (got {order}).")
    if upper <= lower:
        raise ValueError(f"upper must be > lower (got {lower} and {upper}).")

    xi_1d, w_1d = leggauss(order)
    L = upper - lower
    pts_1d = lower + 0.5 * L * (xi_1d + 1.0)
    wts_1d = 0.5 * L * w_1d

    XX, YY = np.meshgrid(pts_1d, pts_1d, indexing="xy")
    WX, WY = np.meshgrid(wts_1d, wts_1d, indexing="xy")
    quad_points = np.column_stack([XX.ravel(), YY.ravel()])
    quad_weights = (WX * WY).ravel()
    return quad_points, quad_weights


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class NonlocalOperators:
    r"""
    Precomputed matrices for the nonlocal operator :math:`\mathcal{W}_{u_h}`
    and its spatial gradient, evaluated at the interior collocation nodes.

    For the RBF interpolant :math:`u_h = a^T X + b^T \tilde G`,

    .. math::
        \mathcal{W}_{u_h}(t, x_i) &= K_a X + K_b \tilde G, \\
        \partial_{x_1} \mathcal{W}_{u_h}(t, x_i) &= dK_{a,x} X + dK_{b,x} \tilde G, \\
        \partial_{x_2} \mathcal{W}_{u_h}(t, x_i) &= dK_{a,y} X + dK_{b,y} \tilde G.

    Attributes
    ----------
    Ka      : np.ndarray, shape ``(n, n)``
    Kb      : np.ndarray, shape ``(n, n_prime + dm)``
    dKa_dx  : np.ndarray, shape ``(n, n)``
    dKa_dy  : np.ndarray, shape ``(n, n)``
    dKb_dx  : np.ndarray, shape ``(n, n_prime + dm)``
    dKb_dy  : np.ndarray, shape ``(n, n_prime + dm)``
    """
    Ka: np.ndarray
    Kb: np.ndarray
    dKa_dx: np.ndarray
    dKa_dy: np.ndarray
    dKb_dx: np.ndarray
    dKb_dy: np.ndarray


# ---------------------------------------------------------------------------
# The single workhorse routine
# ---------------------------------------------------------------------------

def compute_nonlocal_operators(
    mats: RBFInterpolationMatrices,
    quad_points: np.ndarray,
    quad_weights: np.ndarray,
    domain: Domain,
    kernel: Optional[Kernel] = None,
    t: float = 0.0,
) -> NonlocalOperators:
    r"""
    Assemble all six nonlocal-operator matrices in a single vectorised pass.

    Parameters
    ----------
    mats : RBFInterpolationMatrices
        From :func:`rbf_nonlocal.matrices.build_matrices`.
    quad_points : np.ndarray, shape ``(Q, 2)``
        Quadrature nodes (typically from :func:`setup_quadrature` on a box
        containing the domain).
    quad_weights : np.ndarray, shape ``(Q,)``
        Corresponding quadrature weights.
    domain : Domain
        Geometry whose indicator restricts the integral to ``Omega``.
    kernel : Kernel, optional
        Nonlocal interaction kernel. Defaults to ``GaussianKernel(alpha=2.0)``,
        i.e. :math:`k(x, y) = e^{-2\|x - y\|^2}`, the choice used in the paper.
    t : float, optional
        Time at which to evaluate a (possibly) time-dependent kernel. Ignored
        by :class:`GaussianKernel`.

    Returns
    -------
    NonlocalOperators

    Notes
    -----
    For a time-independent kernel (the default), this function should be
    called once at setup. For a time-dependent kernel it must be re-called
    at every time step.
    """
    if kernel is None:
        kernel = GaussianKernel(alpha=2.0)

    quad_points = np.asarray(quad_points, dtype=float)
    quad_weights = np.asarray(quad_weights, dtype=float)

    if quad_points.ndim != 2 or quad_points.shape[1] != 2:
        raise ValueError("quad_points must have shape (Q, 2).")
    if quad_weights.shape != (len(quad_points),):
        raise ValueError("quad_weights must have shape (Q,) matching quad_points.")

    # 1. Evaluate the dual basis at the quadrature points (these are the
    #    a_j(y_q) and b_j(y_q) values).
    a_quad, b_quad = evaluate_basis_functions(quad_points, mats)
    #   a_quad: (Q, n)        b_quad: (Q, n_prime + dm)

    # 2. Combined effective weights: domain indicator times quadrature weight.
    indicator = domain.contains(quad_points[:, 0], quad_points[:, 1]).astype(float)
    eff_weights = quad_weights * indicator                          # (Q,)

    # 3. Pairwise displacements diff[i, q, :] = x_i - y_q.
    diff = mats.interior_points[:, None, :] - quad_points[None, :, :]   # (n, Q, 2)

    # 4. Kernel and its gradient at all (i, q) pairs.
    k_vals = kernel(t, diff)                                        # (n, Q)
    grad_k = kernel.gradient_x(t, diff)                             # (n, Q, 2)

    # 5. Apply effective weights once.
    w_k    = k_vals * eff_weights[None, :]                          # (n, Q)
    w_grad = grad_k * eff_weights[None, :, None]                    # (n, Q, 2)

    # 6. Quadrature contractions over the q-axis.
    #    Ka[i, j]    = sum_q  w_k[i, q]    * a_quad[q, j]
    #    Kb[i, j]    = sum_q  w_k[i, q]    * b_quad[q, j]
    #    dKa[i,j,d]  = sum_q  w_grad[i,q,d] * a_quad[q, j]
    #    dKb[i,j,d]  = sum_q  w_grad[i,q,d] * b_quad[q, j]
    Ka = w_k @ a_quad                                               # (n, n)
    Kb = w_k @ b_quad                                               # (n, n_prime + dm)
    Ka_grad = np.einsum("iqd,qj->ijd", w_grad, a_quad)              # (n, n, 2)
    Kb_grad = np.einsum("iqd,qj->ijd", w_grad, b_quad)              # (n, n_prime + dm, 2)

    return NonlocalOperators(
        Ka=Ka,
        Kb=Kb,
        dKa_dx=Ka_grad[:, :, 0],
        dKa_dy=Ka_grad[:, :, 1],
        dKb_dx=Kb_grad[:, :, 0],
        dKb_dy=Kb_grad[:, :, 1],
    )