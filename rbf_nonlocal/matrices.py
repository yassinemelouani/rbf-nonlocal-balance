"""
RBF + polynomial collocation matrices.

This module assembles the dual-basis representation of the RBF interpolant
described in Section 2 of the paper, with the Schur complement decomposition
that lets the interior coefficients ``X(t)`` be solved separately from the
boundary data ``G(t)``.

Given interior nodes :math:`x_1, \\ldots, x_n` and boundary nodes
:math:`x'_1, \\ldots, x'_{n'}`, the interpolant takes the form

.. math::
    u_h(t, x) = a(x)^T X(t) + b(x)^T \\tilde{G}(t),

where :math:`a(x) \\in \\mathbb{R}^n`, :math:`b(x) \\in \\mathbb{R}^{n' + d_m}`,
and :math:`\\tilde{G}(t)` combines the boundary values with zero polynomial
coefficients.

The matrices needed for this representation are computed once in
:func:`build_matrices` and stored in an :class:`RBFInterpolationMatrices`
container. All subsequent operations — applying the Laplacian or bi-Laplacian
to the dual basis, taking gradients, evaluating pointwise — read from that
container.

All routines are vectorised over evaluation points; nothing iterates over
collocation nodes in pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from . import basis

__all__ = [
    "RBFInterpolationMatrices",
    "build_matrices",
    "compute_basis_laplacians",
    "compute_basis_bilaplacians",
    "compute_basis_gradients",
    "evaluate_basis_functions",
]


# ---------------------------------------------------------------------------
# Polynomial helpers (private; could be made public if useful)
# ---------------------------------------------------------------------------

def _polynomial_basis(eval_points: np.ndarray, polydeg: int) -> np.ndarray:
    r"""
    Monomial basis :math:`\{x^i y^j : i + j \le \mathrm{polydeg}\}` at each
    evaluation point.

    Returns shape ``(n_eval, dm)`` with ``dm = (polydeg+1)(polydeg+2)/2``,
    ordered by total degree: 1; x, y; x^2, xy, y^2; x^3, x^2 y, x y^2, y^3; ...
    """
    x = np.asarray(eval_points[:, 0], dtype=float)
    y = np.asarray(eval_points[:, 1], dtype=float)
    dm = (polydeg + 1) * (polydeg + 2) // 2
    out = np.empty((len(eval_points), dm), dtype=float)
    count = 0
    for total in range(polydeg + 1):
        for i in range(total + 1):
            j = total - i
            out[:, count] = (x ** i) * (y ** j)
            count += 1
    return out


def _polynomial_laplacians(eval_points: np.ndarray, polydeg: int) -> np.ndarray:
    r"""
    Laplacians of the monomials :math:`\{x^i y^j\}`. Shape ``(n_eval, dm)``.

    For a monomial :math:`x^i y^j`,
    :math:`\Delta(x^i y^j) = i(i-1) x^{i-2} y^j + j(j-1) x^i y^{j-2}`.
    """
    x = np.asarray(eval_points[:, 0], dtype=float)
    y = np.asarray(eval_points[:, 1], dtype=float)
    dm = (polydeg + 1) * (polydeg + 2) // 2
    n_eval = len(eval_points)
    out = np.zeros((n_eval, dm), dtype=float)
    count = 0
    for total in range(polydeg + 1):
        for i in range(total + 1):
            j = total - i
            term = np.zeros(n_eval)
            if i >= 2:
                term += i * (i - 1) * (x ** (i - 2)) * (y ** j)
            if j >= 2:
                term += j * (j - 1) * (x ** i) * (y ** (j - 2))
            out[:, count] = term
            count += 1
    return out


def _polynomial_bilaplacians(eval_points: np.ndarray, polydeg: int) -> np.ndarray:
    r"""
    Bi-Laplacians of the monomials. Shape ``(n_eval, dm)``.

    For :math:`x^i y^j`,

    .. math::
        \Delta^2(x^i y^j) = i(i-1)(i-2)(i-3)\, x^{i-4} y^j
                          + 2\, i(i-1) j(j-1)\, x^{i-2} y^{j-2}
                          + j(j-1)(j-2)(j-3)\, x^i y^{j-4}.
    """
    x = np.asarray(eval_points[:, 0], dtype=float)
    y = np.asarray(eval_points[:, 1], dtype=float)
    dm = (polydeg + 1) * (polydeg + 2) // 2
    n_eval = len(eval_points)
    out = np.zeros((n_eval, dm), dtype=float)
    count = 0
    for total in range(polydeg + 1):
        for i in range(total + 1):
            j = total - i
            term = np.zeros(n_eval)
            if i >= 4:
                term += (i * (i - 1) * (i - 2) * (i - 3)
                         * (x ** (i - 4)) * (y ** j))
            if i >= 2 and j >= 2:
                term += (2 * i * (i - 1) * j * (j - 1)
                         * (x ** (i - 2)) * (y ** (j - 2)))
            if j >= 4:
                term += (j * (j - 1) * (j - 2) * (j - 3)
                         * (x ** i) * (y ** (j - 4)))
            out[:, count] = term
            count += 1
    return out


def _polynomial_gradients(eval_points: np.ndarray, polydeg: int) -> np.ndarray:
    r"""
    Gradients of the monomials. Shape ``(n_eval, dm, 2)``,
    with the trailing axis indexing :math:`(\partial_x, \partial_y)`.
    """
    x = np.asarray(eval_points[:, 0], dtype=float)
    y = np.asarray(eval_points[:, 1], dtype=float)
    dm = (polydeg + 1) * (polydeg + 2) // 2
    n_eval = len(eval_points)
    out = np.zeros((n_eval, dm, 2), dtype=float)
    count = 0
    for total in range(polydeg + 1):
        for i in range(total + 1):
            j = total - i
            if i > 0:
                out[:, count, 0] = i * (x ** (i - 1)) * (y ** j)
            if j > 0:
                out[:, count, 1] = j * (x ** i) * (y ** (j - 1))
            count += 1
    return out


# ---------------------------------------------------------------------------
# Container for the precomputed matrices
# ---------------------------------------------------------------------------

@dataclass
class RBFInterpolationMatrices:
    """
    Bundles everything needed for evaluating and applying differential
    operators to the RBF interpolant on a fixed set of nodes.

    Attributes
    ----------
    interior_points : np.ndarray, shape ``(n, 2)``
    boundary_points : np.ndarray, shape ``(n_prime, 2)``
    kind : str
        RBF kernel kind, see :mod:`rbf_nonlocal.basis`.
    tau : float
        Tension parameter (ignored when ``kind='thin_plate'``).
    polydeg : int
        Degree of the polynomial tail.
    n, n_prime, dm : int
        Cached node counts and polynomial dimension
        ``dm = (polydeg+1)(polydeg+2)/2``.
    S, S_inv : np.ndarray, shape ``(n, n)``
        Schur complement of the interpolation matrix and its inverse.
        ``S`` is symmetric, hence so is ``S_inv``.
    B : np.ndarray, shape ``(n, n_prime + dm)``
    C : np.ndarray, shape ``(n_prime + dm, n_prime + dm)``
        :math:`C = A^{-1} + B^T S^{-1} B`, also symmetric.
    A_inv : np.ndarray, shape ``(n_prime + dm, n_prime + dm)``
    """

    interior_points: np.ndarray
    boundary_points: np.ndarray
    kind: str
    tau: float
    polydeg: int
    n: int
    n_prime: int
    dm: int
    S: np.ndarray
    S_inv: np.ndarray
    B: np.ndarray
    C: np.ndarray
    A_inv: np.ndarray


# ---------------------------------------------------------------------------
# Build matrices
# ---------------------------------------------------------------------------

def build_matrices(
    interior_points: np.ndarray,
    boundary_points: np.ndarray,
    kind: str = "tension",
    tau: float = 1.0,
    polydeg: int = 1,
) -> RBFInterpolationMatrices:
    """
    Assemble the Schur-complement matrices for RBF + polynomial collocation.

    Parameters
    ----------
    interior_points : array-like, shape ``(n, 2)``
    boundary_points : array-like, shape ``(n_prime, 2)``
    kind : str
        RBF kind: ``"tension"`` (default, used in the diffusive case) or
        ``"thin_plate"`` (used for the bi-Laplacian / hyper-viscosity case).
    tau : float
        Tension parameter for ``kind='tension'``.
    polydeg : int
        Degree of the polynomial tail; common choices are 1, 2, 3, 4.

    Returns
    -------
    RBFInterpolationMatrices
    """
    interior_points = np.asarray(interior_points, dtype=float)
    boundary_points = np.asarray(boundary_points, dtype=float)

    if interior_points.ndim != 2 or interior_points.shape[1] != 2:
        raise ValueError("interior_points must have shape (n, 2).")
    if boundary_points.ndim != 2 or boundary_points.shape[1] != 2:
        raise ValueError("boundary_points must have shape (n_prime, 2).")
    if polydeg < 0:
        raise ValueError(f"polydeg must be >= 0 (got {polydeg}).")

    n, n_prime = len(interior_points), len(boundary_points)
    dm = (polydeg + 1) * (polydeg + 2) // 2

    all_points = np.vstack([interior_points, boundary_points])
    dists = cdist(all_points, all_points)

    K = basis.phi(dists, kind=kind, tau=tau)
    Q = _polynomial_basis(all_points, polydeg)

    K1, K2, M = K[:n, :n], K[n:, n:], K[:n, n:]
    Q1, Q2 = Q[:n, :], Q[n:, :]

    A_mat = np.block([[K2,    Q2],
                      [Q2.T,  np.zeros((dm, dm))]])
    A_inv = np.linalg.inv(A_mat)
    MQ1 = np.hstack([M, Q1])
    B_mat = MQ1 @ A_inv
    S = K1 - MQ1 @ A_inv @ MQ1.T
    S_inv = np.linalg.inv(S)
    C = A_inv + B_mat.T @ S_inv @ B_mat

    return RBFInterpolationMatrices(
        interior_points=interior_points,
        boundary_points=boundary_points,
        kind=kind,
        tau=tau,
        polydeg=polydeg,
        n=n,
        n_prime=n_prime,
        dm=dm,
        S=S,
        S_inv=S_inv,
        B=B_mat,
        C=C,
        A_inv=A_inv,
    )


# ---------------------------------------------------------------------------
# Differential operators on the dual basis
# ---------------------------------------------------------------------------

def _compute_basis_diffop(
    mats: RBFInterpolationMatrices,
    rbf_op,
    poly_op,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Common engine for :func:`compute_basis_laplacians` and
    :func:`compute_basis_bilaplacians`.

    Given an operator :math:`L` (the 2-D Laplacian or bi-Laplacian here),
    return matrices ``La, Lb`` such that for the interpolant
    :math:`u_h = a^T X + b^T \tilde G`,

    .. math::
        L u_h(x_i) = (L_a)_{ij} X_j + (L_b)_{ij} \tilde G_j .

    Parameters
    ----------
    mats : RBFInterpolationMatrices
    rbf_op : callable
        ``rbf_op(r, kind=..., tau=...) -> ndarray`` of the same shape as
        ``r``, e.g. :func:`basis.laplacian_2d` or :func:`basis.bilaplacian_2d`.
    poly_op : callable
        ``poly_op(eval_points, polydeg) -> (n_eval, dm)``, the same operator
        applied to each polynomial basis function.

    Returns
    -------
    La : np.ndarray, shape ``(n, n)``
    Lb : np.ndarray, shape ``(n, n_prime + dm)``
    """
    interior, boundary = mats.interior_points, mats.boundary_points

    dists_int = cdist(interior, interior)        # (n, n)
    dists_bnd = cdist(interior, boundary)        # (n, n_prime)

    L_lambda = rbf_op(dists_int, kind=mats.kind, tau=mats.tau)
    L_gamma  = rbf_op(dists_bnd, kind=mats.kind, tau=mats.tau)
    L_theta  = poly_op(interior, mats.polydeg)
    L_gamma_theta = np.hstack([L_gamma, L_theta])              # (n, n_prime + dm)

    # S_inv is symmetric, so ``v @ S_inv == S_inv @ v`` and we can compute
    # the entire matrix in two `@` operations instead of a Python loop.
    La = (L_lambda - L_gamma_theta @ mats.B.T) @ mats.S_inv
    Lb = L_gamma_theta @ mats.C - (L_lambda @ mats.S_inv) @ mats.B

    return La, Lb


def compute_basis_laplacians(
    mats: RBFInterpolationMatrices,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Action of the 2-D Laplacian on the dual basis evaluated at the interior
    nodes.

    Returns
    -------
    La : np.ndarray, shape ``(n, n)``
    Lb : np.ndarray, shape ``(n, n_prime + dm)``
        such that :math:`\Delta u_h(x_i) = (L_a)_{ij} X_j + (L_b)_{ij} \tilde G_j`.
    """
    return _compute_basis_diffop(mats, basis.laplacian_2d, _polynomial_laplacians)


def compute_basis_bilaplacians(
    mats: RBFInterpolationMatrices,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Action of the 2-D bi-Laplacian :math:`\Delta^2` on the dual basis at the
    interior nodes. Requires ``mats.kind == 'thin_plate'``: the tension RBF
    is only :math:`C^2` and is not collocation-safe for fourth-order
    operators.

    Returns
    -------
    BiLa : np.ndarray, shape ``(n, n)``
    BiLb : np.ndarray, shape ``(n, n_prime + dm)``
    """
    if mats.kind != "thin_plate":
        raise ValueError(
            "compute_basis_bilaplacians requires kind='thin_plate' "
            f"(got kind={mats.kind!r}). Build matrices with "
            "kind='thin_plate' for the hyper-viscosity / bi-Laplacian regime."
        )
    return _compute_basis_diffop(mats, basis.bilaplacian_2d, _polynomial_bilaplacians)


# ---------------------------------------------------------------------------
# Gradients of the dual basis
# ---------------------------------------------------------------------------

def compute_basis_gradients(
    eval_points: np.ndarray,
    mats: RBFInterpolationMatrices,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Gradients :math:`\nabla a(x)` and :math:`\nabla b(x)` of the dual basis at
    arbitrary evaluation points.

    Parameters
    ----------
    eval_points : array-like, shape ``(n_eval, 2)``
    mats : RBFInterpolationMatrices

    Returns
    -------
    grad_a : np.ndarray, shape ``(n_eval, n, 2)``
    grad_b : np.ndarray, shape ``(n_eval, n_prime + dm, 2)``
    """
    eval_points = np.asarray(eval_points, dtype=float)
    interior, boundary = mats.interior_points, mats.boundary_points

    # dx[k, i, :] = eval_points[k] - nodes[i]
    dx_int = eval_points[:, None, :] - interior[None, :, :]    # (n_eval, n,       2)
    dx_bnd = eval_points[:, None, :] - boundary[None, :, :]    # (n_eval, n_prime, 2)

    # Avoid 0/0 at coincident points: phi'(0) = 0 for both kernels here, so
    # a tiny floor on r is harmless.
    r_int = np.maximum(np.linalg.norm(dx_int, axis=2), 1e-12)  # (n_eval, n)
    r_bnd = np.maximum(np.linalg.norm(dx_bnd, axis=2), 1e-12)  # (n_eval, n_prime)

    phi_p_int = basis.phi_prime(r_int, kind=mats.kind, tau=mats.tau)
    phi_p_bnd = basis.phi_prime(r_bnd, kind=mats.kind, tau=mats.tau)

    # nabla phi(|x - x_i|) = phi'(r) * (x - x_i) / r
    grad_lambda = (phi_p_int / r_int)[..., None] * dx_int           # (n_eval, n,            2)
    grad_gamma  = (phi_p_bnd / r_bnd)[..., None] * dx_bnd           # (n_eval, n_prime,      2)
    grad_theta  = _polynomial_gradients(eval_points, mats.polydeg)  # (n_eval, dm,           2)

    grad_gamma_theta = np.concatenate([grad_gamma, grad_theta], axis=1)  # (n_eval, n_prime + dm, 2)

    # NumPy's matmul broadcasts (M, K) against (..., K, N) into (..., M, N),
    # so we can do these batched operations without einsum.
    S_inv_grad_lambda = mats.S_inv @ grad_lambda                              # (n_eval, n,             2)
    grad_a = -mats.S_inv @ (mats.B @ grad_gamma_theta) + S_inv_grad_lambda    # (n_eval, n,             2)
    grad_b =  mats.C @ grad_gamma_theta - mats.B.T @ S_inv_grad_lambda        # (n_eval, n_prime + dm,  2)

    return grad_a, grad_b


# ---------------------------------------------------------------------------
# Pointwise evaluation of the dual basis
# ---------------------------------------------------------------------------

def evaluate_basis_functions(
    eval_points: np.ndarray,
    mats: RBFInterpolationMatrices,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Evaluate the dual-basis functions ``a(x)`` and ``b(x)`` at arbitrary
    points so that :math:`u_h(t, x) = a(x)^T X(t) + b(x)^T \tilde G(t)`.

    Parameters
    ----------
    eval_points : array-like, shape ``(n_eval, 2)``
    mats : RBFInterpolationMatrices

    Returns
    -------
    a_mat : np.ndarray, shape ``(n_eval, n)``
    b_mat : np.ndarray, shape ``(n_eval, n_prime + dm)``
    """
    eval_points = np.asarray(eval_points, dtype=float)
    interior, boundary = mats.interior_points, mats.boundary_points

    dists_int = cdist(eval_points, interior)
    dists_bnd = cdist(eval_points, boundary)
    lambda_mat = basis.phi(dists_int, kind=mats.kind, tau=mats.tau)
    gamma_mat  = basis.phi(dists_bnd, kind=mats.kind, tau=mats.tau)
    theta_mat  = _polynomial_basis(eval_points, mats.polydeg)

    gamma_theta_mat = np.hstack([gamma_mat, theta_mat])

    a_mat = (lambda_mat - gamma_theta_mat @ mats.B.T) @ mats.S_inv
    b_mat = gamma_theta_mat @ mats.C - (lambda_mat @ mats.S_inv) @ mats.B

    return a_mat, b_mat