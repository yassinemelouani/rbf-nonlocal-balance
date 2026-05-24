"""
Semi-discrete right-hand side and Frechet derivative.

After RBF + polynomial collocation, the system

.. math::
    \\partial_t u + \\nabla \\cdot (V(\\mathcal{W}_u) \\otimes u)
    + \\mathcal{L} u = \\mathcal{N}(u) + f

reduces to a coupled ODE for the interior coefficients :math:`X(t)`:

.. math::
    \\dot X = \\Phi(t, X),
    \\qquad \\Phi(t, X) = -\\, [\\text{convective}] - [\\text{regularizer}]
                          + \\mathcal{N}(u_h) + f.

This module supplies:

* :func:`compute_rhs` — the function :math:`\\Phi(t, X)`.
* :func:`frechet_derivative` — the matrix-vector product
  :math:`\\delta X \\mapsto \\partial_X \\Phi(t, X)\\,\\delta X`, derived
  analytically (no finite differences).
* :func:`make_frechet_operator` — wraps the Frechet derivative as a
  :class:`scipy.sparse.linalg.LinearOperator` that GMRES can consume.

Both ``regularization='laplacian'`` (with diffusion matrix ``nu``) and
``regularization='bilaplacian'`` (with hyper-viscosity ``epsilon``) are
handled by the same routines, controlled by which Laplacian / bi-Laplacian
matrices are passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .matrices import RBFInterpolationMatrices
from .nonlocal_ops import NonlocalOperators
from .reaction import Reaction
from .velocity import Velocity

__all__ = [
    "RegularizationOperator",
    "compute_rhs",
    "frechet_derivative",
    "make_frechet_operator",
]


# ---------------------------------------------------------------------------
# Regularization operator container
# ---------------------------------------------------------------------------

@dataclass
class RegularizationOperator:
    """
    Bundles the higher-order operator that closes the system.

    For ``mode='laplacian'``, the contribution to the right-hand side is
    :math:`+\\nu\\,\\Delta u_h`, with ``coeff = nu`` (a ``(2, 2)`` matrix
    coupling the two components).

    For ``mode='bilaplacian'``, the contribution is
    :math:`-\\varepsilon\\,\\Delta^2 u_h`, with ``coeff = epsilon`` (a
    scalar applied component-wise).

    Attributes
    ----------
    mode : str
        ``"laplacian"`` or ``"bilaplacian"``.
    coeff : np.ndarray or float
        Diffusion matrix ``nu`` or hyper-viscosity ``epsilon``.
    Xa : np.ndarray, shape ``(n, n)``
        Interior part of the operator on the dual basis (from
        :func:`compute_basis_laplacians` or
        :func:`compute_basis_bilaplacians`).
    Xb : np.ndarray, shape ``(n, n_prime + dm)``
        Boundary part of the same operator.
    """

    mode: str
    coeff: object
    Xa: np.ndarray
    Xb: np.ndarray

    def apply(self, X: np.ndarray, G_tilde: np.ndarray) -> np.ndarray:
        r"""
        Return the *right-hand-side* contribution of the regulariser, signed
        so it can be added directly to :math:`\Phi(t, X)`.

        For ``mode='laplacian'``:   returns ``nu @ (X^T (Xa^T)) ...``  i.e.
        :math:`+\nu\,\Delta u_h` evaluated at the interior nodes.

        For ``mode='bilaplacian'``: returns
        :math:`-\varepsilon\,\Delta^2 u_h` at the interior nodes.

        Parameters
        ----------
        X : np.ndarray, shape ``(2, n)``
            Interior coefficients per component.
        G_tilde : np.ndarray, shape ``(2, n_prime + dm)``
            Boundary data + zero-padded polynomial coefficients per component.

        Returns
        -------
        np.ndarray, shape ``(n, 2)``
            Contribution to ``Phi`` at every interior node, per component.
        """
        # Lu_int has shape (2, n): Lu_int[d, i] = (Xa @ X[d]) + (Xb @ G_tilde[d])
        Lu = X @ self.Xa.T + G_tilde @ self.Xb.T            # (2, n)

        if self.mode == "laplacian":
            # +ν Δu, with ν a 2x2 matrix
            nu = np.asarray(self.coeff, dtype=float)
            return (nu @ Lu).T                              # (n, 2)
        elif self.mode == "bilaplacian":
            # -ε Δ²u, scalar
            return -float(self.coeff) * Lu.T                # (n, 2)
        else:
            raise ValueError(
                f"Unknown regularization mode {self.mode!r}; "
                "expected 'laplacian' or 'bilaplacian'."
            )


# ---------------------------------------------------------------------------
# Helpers — boundary lift
# ---------------------------------------------------------------------------

def _make_G_tilde(
    G: np.ndarray, mats: RBFInterpolationMatrices,
) -> np.ndarray:
    """
    Lift Dirichlet data ``G`` of shape ``(2, n_prime)`` to the full
    ``(2, n_prime + dm)`` vector by zero-padding the polynomial slots.
    """
    G = np.asarray(G, dtype=float)
    if G.shape != (2, mats.n_prime):
        raise ValueError(
            f"G must have shape (2, n_prime) = (2, {mats.n_prime}); "
            f"got {G.shape}."
        )
    return np.hstack([G, np.zeros((2, mats.dm))])


# ---------------------------------------------------------------------------
# compute_rhs
# ---------------------------------------------------------------------------

def compute_rhs(
    t: float,
    X: np.ndarray,
    G: np.ndarray,
    f: np.ndarray,
    mats: RBFInterpolationMatrices,
    nl_ops: NonlocalOperators,
    reg: RegularizationOperator,
    velocity: Velocity,
    reaction: Reaction,
) -> np.ndarray:
    r"""
    Evaluate :math:`\Phi(t, X)` at the interior nodes.

    Parameters
    ----------
    t : float
        Current time. Forwarded only so a future time-dependent kernel
        could re-evaluate ``nl_ops``; not used by the default Gaussian.
    X : np.ndarray, shape ``(2, n)``
        Interior coefficients per component.
    G : np.ndarray, shape ``(2, n_prime)``
        Dirichlet boundary data per component at this instant.
    f : np.ndarray, shape ``(n, 2)``
        Source term at this instant (zero in simulation mode).
    mats : RBFInterpolationMatrices
    nl_ops : NonlocalOperators
        Precomputed Ka, Kb and their gradients.
    reg : RegularizationOperator
    velocity : Velocity
    reaction : Reaction

    Returns
    -------
    rhs : np.ndarray, shape ``(2, n)``
        Right-hand side per component, suitable for direct use in BDF
        time stepping (i.e., ``X^{n+1} = ... + dt * gamma_0 * rhs``).
    """
    G_tilde = _make_G_tilde(G, mats)               # (2, n_prime + dm)
    n = mats.n

    # 1. Nonlocal density Wu and its spatial gradient at the interior nodes.
    Wu     = X @ nl_ops.Ka.T  + G_tilde @ nl_ops.Kb.T          # (2, n)
    grad_Wu_x = X @ nl_ops.dKa_dx.T + G_tilde @ nl_ops.dKb_dx.T  # (2, n)
    grad_Wu_y = X @ nl_ops.dKa_dy.T + G_tilde @ nl_ops.dKb_dy.T  # (2, n)

    # Reshape to (n, 2[component]) to feed velocity.field/jacobian, which
    # expect the trailing axis to index components.
    Wu_T   = Wu.T                                              # (n, 2)
    gradWu = np.stack([grad_Wu_x.T, grad_Wu_y.T], axis=-1)     # (n, 2, 2)  [..., d, dim]

    # 2. Velocity and ∇·V at every interior node.
    V  = velocity.field(Wu_T)                                  # (n, 2)
    JV = velocity.jacobian(Wu_T)                               # (n, 2, 2): [..., d, c]
    # ∇V[i, d, dim] = JV[i, d, c] · ∇Wu[i, c, dim]
    grad_V = np.einsum("idc,icj->idj", JV, gradWu)             # (n, 2, 2)
    div_V  = grad_V[..., 0, 0] + grad_V[..., 1, 1]             # (n,)

    # 3. ∇u at the interior nodes via the basis gradients precomputed for
    #    interior eval in `mats`. We use compute_basis_gradients here in
    #    the time-loop hot path, so we precompute it once outside (see
    #    solver.py). Here we rely on the caller having supplied them via
    #    nl_ops/reg if needed; for compute_rhs we recompute on the fly
    #    using the formula
    #
    #       ∇u_h(x_i) = ∂_x a(x_i) X[d] + ∂_x b(x_i) G_tilde[d]
    #
    #    The basis-gradient matrices for evaluation at the interior nodes
    #    are stashed in `mats` lazily; see solver.py.
    grad_a, grad_b = _get_interior_gradient_matrices(mats)     # (n, n, 2), (n, n_prime+dm, 2)
    # ∇u[i, d, dim]
    grad_u = np.einsum("inj,dn->idj", grad_a, X) + np.einsum(
        "inj,dn->idj", grad_b, G_tilde
    )

    # 4. Convective divergence in product form:
    #       ∇·(V ⊗ u) = (V·∇u) + u (∇·V)
    u_int      = X.T + 0.0 * Wu_T  # placeholder shape, see below
    # u_h at interior nodes: u_h(x_i) = a(x_i)^T X[d] + b(x_i)^T G_tilde[d]
    # By construction a(x_i) = e_i and b(x_i) = 0, so:
    u_int = X.T                                                # (n, 2)
    Vdotgrad_u = np.einsum("ij,idj->id", V, grad_u)            # (n, 2)
    conv = Vdotgrad_u + u_int * div_V[:, None]                 # (n, 2)

    # 5. Reaction.
    Nval = reaction.field(u_int)                               # (n, 2)

    # 6. Regulariser (signed appropriately by reg.apply).
    reg_term = reg.apply(X, G_tilde)                           # (n, 2)

    # 7. Assemble Φ:  ∂_t u = -conv + reg + N + f    →   Φ = -conv + reg + N + f.
    phi = -conv + reg_term + Nval + np.asarray(f, dtype=float)

    # Return shape (2, n) for direct use in BDF on X.
    return phi.T


# ---------------------------------------------------------------------------
# frechet_derivative
# ---------------------------------------------------------------------------

def frechet_derivative(
    delta_X: np.ndarray,
    X: np.ndarray,
    G: np.ndarray,
    mats: RBFInterpolationMatrices,
    nl_ops: NonlocalOperators,
    reg: RegularizationOperator,
    velocity: Velocity,
    reaction: Reaction,
) -> np.ndarray:
    r"""
    Matrix-vector product :math:`\partial_X \Phi(t, X) \cdot \delta X`.

    All terms are differentiated analytically; no finite differences are
    used. The boundary data ``G`` is held fixed (Dirichlet), so the lift
    ``G_tilde`` does not contribute a perturbation.

    Parameters
    ----------
    delta_X : np.ndarray, shape ``(2, n)``
        Direction in which to evaluate the derivative.
    X, G, mats, nl_ops, reg, velocity, reaction
        Same meaning as in :func:`compute_rhs`.

    Returns
    -------
    np.ndarray, shape ``(2, n)``
    """
    G_tilde = _make_G_tilde(G, mats)
    delta_X = np.asarray(delta_X, dtype=float)
    if delta_X.shape != X.shape:
        raise ValueError(
            f"delta_X must have shape {X.shape}; got {delta_X.shape}."
        )

    # 1. Wu, ∇Wu at base point.
    Wu        = X @ nl_ops.Ka.T  + G_tilde @ nl_ops.Kb.T          # (2, n)
    grad_Wu_x = X @ nl_ops.dKa_dx.T + G_tilde @ nl_ops.dKb_dx.T   # (2, n)
    grad_Wu_y = X @ nl_ops.dKa_dy.T + G_tilde @ nl_ops.dKb_dy.T   # (2, n)
    Wu_T   = Wu.T                                                 # (n, 2)
    gradWu = np.stack([grad_Wu_x.T, grad_Wu_y.T], axis=-1)        # (n, 2, 2)

    # 2. Perturbations of Wu and ∇Wu (linear in delta_X; only the
    #    interior block contributes).
    dWu        = (delta_X @ nl_ops.Ka.T).T                        # (n, 2)
    dgrad_Wu_x = (delta_X @ nl_ops.dKa_dx.T).T                    # (n, 2)
    dgrad_Wu_y = (delta_X @ nl_ops.dKa_dy.T).T                    # (n, 2)
    dgradWu = np.stack([dgrad_Wu_x, dgrad_Wu_y], axis=-1)         # (n, 2, 2)

    # 3. Velocity, its first and second derivatives at base point.
    V  = velocity.field(Wu_T)                                     # (n, 2)
    JV = velocity.jacobian(Wu_T)                                  # (n, 2, 2): [..., d, c]
    HV = velocity.hessian(Wu_T)                                   # (n, 2, 2, 2): [..., d, c, o]

    #    δV[i, d] = JV[i, d, c] · δW[i, c]
    dV = np.einsum("idc,ic->id", JV, dWu)                         # (n, 2)
    #    δ(∇V)[i, d, dim]
    #       = HV[i, d, c, o] · δW[i, o] · ∇W[i, c, dim]
    #       + JV[i, d, c]    · δ(∇W)[i, c, dim]
    dgrad_V = (
        np.einsum("idco,io,icj->idj", HV, dWu, gradWu)
        + np.einsum("idc,icj->idj", JV, dgradWu)
    )                                                             # (n, 2, 2)
    ddiv_V = dgrad_V[..., 0, 0] + dgrad_V[..., 1, 1]              # (n,)

    # 4. Perturbed gradient of u (linear in δX).
    grad_a, grad_b = _get_interior_gradient_matrices(mats)
    grad_u  = np.einsum("inj,dn->idj", grad_a, X)       + np.einsum("inj,dn->idj", grad_b, G_tilde)
    dgrad_u = np.einsum("inj,dn->idj", grad_a, delta_X)           # boundary unchanged

    # 5. δ of u_h at the nodes is just δX (since a(x_i) = e_i).
    u_int  = X.T                                                  # (n, 2)
    du_int = delta_X.T                                            # (n, 2)

    # 6. δ of the convective divergence:
    #       δ[(V·∇u) + u (∇·V)]
    #     = (δV·∇u) + (V·δ∇u) + (δu)(∇·V) + u (δ∇·V)
    div_V  = (np.einsum("idc,icj->idj", JV, gradWu))[..., 0, 0] \
           + (np.einsum("idc,icj->idj", JV, gradWu))[..., 1, 1]   # (n,)
    dconv = (
        np.einsum("ij,idj->id", dV, grad_u)
        + np.einsum("ij,idj->id", V, dgrad_u)
        + du_int * div_V[:, None]
        + u_int  * ddiv_V[:, None]
    )                                                             # (n, 2)

    # 7. δ of the reaction.
    JN = reaction.jacobian(u_int)                                 # (n, 2, 2)
    dN = np.einsum("idc,ic->id", JN, du_int)                      # (n, 2)

    # 8. δ of the regulariser. Since reg is linear in (X, G_tilde) and the
    #    boundary block is fixed, the perturbation only sees δX.
    if reg.mode == "laplacian":
        nu = np.asarray(reg.coeff, dtype=float)
        dreg = (nu @ (delta_X @ reg.Xa.T)).T                      # (n, 2)
    elif reg.mode == "bilaplacian":
        dreg = -float(reg.coeff) * (delta_X @ reg.Xa.T).T         # (n, 2)
    else:
        raise ValueError(f"Unknown regularization mode {reg.mode!r}.")

    # 9. Assemble δΦ.
    dphi = -dconv + dreg + dN                                     # (n, 2)
    return dphi.T


# ---------------------------------------------------------------------------
# Closure for use with scipy.sparse.linalg.LinearOperator (GMRES)
# ---------------------------------------------------------------------------

def make_frechet_operator(
    X: np.ndarray,
    G: np.ndarray,
    mats: RBFInterpolationMatrices,
    nl_ops: NonlocalOperators,
    reg: RegularizationOperator,
    velocity: Velocity,
    reaction: Reaction,
    *,
    bdf_factor: float = 1.0,
) -> LinearOperator:
    r"""
    Wrap :func:`frechet_derivative` as a SciPy ``LinearOperator`` of size
    ``2 n``.

    The Newton system at a given BDF step has the form

    .. math::
        (I - \mathrm{bdf\_factor} \cdot \partial_X \Phi)\,\delta X = r,

    where ``bdf_factor = dt * gamma_0`` for the chosen BDF order and ``r``
    is the current Newton residual. This function returns the operator
    that maps :math:`\delta X` to the left-hand side.

    Parameters
    ----------
    X : np.ndarray, shape ``(2, n)``
        Current Newton iterate.
    G, mats, nl_ops, reg, velocity, reaction
        Same meaning as in :func:`frechet_derivative`.
    bdf_factor : float
        ``dt * gamma_0`` for the active BDF order. Defaults to 1, which
        gives the bare Frechet derivative.

    Returns
    -------
    LinearOperator
        Acts on flat vectors of length ``2 n``; the user is expected to
        flatten/reshape via ``np.reshape`` on the way in and out, which
        ``time_integration.newton_krylov_solve`` handles.
    """
    n = mats.n
    size = 2 * n

    def matvec(v):
        delta_X = v.reshape(2, n)
        dphi = frechet_derivative(delta_X, X, G, mats, nl_ops, reg, velocity, reaction)
        return (delta_X - bdf_factor * dphi).reshape(-1)

    return LinearOperator((size, size), matvec=matvec, dtype=float)


# ---------------------------------------------------------------------------
# Lazy-cached interior-evaluation gradient matrices
# ---------------------------------------------------------------------------
#
# `compute_basis_gradients(mats.interior_points, mats)` is independent of t
# and X, so we compute it once and cache it on the matrices object. This
# avoids redoing an O(n^2 * Q) call on every RHS evaluation (i.e. inside
# every Newton iteration of every BDF step).

_GRAD_CACHE_ATTR = "_interior_basis_gradient_cache"


def _get_interior_gradient_matrices(mats: RBFInterpolationMatrices):
    cached = getattr(mats, _GRAD_CACHE_ATTR, None)
    if cached is not None:
        return cached
    # Defer the import to break a potential cycle.
    from .matrices import compute_basis_gradients
    grad_a, grad_b = compute_basis_gradients(mats.interior_points, mats)
    setattr(mats, _GRAD_CACHE_ATTR, (grad_a, grad_b))
    return grad_a, grad_b