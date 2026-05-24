"""
Radial basis functions used by the rbf_nonlocal package.

Two kernels are provided, selectable by a single ``kind`` keyword:

* ``"tension"``    — the radial basis function under tension,

  .. math::
      \\phi(r) = -\\frac{1}{2\\tau^3}\\bigl(e^{-\\tau r} + \\tau r\\bigr),

  conditionally positive definite of order 1 (parameter :math:`\\tau > 0`),
  :math:`C^2` on :math:`\\mathbb{R}_+`. Used for the diffusive case
  :math:`\\mathcal{L}u = -\\nu\\,\\Delta u`.

* ``"thin_plate"`` — the polyharmonic / thin-plate spline,

  .. math::
      \\phi(r) = r^4 \\ln r,\\qquad \\phi(0)=0,

  conditionally positive definite of order 3, :math:`C^3` on
  :math:`\\mathbb{R}_+`. Used for the hyper-viscosity case
  :math:`\\mathcal{L}u = \\varepsilon\\,\\Delta^2 u`, where the bi-Laplacian
  of the basis must be well defined.

For each kernel the module exposes :func:`phi`, :func:`phi_prime`,
:func:`laplacian_2d`, and :func:`bilaplacian_2d`. All routines are vectorised
in ``r`` and safe at ``r = 0`` (no ``NaN`` / ``Inf`` is returned at the origin
unless the underlying expression genuinely diverges, which only happens for
the bi-Laplacian of the thin-plate spline; in that case the result is clipped
at a small ``eps``).

References
----------
Y. Melouani, A. Bouhamidi, I. El Harraki. *A Meshless Radial Basis Function
Method for Nonlocal Balance Equations*, Section 2 (Table of usual RBFs).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "phi",
    "phi_prime",
    "laplacian_2d",
    "bilaplacian_2d",
    "supported_kinds",
    "needs_tau",
]


# ---------------------------------------------------------------------------
# Tension RBF:   phi(r) = -(exp(-tau r) + tau r) / (2 tau^3)
# ---------------------------------------------------------------------------

def _phi_tension(r: np.ndarray, tau: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return -(np.exp(-tau * r) + tau * r) / (2.0 * tau**3)


def _phi_prime_tension(r: np.ndarray, tau: float) -> np.ndarray:
    r"""d/dr [phi] = -(1 - exp(-tau r)) / (2 tau^2). Vanishes as r -> 0."""
    r = np.asarray(r, dtype=float)
    return -(1.0 - np.exp(-tau * r)) / (2.0 * tau**2)


def _laplacian_tension_2d(r: np.ndarray, tau: float) -> np.ndarray:
    r"""
    2-D Laplacian of the radial tension RBF:

        Δφ(r) = φ''(r) + φ'(r)/r
              = -[τ r e^{-τr} + (1 - e^{-τr})] / (2 τ^2 r).

    The point r = 0 is a removable singularity; from a Taylor expansion,

        Δφ(r) = -1/τ + (3/4) r + O(r^2),

    which is used near the origin to avoid 0/0.
    """
    r = np.asarray(r, dtype=float)
    out = np.empty_like(r)

    # Threshold below which the Taylor expansion is more accurate than
    # the closed form (which is 0/0 in floating point).
    small = 1.0e-6 / tau
    big_mask = r >= small

    if np.any(big_mask):
        rb = r[big_mask]
        e = np.exp(-tau * rb)
        out[big_mask] = -(tau * rb * e + (1.0 - e)) / (2.0 * tau**2 * rb)

    sm_mask = ~big_mask
    if np.any(sm_mask):
        rs = r[sm_mask]
        out[sm_mask] = -1.0 / tau + 0.75 * rs

    return out


def _bilaplacian_tension_2d(r: np.ndarray, tau: float) -> np.ndarray:
    """The tension RBF is only C^2; its bi-Laplacian is not collocation-safe."""
    raise NotImplementedError(
        "The tension RBF is C^2; its bi-Laplacian is not used by the "
        "collocation method. Use kind='thin_plate' for hyper-viscosity."
    )


# ---------------------------------------------------------------------------
# Thin-plate spline:   phi(r) = r^4 log(r)
# ---------------------------------------------------------------------------

def _phi_thin_plate(r: np.ndarray, tau: float | None = None) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    mask = r > 1.0e-15
    out[mask] = r[mask] ** 4 * np.log(r[mask])
    return out


def _phi_prime_thin_plate(r: np.ndarray, tau: float | None = None) -> np.ndarray:
    r"""d/dr [r^4 log r] = r^3 (4 log r + 1). Vanishes as r -> 0."""
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    mask = r > 1.0e-15
    out[mask] = r[mask] ** 3 * (4.0 * np.log(r[mask]) + 1.0)
    return out


def _laplacian_thin_plate_2d(r: np.ndarray, tau: float | None = None) -> np.ndarray:
    r"""Δ(r^4 log r) = 16 r^2 log r + 8 r^2 in 2-D. Equals 0 at r = 0."""
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    mask = r > 1.0e-15
    rm = r[mask]
    out[mask] = 16.0 * rm**2 * np.log(rm) + 8.0 * rm**2
    return out


def _bilaplacian_thin_plate_2d(
    r: np.ndarray, tau: float | None = None, eps: float = 1.0e-8
) -> np.ndarray:
    r"""
    Δ^2(r^4 log r) = 64 log(r) + 96 in 2-D.

    Diverges as r -> 0; the input ``r`` is clipped at ``eps`` so the value
    remains finite. The clipping is benign because the singularity is
    integrable and is never sampled exactly at ``r = 0`` away from the
    diagonal, while the on-diagonal contribution is absorbed into the
    interpolation matrix.
    """
    r = np.asarray(r, dtype=float)
    r_safe = np.maximum(r, eps)
    return 64.0 * np.log(r_safe) + 96.0


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

_KINDS = {
    "tension": {
        "phi":         _phi_tension,
        "phi_prime":   _phi_prime_tension,
        "laplacian":   _laplacian_tension_2d,
        "bilaplacian": _bilaplacian_tension_2d,
        "needs_tau":   True,
    },
    "thin_plate": {
        "phi":         _phi_thin_plate,
        "phi_prime":   _phi_prime_thin_plate,
        "laplacian":   _laplacian_thin_plate_2d,
        "bilaplacian": _bilaplacian_thin_plate_2d,
        "needs_tau":   False,
    },
}


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(
            f"Unknown RBF kind {kind!r}. Supported kinds: {tuple(_KINDS)}."
        )


def _call(kind: str, what: str, r, tau):
    _check_kind(kind)
    f = _KINDS[kind][what]
    return f(r, tau) if _KINDS[kind]["needs_tau"] else f(r)


def phi(r, kind: str = "tension", tau: float = 1.0):
    """RBF value :math:`\\phi(r)`. ``tau`` is ignored when ``kind='thin_plate'``."""
    return _call(kind, "phi", r, tau)


def phi_prime(r, kind: str = "tension", tau: float = 1.0):
    """First radial derivative :math:`\\phi'(r)`."""
    return _call(kind, "phi_prime", r, tau)


def laplacian_2d(r, kind: str = "tension", tau: float = 1.0):
    """2-D Laplacian :math:`\\Delta\\phi(r)` of the radial RBF."""
    return _call(kind, "laplacian", r, tau)


def bilaplacian_2d(r, kind: str = "thin_plate", tau: float = 1.0):
    """
    2-D bi-Laplacian :math:`\\Delta^2\\phi(r)` of the radial RBF.

    Only implemented for ``kind='thin_plate'``; calling it with
    ``kind='tension'`` raises :class:`NotImplementedError` because the
    tension RBF lacks the regularity required for collocation of
    fourth-order operators.
    """
    return _call(kind, "bilaplacian", r, tau)


def supported_kinds() -> tuple[str, ...]:
    """Return the tuple of supported RBF kinds."""
    return tuple(_KINDS.keys())


def needs_tau(kind: str) -> bool:
    """True if ``kind`` uses a tension parameter :math:`\\tau`."""
    _check_kind(kind)
    return _KINDS[kind]["needs_tau"]