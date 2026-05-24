"""
Two-dimensional irregular domains for the rbf_nonlocal package.

Two domains are provided:

* :class:`FlowerDomain` — flower-shaped domain centred at (0.5, 0.5),

  .. math::
      r(\\theta) = a + b\\cos(n_{\\rm petals}\\,\\theta).

  Defaults (n_petals=5, a=0.3, b=0.15) reproduce the geometry used in the
  paper.

* :class:`GearDomain` — gear-shaped domain with smooth cosine-profile teeth,
  parametrised by ``n_teeth``, ``r_base``, ``tooth_height``, ``tooth_width``.
  Defaults (12, 0.2, 0.15, 0.6) reproduce the paper's geometry.

Each domain exposes the same interface (``radius``, ``contains``,
``generate_points``, ``boundary_curve``, ``bounding_box``), so the rest of
the package can be written in a geometry-agnostic way.

A :func:`make_domain` factory is provided for string-based selection from
configuration files or the command line, e.g. ``make_domain("flower")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "Domain",
    "FlowerDomain",
    "GearDomain",
    "make_domain",
    "halton_sequence",
]


# ---------------------------------------------------------------------------
# Halton sequence
# ---------------------------------------------------------------------------

def halton_sequence(n: int, base: int, skip: int = 20) -> np.ndarray:
    """
    Generate ``n`` points of the 1-D van der Corput / Halton sequence in
    base ``base``, skipping the first ``skip`` indices for better uniformity
    near the origin.

    Parameters
    ----------
    n : int
        Number of points to generate.
    base : int
        Prime base of the sequence (use distinct primes for distinct
        coordinates: 2, 3, 5, 7, 11, 13, ...).
    skip : int, optional
        Number of leading sequence elements to discard.

    Returns
    -------
    np.ndarray of shape ``(n,)``, with entries in ``[0, 1)``.
    """
    seq = np.zeros(n, dtype=float)
    for i in range(n):
        f = 1.0
        r = 0.0
        idx = i + skip + 1
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        seq[i] = r
    return seq


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Domain:
    """
    Abstract base class for a 2-D irregular domain centred near (0.5, 0.5).

    Concrete subclasses must implement :meth:`radius`, :meth:`contains`,
    and :meth:`generate_points`. They must also expose ``self.center`` as a
    ``(cx, cy)`` tuple.
    """

    name: str = "abstract"

    # -------- helpers usable from the base class --------

    def boundary_curve(self, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(x, y)`` arrays tracing the domain boundary."""
        theta = np.linspace(0.0, 2.0 * np.pi, n_points)
        r = self.radius(theta)
        cx, cy = self.center
        return cx + r * np.cos(theta), cy + r * np.sin(theta)

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box ``(x_min, x_max, y_min, y_max)``."""
        x, y = self.boundary_curve(n_points=720)
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())

    # -------- methods to be implemented by subclasses --------

    def radius(self, theta):  # pragma: no cover - abstract
        raise NotImplementedError

    def contains(self, x, y):  # pragma: no cover - abstract
        raise NotImplementedError

    def generate_points(  # pragma: no cover - abstract
        self, n_interior: int, n_boundary: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Flower domain
# ---------------------------------------------------------------------------

@dataclass
class FlowerDomain(Domain):
    """
    Flower-shaped domain ``r(theta) = a + b cos(n_petals * theta)``.

    Parameters
    ----------
    n_petals : int
        Number of petals (default 5).
    a, b : float
        Mean radius and oscillation amplitude. The boundary curve is
        ``a + b cos(n_petals * theta)``; positivity requires ``a > b``.
    center : (float, float)
        Domain centre, default ``(0.5, 0.5)`` to fit the unit square.
    margin : float
        Minimum distance an interior point must keep from the boundary.
        A small positive value avoids ill-conditioned RBF systems.
    """

    n_petals: int = 5
    a: float = 0.3
    b: float = 0.15
    center: Tuple[float, float] = (0.5, 0.5)
    margin: float = 0.02

    name: str = "flower"

    def __post_init__(self) -> None:
        if self.a <= self.b:
            raise ValueError(
                f"FlowerDomain requires a > b for a positive radius "
                f"(got a={self.a}, b={self.b})."
            )

    def radius(self, theta):
        theta = np.asarray(theta, dtype=float)
        return self.a + self.b * np.cos(self.n_petals * theta)

    def contains(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cx, cy = self.center
        x_c, y_c = x - cx, y - cy
        r = np.sqrt(x_c**2 + y_c**2)
        theta = np.arctan2(y_c, x_c)
        return r <= self.radius(theta)

    def generate_points(
        self,
        n_interior: int,
        n_boundary: int,
        candidate_factor: int = 6,
        halton_base_x: int = 2,
        halton_base_y: int = 3,
        halton_base_theta: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate quasi-random interior and boundary collocation points.

        Interior points are rejection-sampled from a 2-D Halton sequence on
        ``(0,1)^2`` and kept only if they sit at least ``self.margin``
        inside the boundary. Boundary points come from a separate Halton
        sequence in ``theta``.

        Returns
        -------
        interior : np.ndarray, shape ``(n_interior, 2)``
        boundary : np.ndarray, shape ``(n_boundary, 2)``
        """
        cx, cy = self.center

        # --- Interior points ---
        n_candidates = max(n_interior * candidate_factor, n_interior + 50)
        hx = halton_sequence(n_candidates, halton_base_x)
        hy = halton_sequence(n_candidates, halton_base_y)

        x_c = hx - cx
        y_c = hy - cy
        r = np.sqrt(x_c**2 + y_c**2)
        theta = np.arctan2(y_c, x_c)
        keep = self.radius(theta) - r >= self.margin

        idx_keep = np.flatnonzero(keep)
        if idx_keep.size < n_interior:
            raise RuntimeError(
                f"FlowerDomain: only {idx_keep.size} of {n_interior} interior "
                f"points satisfied the margin constraint; increase "
                f"candidate_factor (currently {candidate_factor})."
            )
        take = idx_keep[:n_interior]
        interior = np.column_stack([hx[take], hy[take]])

        # --- Boundary points (Halton in theta) ---
        ht = halton_sequence(n_boundary, halton_base_theta)
        thetas = 2.0 * np.pi * ht
        rb = self.radius(thetas)
        boundary = np.column_stack([cx + rb * np.cos(thetas),
                                    cy + rb * np.sin(thetas)])

        return interior, boundary


# ---------------------------------------------------------------------------
# Gear domain
# ---------------------------------------------------------------------------

@dataclass
class GearDomain(Domain):
    """
    Gear-shaped domain with smooth cosine-profile teeth.

    Parameters
    ----------
    n_teeth : int
        Number of teeth around the gear.
    r_base : float
        Base (root) radius.
    tooth_height : float
        Additional radius at the top of a tooth.
    tooth_width : float
        Fraction of each tooth period occupied by the tooth proper, in
        ``(0, 1)``. Smaller values give narrower teeth with wider gaps.
    center : (float, float)
        Domain centre, default ``(0.5, 0.5)`` to fit the unit square.
    """

    n_teeth: int = 12
    r_base: float = 0.20
    tooth_height: float = 0.15
    tooth_width: float = 0.60
    center: Tuple[float, float] = (0.5, 0.5)

    name: str = "gear"

    def __post_init__(self) -> None:
        if not 0.0 < self.tooth_width < 1.0:
            raise ValueError(
                f"GearDomain requires 0 < tooth_width < 1 (got {self.tooth_width})."
            )
        if self.r_base <= 0 or self.tooth_height < 0:
            raise ValueError("GearDomain requires r_base > 0 and tooth_height >= 0.")

    def radius(self, theta):
        """
        Vectorised gear radius. Inside each tooth period, the central
        ``tooth_width`` fraction carries a raised-cosine bump of height
        ``tooth_height``; the rest stays at ``r_base``.
        """
        theta = np.asarray(theta, dtype=float)
        scalar_input = theta.ndim == 0
        if scalar_input:
            theta = theta[np.newaxis]

        tooth_period = 2.0 * np.pi / self.n_teeth
        normalized_phase = (theta % tooth_period) / tooth_period

        r = np.full_like(theta, self.r_base, dtype=float)

        tooth_start = (1.0 - self.tooth_width) / 2.0
        tooth_end = (1.0 + self.tooth_width) / 2.0
        on_tooth = (normalized_phase >= tooth_start) & (normalized_phase < tooth_end)

        if np.any(on_tooth):
            tooth_local = (normalized_phase[on_tooth] - tooth_start) / self.tooth_width
            height = self.tooth_height * (
                0.5 + 0.5 * np.cos(2.0 * np.pi * (tooth_local - 0.5))
            )
            r[on_tooth] = self.r_base + height

        return r.item() if scalar_input else r

    def contains(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cx, cy = self.center
        x_c, y_c = x - cx, y - cy
        r = np.sqrt(x_c**2 + y_c**2)
        theta = np.arctan2(y_c, x_c) % (2.0 * np.pi)
        return r <= self.radius(theta)

    def generate_points(
        self,
        n_interior: int,
        n_boundary: int,
        candidate_factor: int = 8,
        max_batches: int = 10,
        boundary_perturb_base: int = 11,
        boundary_perturb_strength: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate quasi-random interior and boundary collocation points.

        Interior points are rejection-sampled from a 2-D Halton sequence in
        the gear's bounding box; if a single batch does not yield enough
        in-domain points, more batches are drawn (with shifted skip values)
        up to ``max_batches`` times.

        Boundary points use stratified angles (one per segment of width
        ``2 pi / n_boundary``) with a small Halton perturbation inside each
        segment, controlled by ``boundary_perturb_strength`` in ``[0, 1]``.
        This guarantees full angular coverage, which matters for the
        gear's sharp tooth-to-valley transitions.

        Returns
        -------
        interior : np.ndarray, shape ``(n_interior, 2)``
        boundary : np.ndarray, shape ``(n_boundary, 2)``
        """
        cx, cy = self.center
        r_max = self.r_base + self.tooth_height
        x_min = max(0.0, cx - r_max)
        x_max = min(1.0, cx + r_max)
        y_min = max(0.0, cy - r_max)
        y_max = min(1.0, cy + r_max)

        # --- Interior points ---
        n_candidates = max(n_interior * candidate_factor, n_interior + 50)
        kept_x: list = []
        kept_y: list = []

        for batch in range(max_batches):
            hx = halton_sequence(n_candidates, 2, skip=20 + batch * 50)
            hy = halton_sequence(n_candidates, 3, skip=20 + batch * 50)
            xs = x_min + hx * (x_max - x_min)
            ys = y_min + hy * (y_max - y_min)

            inside = self.contains(xs, ys)
            idx = np.flatnonzero(inside)
            need = n_interior - len(kept_x)
            take = idx[:need]
            kept_x.extend(xs[take].tolist())
            kept_y.extend(ys[take].tolist())

            if len(kept_x) >= n_interior:
                break

        if len(kept_x) < n_interior:
            raise RuntimeError(
                f"GearDomain: only {len(kept_x)} of {n_interior} interior "
                f"points generated after {max_batches} batches; increase "
                f"candidate_factor or max_batches."
            )

        interior = np.column_stack([kept_x[:n_interior], kept_y[:n_interior]])

        # --- Boundary points: stratified angles + Halton perturbation ---
        if not 0.0 <= boundary_perturb_strength <= 1.0:
            raise ValueError(
                "boundary_perturb_strength must be in [0, 1] "
                f"(got {boundary_perturb_strength})."
            )

        base_angles = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
        h_perturb = halton_sequence(n_boundary, boundary_perturb_base, skip=30)
        segment_width = 2.0 * np.pi / n_boundary
        thetas = (
            base_angles
            + (h_perturb - 0.5) * segment_width * boundary_perturb_strength
        ) % (2.0 * np.pi)

        rb = self.radius(thetas)
        boundary = np.column_stack([cx + rb * np.cos(thetas),
                                    cy + rb * np.sin(thetas)])

        return interior, boundary


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "flower": FlowerDomain,
    "gear":   GearDomain,
}


def make_domain(name: str, **kwargs) -> Domain:
    """
    Construct a domain by name.

    Parameters
    ----------
    name : str
        ``"flower"`` or ``"gear"`` (case-insensitive).
    **kwargs
        Forwarded to the underlying domain class.

    Returns
    -------
    Domain
        A concrete domain instance.

    Examples
    --------
    >>> dom = make_domain("flower", n_petals=6)
    >>> interior, boundary = dom.generate_points(80, 80)
    >>> dom = make_domain("gear", n_teeth=8, tooth_width=0.5)
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown domain {name!r}. Available: {tuple(_REGISTRY)}."
        )
    return _REGISTRY[key](**kwargs)