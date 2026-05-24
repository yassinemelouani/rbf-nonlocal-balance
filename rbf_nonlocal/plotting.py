"""
Publication-quality plotting utilities for rbf_nonlocal.

Solutions are visualised by **interpolating** the RBF approximation onto a
fine regular grid (default 300x300) and masking points outside the
irregular domain. This decouples the visual resolution from the number of
collocation nodes, giving smooth contours and 3-D surfaces irrespective of
the underlying mesh size.

Figure types
------------
* :func:`plot_collocation_points` — interior + boundary nodes with the
  domain boundary.
* :func:`plot_solution_field` — 2-D contour of one solution component
  (numerical or exact).
* :func:`plot_relative_error_field` — log-scale relative-error contour
  (validation mode only).
* :func:`plot_solution_3d` — 3-D surface with a domain-footprint shadow
  on the floor.
* :func:`plot_error_history` — :math:`L^\\infty` and :math:`L^2` errors
  vs time on a log scale.
* :func:`plot_iteration_diagnostics` — Newton and GMRES iteration counts
  per BDF step.
* :func:`plot_solution` — dispatcher producing the full default set.

Output
------
Every figure function accepts ``save_dir`` and ``formats``. With
``save_dir=None`` (the default) figures are shown via ``plt.show()``;
otherwise they are written to ``<save_dir>/<filename>.<ext>`` for each
``ext`` in ``formats``. To produce paper-ready output use, e.g.,
``formats=("pdf", "png")``.

Styling
-------
A serif-font, 300 DPI, ``bbox_inches='tight'`` style is applied to each
figure via :func:`matplotlib.rc_context`, so it does not leak into the
caller's matplotlib state.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

from .matrices import evaluate_basis_functions

if TYPE_CHECKING:
    from .solver import NonlocalBalanceSolver, SolverResult


__all__ = [
    "plot_collocation_points",
    "plot_solution_field",
    "plot_relative_error_field",
    "plot_solution_3d",
    "plot_error_history",
    "plot_iteration_diagnostics",
    "plot_solution",
]


# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------
#
# Applied per-figure via plt.rc_context so it does not pollute the user's
# global matplotlib state.

_PAPER_STYLE = {
    "font.family":      "serif",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.1,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_step(result: "SolverResult", which: Union[str, int]) -> int:
    """Translate ``which`` ('initial'/'final'/int) to a non-negative index."""
    if isinstance(which, int):
        n_t = len(result.times)
        if not -n_t <= which < n_t:
            raise IndexError(
                f"step index {which} out of range for {n_t} time points."
            )
        return which % n_t
    if which == "initial":
        return 0
    if which == "final":
        return len(result.times) - 1
    raise ValueError(
        f"which must be 'initial', 'final', or an int (got {which!r})."
    )


def _bbox_with_pad(solver: "NonlocalBalanceSolver", pad_frac: float = 0.05):
    xmin, xmax, ymin, ymax = solver.domain.bounding_box()
    pad = pad_frac * max(xmax - xmin, ymax - ymin)
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _make_grid(solver: "NonlocalBalanceSolver", resolution: int):
    """Regular grid spanning the domain's padded bounding box."""
    xmin, xmax, ymin, ymax = _bbox_with_pad(solver, pad_frac=0.02)
    grid_x, grid_y = np.mgrid[
        xmin:xmax:resolution * 1j,
        ymin:ymax:resolution * 1j,
    ]
    return grid_x, grid_y


def _evaluate_numerical_on_grid(
    solver: "NonlocalBalanceSolver",
    X: np.ndarray,                 # (2, n)
    G: np.ndarray,                 # (2, n_prime)
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    chunk_size: int = 50_000,
):
    """
    Evaluate the RBF interpolant on the grid, masking points outside the
    domain. Evaluation is done only at in-domain points and processed in
    chunks so peak memory is bounded even at high resolutions.

    Returns
    -------
    u_num : np.ndarray, shape ``(grid.shape[0], grid.shape[1], 2)``
        With NaN outside the domain.
    mask : np.ndarray, bool, shape ``(grid.shape[0], grid.shape[1])``
        ``True`` outside the domain.
    """
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    n_grid = grid_points.shape[0]

    in_domain = solver.domain.contains(grid_points[:, 0], grid_points[:, 1])
    mask = ~in_domain.reshape(grid_x.shape)

    G_tilde = np.hstack([G, np.zeros((2, solver.mats.dm))])

    inside_idx = np.flatnonzero(in_domain)
    inside_pts = grid_points[inside_idx]

    u_inside = np.empty((len(inside_idx), 2), dtype=float)
    for start in range(0, len(inside_idx), chunk_size):
        end = min(start + chunk_size, len(inside_idx))
        a_chunk, b_chunk = evaluate_basis_functions(
            inside_pts[start:end], solver.mats,
        )
        # u_h(x) = a(x)^T X[d] + b(x)^T G_tilde[d]
        u_inside[start:end] = a_chunk @ X.T + b_chunk @ G_tilde.T

    u_flat = np.full((n_grid, 2), np.nan, dtype=float)
    u_flat[inside_idx] = u_inside
    u_num = u_flat.reshape(grid_x.shape + (2,))
    return u_num, mask


def _evaluate_exact_on_grid(
    solver: "NonlocalBalanceSolver",
    t: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Manufactured solution on the grid; returns ``None`` in simulation mode."""
    if solver.exact is None:
        return None
    u_exact = solver.exact.u(t, grid_x, grid_y).copy()  # (res, res, 2)
    u_exact[mask, :] = np.nan
    return u_exact


def _save_or_show(
    fig,
    save_dir: Optional[str],
    name: str,
    formats: Sequence[str] = ("png",),
) -> None:
    if save_dir is None:
        plt.show()
        return
    os.makedirs(save_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(save_dir, f"{name}.{fmt}")
        fig.savefig(path, format=fmt)
    plt.close(fig)


def _set_axes_2d(ax, solver: "NonlocalBalanceSolver") -> None:
    """Equal aspect ratio + padded bbox limits common to every 2D figure."""
    xmin, xmax, ymin, ymax = _bbox_with_pad(solver)
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def _draw_boundary_2d(ax, solver: "NonlocalBalanceSolver", **kwargs) -> None:
    bx, by = solver.domain.boundary_curve(n_points=720)
    ax.plot(bx, by, **{"color": "k", "linewidth": 1.5, **kwargs})


def _component_cmap(component: int, override: Optional[str] = None) -> str:
    if override is not None:
        return override
    return "viridis" if component == 0 else "plasma"


# ---------------------------------------------------------------------------
# 1. Collocation points
# ---------------------------------------------------------------------------

def plot_collocation_points(
    solver: "NonlocalBalanceSolver",
    *,
    save_dir: Optional[str] = None,
    filename: str = "collocation_points",
    formats: Sequence[str] = ("png",),
) -> "plt.Figure":
    """Scatter of interior and boundary collocation nodes with the boundary curve."""
    with plt.rc_context(_PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_boundary_2d(ax, solver, label="Domain boundary")
        ax.scatter(
            solver.interior_points[:, 0], solver.interior_points[:, 1],
            c="darkred", marker="x", s=20, alpha=0.7,
            label="Interior points",
        )
        ax.scatter(
            solver.boundary_points[:, 0], solver.boundary_points[:, 1],
            facecolors="none", edgecolors="darkblue", marker="o",
            s=25, linewidths=1, label="Boundary points",
        )
        ax.set_title("Collocation points distribution")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        _set_axes_2d(ax, solver)
        fig.tight_layout()
        _save_or_show(fig, save_dir, filename, formats)
        return fig


# ---------------------------------------------------------------------------
# 2. Solution field, 2-D
# ---------------------------------------------------------------------------

def plot_solution_field(
    solver: "NonlocalBalanceSolver",
    result: "SolverResult",
    component: int,
    *,
    mode: str = "numerical",
    step: Union[str, int] = "final",
    resolution: int = 300,
    n_levels: int = 20,
    cmap: Optional[str] = None,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    formats: Sequence[str] = ("png",),
) -> "plt.Figure":
    """
    Filled-contour plot of a single solution component, evaluated on a
    fine grid via the RBF interpolant.

    Parameters
    ----------
    component : 0 or 1
    mode : ``"numerical"`` or ``"exact"``. The latter requires validation mode.
    step : ``"initial"``, ``"final"``, or an integer step index.
    resolution : grid resolution per axis.
    """
    if component not in (0, 1):
        raise ValueError(f"component must be 0 or 1 (got {component}).")
    if mode == "exact" and solver.exact is None:
        raise RuntimeError("mode='exact' requires validation mode.")

    step_idx = _select_step(result, step)
    t = float(result.times[step_idx])

    grid_x, grid_y = _make_grid(solver, resolution)
    u_num, mask = _evaluate_numerical_on_grid(
        solver, result.history[step_idx], result.boundary_history[step_idx],
        grid_x, grid_y,
    )
    if mode == "numerical":
        Z = u_num[:, :, component]
        title_suffix = "Numerical"
    else:
        u_exact = _evaluate_exact_on_grid(solver, t, grid_x, grid_y, mask)
        Z = u_exact[:, :, component]
        title_suffix = "Exact"

    cmap_used = _component_cmap(component, cmap)

    with plt.rc_context(_PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
        if zmax <= zmin:
            zmax = zmin + 1.0e-12  # degenerate constant field
        levels = np.linspace(zmin, zmax, n_levels)
        cs = ax.contourf(grid_x, grid_y, Z, levels=levels, cmap=cmap_used)
        ax.contour(
            grid_x, grid_y, Z, levels=levels[::4],
            colors="black", alpha=0.3, linewidths=0.5,
        )
        _draw_boundary_2d(ax, solver)
        ax.set_title(f"$u_{component+1}$ ({title_suffix}) at $t={t:.3f}$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        cbar = plt.colorbar(cs, ax=ax, label=f"$u_{component+1}$")
        cbar.ax.tick_params(labelsize=9)
        _set_axes_2d(ax, solver)
        fig.tight_layout()
        if filename is None:
            filename = f"u{component+1}_{mode}_contour"
        _save_or_show(fig, save_dir, filename, formats)
        return fig


# ---------------------------------------------------------------------------
# 3. Relative-error field, 2-D
# ---------------------------------------------------------------------------

def plot_relative_error_field(
    solver: "NonlocalBalanceSolver",
    result: "SolverResult",
    component: int,
    *,
    step: Union[str, int] = "final",
    resolution: int = 300,
    err_floor: float = 1.0e-6,
    err_ceiling: float = 1.0,
    n_levels: int = 20,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    formats: Sequence[str] = ("png",),
) -> "plt.Figure":
    """
    Log-scale pointwise relative-error contour, validation mode only.

    The colormap ranges from ``err_floor`` to ``err_ceiling``, both on a
    log scale, matching the convention in the paper figures.
    """
    if component not in (0, 1):
        raise ValueError(f"component must be 0 or 1 (got {component}).")
    if solver.exact is None:
        raise RuntimeError(
            "Relative-error plot requires validation mode (no exact "
            "solution is available)."
        )

    step_idx = _select_step(result, step)
    t = float(result.times[step_idx])

    grid_x, grid_y = _make_grid(solver, resolution)
    u_num, mask = _evaluate_numerical_on_grid(
        solver, result.history[step_idx], result.boundary_history[step_idx],
        grid_x, grid_y,
    )
    u_exact = _evaluate_exact_on_grid(solver, t, grid_x, grid_y, mask)

    rel_err = np.abs(u_num[:, :, component] - u_exact[:, :, component]) / (
        np.abs(u_exact[:, :, component]) + 1.0e-10
    )
    rel_err[mask] = np.nan
    # Clip below floor for stable LogNorm.
    rel_err = np.where(rel_err < err_floor, err_floor, rel_err)

    with plt.rc_context(_PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        levels = np.logspace(np.log10(err_floor), np.log10(err_ceiling), n_levels)
        cs = ax.contourf(
            grid_x, grid_y, rel_err,
            levels=levels, cmap="hot",
            norm=mcolors.LogNorm(vmin=err_floor, vmax=err_ceiling),
        )
        _draw_boundary_2d(ax, solver)
        ax.set_title(f"Relative error in $u_{component+1}$ at $t={t:.3f}$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        cbar = plt.colorbar(cs, ax=ax, label="Relative error")
        cbar.ax.tick_params(labelsize=9)
        _set_axes_2d(ax, solver)
        fig.tight_layout()
        if filename is None:
            filename = f"u{component+1}_relative_error"
        _save_or_show(fig, save_dir, filename, formats)
        return fig


# ---------------------------------------------------------------------------
# 4. Solution surface, 3-D
# ---------------------------------------------------------------------------

def plot_solution_3d(
    solver: "NonlocalBalanceSolver",
    result: "SolverResult",
    component: int,
    *,
    mode: str = "numerical",
    step: Union[str, int] = "final",
    resolution: int = 200,
    cmap: Optional[str] = None,
    elev: float = 25.0,
    azim: float = 45.0,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    formats: Sequence[str] = ("png",),
) -> "plt.Figure":
    """
    3-D surface plot with the domain footprint drawn as a shadow on the
    floor and the boundary curve traced over it.

    A lower default ``resolution`` (200) is used because matplotlib's
    ``plot_surface`` slows down sharply with quad count.
    """
    if component not in (0, 1):
        raise ValueError(f"component must be 0 or 1 (got {component}).")
    if mode == "exact" and solver.exact is None:
        raise RuntimeError("mode='exact' requires validation mode.")

    step_idx = _select_step(result, step)
    t = float(result.times[step_idx])

    grid_x, grid_y = _make_grid(solver, resolution)
    u_num, mask = _evaluate_numerical_on_grid(
        solver, result.history[step_idx], result.boundary_history[step_idx],
        grid_x, grid_y,
    )
    if mode == "numerical":
        Z = u_num[:, :, component]
        title_suffix = "Numerical"
    else:
        u_exact = _evaluate_exact_on_grid(solver, t, grid_x, grid_y, mask)
        Z = u_exact[:, :, component]
        title_suffix = "Exact"

    Z_masked = np.ma.masked_invalid(Z)
    cmap_used = _component_cmap(component, cmap)

    with plt.rc_context(_PAPER_STYLE):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        zmin = float(np.nanmin(Z))
        zmax = float(np.nanmax(Z))
        z_range = zmax - zmin if zmax > zmin else 1.0
        z_floor = zmin - 0.10 * z_range

        surf = ax.plot_surface(
            grid_x, grid_y, Z_masked,
            cmap=cmap_used, edgecolor="none", alpha=0.95,
            linewidth=0, antialiased=True, shade=True,
            vmin=zmin, vmax=zmax,
        )

        # Domain footprint shadow on the floor.
        shadow = np.full_like(grid_x, z_floor)
        shadow[mask] = np.nan
        try:
            eps = max(abs(z_floor) * 1.0e-6, 1.0e-6)
            ax.contourf(
                grid_x, grid_y, shadow,
                levels=[z_floor - eps, z_floor + eps],
                zdir="z", offset=z_floor,
                colors=["gray"], alpha=0.3,
            )
        except Exception:
            # Some matplotlib versions are picky with degenerate fields;
            # the boundary trace below still gives the visual cue.
            pass

        # Boundary curve traced at the floor level.
        bx, by = solver.domain.boundary_curve(n_points=720)
        bz = np.full_like(bx, z_floor)
        ax.plot(bx, by, bz, "k-", linewidth=2, alpha=0.8)

        ax.set_title(f"$u_{component+1}$ ({title_suffix}) at $t={t:.3f}$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel(f"$u_{component+1}$")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 0.5])

        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
        cbar.ax.tick_params(labelsize=9)
        fig.tight_layout()
        if filename is None:
            filename = f"u{component+1}_{mode}_3d"
        _save_or_show(fig, save_dir, filename, formats)
        return fig


# ---------------------------------------------------------------------------
# 5. Error history (validation only)
# ---------------------------------------------------------------------------

def plot_error_history(
    solver: "NonlocalBalanceSolver",
    result: "SolverResult",
    *,
    save_dir: Optional[str] = None,
    filename: str = "error_history",
    formats: Sequence[str] = ("png",),
) -> "plt.Figure":
    """Relative L^inf and L^2 errors vs time, log-y. Validation mode only."""
    if result.err_inf is None or result.err_l2 is None:
        raise RuntimeError(
            "plot_error_history requires validation mode "
            "(result.err_inf / err_l2 are unset)."
        )

    with plt.rc_context(_PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(result.times, result.err_inf, "b-",  linewidth=1.5,
                    label=r"$\|u_h - u\|_\infty / \|u\|_\infty$")
        ax.semilogy(result.times, result.err_l2,  "r--", linewidth=1.5,
                    label=r"$\|u_h - u\|_2     / \|u\|_2$",     alpha=0.8)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Relative error")
        ax.set_title("Temporal evolution of relative errors")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, which="both", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_xlim([result.times[0], result.times[-1]])
        fig.tight_layout()
        _save_or_show(fig, save_dir, filename, formats)
        return fig


# ---------------------------------------------------------------------------
# 6. Iteration diagnostics
# ---------------------------------------------------------------------------

def plot_iteration_diagnostics(
    solver: "NonlocalBalanceSolver",
    result: "SolverResult",
    *,
    save_dir: Optional[str] = None,
    filename: str = "iteration_diagnostics",
    formats: Sequence[str] = ("png",),
) -> "plt.Figure":
    """Newton and GMRES iteration counts per BDF step."""
    with plt.rc_context(_PAPER_STYLE):
        steps = np.arange(1, len(result.bdf_orders) + 1)
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

        ax_top.plot(steps, result.newton_iters, marker=".", linestyle="-")
        ax_top.set_ylabel("Newton iterations")
        ax_top.grid(True, linewidth=0.5, alpha=0.5)

        ax_bot.plot(steps, result.gmres_iters, marker=".", linestyle="-", color="C1")
        ax_bot.set_xlabel("BDF step")
        ax_bot.set_ylabel("GMRES iterations (total)")
        ax_bot.grid(True, linewidth=0.5, alpha=0.5)

        fig.tight_layout()
        _save_or_show(fig, save_dir, filename, formats)
        return fig


# ---------------------------------------------------------------------------
# 7. Dispatcher
# ---------------------------------------------------------------------------

def plot_solution(
    solver: "NonlocalBalanceSolver",
    result: "SolverResult",
    *,
    save_dir: Optional[str] = None,
    resolution: int = 300,
    resolution_3d: int = 200,
    formats: Sequence[str] = ("png",),
    which: str = "final",
) -> None:
    """
    Produce the standard set of figures for a run.

    Always plots: collocation points; iteration diagnostics; numerical 2-D
    contour and 3-D surface for both components.

    In validation mode additionally plots: error history; exact 2-D
    contour and 3-D surface for both components; log-scale relative-error
    contour for both components.

    Parameters
    ----------
    save_dir : str, optional
        Directory in which to write all figures. If ``None``, each figure
        is shown via ``plt.show()``.
    resolution : int
        Grid resolution per axis for 2-D contours.
    resolution_3d : int
        Grid resolution per axis for 3-D surfaces (default 200; matplotlib's
        ``plot_surface`` is slow at high quad counts).
    formats : tuple of str
        File formats for ``save_dir`` output. Use ``("pdf", "png")`` for
        paper-ready output.
    which : 'final' (default), 'all', or a step keyword/integer.
        With ``'all'`` the ``initial`` step is also plotted.
    """
    if which not in ("final", "all"):
        raise ValueError(f"which must be 'final' or 'all' (got {which!r}).")

    plot_collocation_points(solver, save_dir=save_dir, formats=formats)
    plot_iteration_diagnostics(solver, result, save_dir=save_dir, formats=formats)

    if which == "all":
        for d in range(2):
            plot_solution_field(
                solver, result, component=d, mode="numerical", step="initial",
                resolution=resolution, save_dir=save_dir, formats=formats,
                filename=f"u{d+1}_numerical_contour_initial",
            )

    # Numerical 2-D and 3-D, always.
    for d in range(2):
        plot_solution_field(
            solver, result, component=d, mode="numerical",
            resolution=resolution, save_dir=save_dir, formats=formats,
        )
        plot_solution_3d(
            solver, result, component=d, mode="numerical",
            resolution=resolution_3d, save_dir=save_dir, formats=formats,
        )

    if solver.exact is not None:
        plot_error_history(solver, result, save_dir=save_dir, formats=formats)
        for d in range(2):
            plot_solution_field(
                solver, result, component=d, mode="exact",
                resolution=resolution, save_dir=save_dir, formats=formats,
            )
            plot_solution_3d(
                solver, result, component=d, mode="exact",
                resolution=resolution_3d, save_dir=save_dir, formats=formats,
            )
            plot_relative_error_field(
                solver, result, component=d,
                resolution=resolution, save_dir=save_dir, formats=formats,
            )