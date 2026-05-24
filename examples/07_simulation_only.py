#!/usr/bin/env python
"""
Simulation-mode demo (no analytical solution).

Solves the advection-diffusion system from a Gaussian-bump initial
condition with zero Dirichlet boundary data and no forcing. There is no
manufactured solution, so no error is reported; the script just produces
contour and 3-D plots of the numerical solution.

This is the script to copy as a starting point for your own problem:
edit ``my_initial_condition``, ``my_boundary_condition``, optionally add
``my_source_term``, and adjust the solver keyword arguments.

Usage
-----
    python examples/07_simulation_only.py
    python examples/07_simulation_only.py --domain gear --plot
    python examples/07_simulation_only.py --plot --t-final 3.0
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from rbf_nonlocal import NonlocalBalanceSolver


# ---------------------------------------------------------------------------
# User-supplied problem data
# ---------------------------------------------------------------------------

def my_initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Two Gaussian bumps centred near the domain centre; first component
    peaked, second component flat.

    Parameters
    ----------
    x, y : np.ndarray, shape ``(n,)``
        Interior collocation coordinates.

    Returns
    -------
    np.ndarray, shape ``(n, 2)``
    """
    cx, cy = 0.5, 0.5
    bump = np.exp(-30.0 * ((x - cx) ** 2 + (y - cy) ** 2))
    out = np.empty((x.size, 2), dtype=float)
    out[:, 0] = bump
    out[:, 1] = 0.5 * bump
    return out


def my_boundary_condition(t: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Homogeneous Dirichlet data at all times.

    Parameters
    ----------
    t : float
    x, y : np.ndarray, shape ``(n_prime,)``
        Boundary collocation coordinates.

    Returns
    -------
    np.ndarray, shape ``(n_prime, 2)``
    """
    return np.zeros((x.size, 2), dtype=float)


# Set this to a callable to add a forcing term, or leave as ``None`` for f = 0.
my_source_term = None
# Example of a non-trivial source:
#
# def my_source_term(t, x, y):
#     out = np.zeros((x.size, 2), dtype=float)
#     out[:, 0] = 0.05 * np.cos(np.pi * x) * np.exp(-t)
#     return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulation-mode demo: solve from a user-supplied "
                    "initial condition without comparing to an analytical "
                    "solution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--domain", choices=("flower", "gear"), default="flower",
    )
    p.add_argument(
        "--velocity", choices=("exp_density",),
        default="exp_density",
    )
    p.add_argument(
        "--regularization", choices=("laplacian", "bilaplacian"),
        default="laplacian",
    )
    p.add_argument("--n-interior", type=int, default=80)
    p.add_argument("--n-boundary", type=int, default=80)
    p.add_argument("--polydeg", type=int, default=4)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument(
        "--nu", type=float, default=0.01,
        help="Diagonal entry of nu (used when --regularization=laplacian).",
    )
    p.add_argument(
        "--epsilon", type=float, default=1.0e-5,
        help="Hyper-viscosity (used when --regularization=bilaplacian).",
    )
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--t-final", type=float, default=2.0)
    p.add_argument("--plot", action="store_true",
                   help="Save figures to figures/07_simulation_only/.")
    p.add_argument("--no-progress", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_argparser().parse_args()

    # Build keyword arguments common to both regimes.
    common_kwargs = dict(
        domain=args.domain,
        velocity=args.velocity,
        regularization=args.regularization,
        mode="simulation",
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        polydeg=args.polydeg,
        tau=args.tau,
        dt=args.dt,
        t_final=args.t_final,
        initial_condition=my_initial_condition,
        boundary_condition=my_boundary_condition,
        source_term=my_source_term,
        progress=not args.no_progress,
    )

    if args.regularization == "laplacian":
        solver = NonlocalBalanceSolver(
            **common_kwargs,
            nu=[[args.nu, 0.0], [0.0, args.nu]],
        )
        regulariser_label = f"nu = {args.nu:.0e}"
    else:
        solver = NonlocalBalanceSolver(
            **common_kwargs,
            epsilon=args.epsilon,
        )
        regulariser_label = f"epsilon = {args.epsilon:.0e}"

    print("\n=== Simulation-mode run ===")
    print(f"  domain={args.domain}, velocity={args.velocity}")
    print(f"  regularization={args.regularization} ({regulariser_label})")
    print(f"  n_interior={args.n_interior}, n_boundary={args.n_boundary}")
    print(f"  dt={args.dt}, t_final={args.t_final}")

    t0 = time.perf_counter()
    result = solver.run()
    wall = time.perf_counter() - t0
    print(f"\nWall time: {wall:.2f} s ({result.n_steps} BDF steps)")

    if args.plot:
        save_dir = os.path.join("figures", "07_simulation_only")
        print(f"Saving figures to {save_dir}/")
        solver.plot_solution(result, save_dir=save_dir, which="final")


if __name__ == "__main__":
    main()