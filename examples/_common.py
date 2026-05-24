"""
Shared CLI / runner / table-printer for the advection-diffusion examples.

Each numbered example script imports :func:`run_advdiff_example` from here
and supplies its own domain, velocity, and figure subdirectory; everything
else (CLI parsing, the convergence and nu-sweep loops, error-table
formatting, optional plotting) is implemented once here.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from rbf_nonlocal import NonlocalBalanceSolver


@dataclass
class RunConfig:
    """Discretisation knobs for one solve."""
    n_interior: int
    n_boundary: int
    nu_diag:    float
    polydeg:    int  = 4
    tau:        float = 10.0
    dt:         float = 0.05
    t_final:    float = 0.5


# Default sweep values. The paper uses {30, 50, 80} for convergence and
# {1e-2, 1e-3, 1e-4, 1e-5} for the nu sweep; a single default run uses
# n_interior = n_boundary = 80 and nu = 1e-2.
DEFAULT_RESOLUTIONS = (30, 50, 80)
DEFAULT_NU_VALUES   = (2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005)
DEFAULT_T_FINAL     = 5.0
DEFAULT_DT          = 0.01


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser(description: str) -> argparse.ArgumentParser:
    """Standard argument parser shared by every advection-diffusion script."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-interior", type=int, nargs="+", default=None,
        metavar="N",
        help="One or more interior-node counts; same value used for boundary "
             "nodes. If multiple, prints a convergence table.",
    )
    p.add_argument(
        "--nu-sweep", action="store_true",
        help=f"Sweep over nu in {DEFAULT_NU_VALUES}.",
    )
    p.add_argument(
        "--nu", type=float, default=1.0e-2,
        help="Diagonal entry of the diffusion matrix nu (used unless "
             "--nu-sweep is given).",
    )
    p.add_argument(
        "--dt", type=float, default=DEFAULT_DT,
        help="Time step.",
    )
    p.add_argument(
        "--t-final", type=float, default=DEFAULT_T_FINAL,
        help="Final time.",
    )
    p.add_argument(
        "--polydeg", type=int, default=4,
        help="Polynomial-tail degree.",
    )
    p.add_argument(
        "--tau", type=float, default=10.0,
        help="Tension parameter of the RBF.",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Save default figures to figures/<script-stem>/ after the headline run.",
    )
    p.add_argument(
        "--no-progress", action="store_true",
        help="Suppress per-step progress prints.",
    )
    return p


# ---------------------------------------------------------------------------
# Single solve
# ---------------------------------------------------------------------------

def run_one(
    cfg: RunConfig,
    *,
    domain: str,
    velocity: str,
    progress: bool = True,
):
    """Build a solver, run it, return ``(solver, result, wall_time)``."""
    nu = [[cfg.nu_diag, 0.0], [0.0, cfg.nu_diag]]

    solver = NonlocalBalanceSolver(
        domain=domain,
        velocity=velocity,
        regularization="laplacian",
        mode="validation",
        n_interior=cfg.n_interior,
        n_boundary=cfg.n_boundary,
        polydeg=cfg.polydeg,
        tau=cfg.tau,
        nu=nu,
        dt=cfg.dt,
        t_final=cfg.t_final,
        progress=progress,
    )

    t0 = time.perf_counter()
    result = solver.run()
    wall = time.perf_counter() - t0
    return solver, result, wall


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_resolution_table(rows: List[tuple]) -> None:
    """rows: [(n_interior, dt, err_inf, err_l2, wall_seconds), ...]"""
    print()
    print(f"{'n_interior':>10} {'dt':>8} "
          f"{'L_inf rel err':>14} {'L_2 rel err':>14} {'wall (s)':>10}")
    print("-" * 60)
    for n, dt, ei, el, wt in rows:
        print(f"{n:>10d} {dt:>8.4f} {ei:>14.3e} {el:>14.3e} {wt:>10.2f}")
    print()


def print_nu_sweep_table(rows: List[tuple]) -> None:
    """rows: [(nu, n_interior, err_inf, err_l2, wall_seconds), ...]"""
    print()
    print(f"{'nu':>10} {'n_interior':>10} "
          f"{'L_inf rel err':>14} {'L_2 rel err':>14} {'wall (s)':>10}")
    print("-" * 60)
    for nu, n, ei, el, wt in rows:
        print(f"{nu:>10.0e} {n:>10d} {ei:>14.3e} {el:>14.3e} {wt:>10.2f}")
    print()


# ---------------------------------------------------------------------------
# High-level entry point used by every script
# ---------------------------------------------------------------------------

def run_advdiff_example(
    *,
    domain: str,
    velocity: str,
    figure_subdir: str,
    description: str,
    argv: Sequence[str] | None = None,
) -> None:
    """
    Parse CLI, dispatch to the appropriate sweep, and (optionally) plot.

    Parameters
    ----------
    domain, velocity : strings forwarded to NonlocalBalanceSolver.
    figure_subdir : subdirectory name under ``figures/`` for output PNGs.
    description : displayed in --help.
    argv : tested by overriding sys.argv (default).
    """
    parser = build_argparser(description)
    args = parser.parse_args(argv)
    progress = not args.no_progress

    # ---- Sweep dispatch ----
    if args.nu_sweep:
        # nu sweep at one resolution (default 80).
        n = (args.n_interior or [80])[0]
        rows = []
        last_solver, last_result = None, None
        for nu_val in DEFAULT_NU_VALUES:
            cfg = RunConfig(
                n_interior=n, n_boundary=n, nu_diag=nu_val,
                polydeg=args.polydeg, tau=args.tau,
                dt=args.dt, t_final=args.t_final,
            )
            print(f"\n=== nu = {nu_val:.0e}  (n_interior = {n}) ===")
            s, r, wt = run_one(cfg, domain=domain, velocity=velocity, progress=progress)
            rows.append((nu_val, n, r.final_err_inf, r.final_err_l2, wt))
            last_solver, last_result = s, r
        print_nu_sweep_table(rows)

    elif args.n_interior is not None and len(args.n_interior) > 1:
        # Convergence table at fixed nu.
        rows = []
        last_solver, last_result = None, None
        for n in args.n_interior:
            cfg = RunConfig(
                n_interior=n, n_boundary=n, nu_diag=args.nu,
                polydeg=args.polydeg, tau=args.tau,
                dt=args.dt, t_final=args.t_final,
            )
            print(f"\n=== n_interior = {n}  (nu = {args.nu:.0e}) ===")
            s, r, wt = run_one(cfg, domain=domain, velocity=velocity, progress=progress)
            rows.append((n, args.dt, r.final_err_inf, r.final_err_l2, wt))
            last_solver, last_result = s, r
        print_resolution_table(rows)

    else:
        # Single default run.
        n = (args.n_interior or [80])[0]
        cfg = RunConfig(
            n_interior=n, n_boundary=n, nu_diag=args.nu,
            polydeg=args.polydeg, tau=args.tau,
            dt=args.dt, t_final=args.t_final,
        )
        print(f"\n=== Single run: n_interior = {n}, nu = {args.nu:.0e} ===")
        last_solver, last_result, wt = run_one(
            cfg, domain=domain, velocity=velocity, progress=progress,
        )
        print(f"\nFinal L_inf rel error: {last_result.final_err_inf:.3e}")
        print(f"Final L_2   rel error: {last_result.final_err_l2:.3e}")
        print(f"Wall time:             {wt:.2f} s")

    # ---- Optional figures ----
    if args.plot and last_solver is not None:
        save_dir = os.path.join("figures", figure_subdir)
        print(f"\nSaving figures to {save_dir}/")
        last_solver.plot_solution(last_result, save_dir=save_dir, which="final")