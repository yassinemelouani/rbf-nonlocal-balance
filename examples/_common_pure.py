"""
Shared CLI / runner / table-printer for the pure-convection (bi-Laplacian)
examples (05 and 06).

Mirrors the structure of ``_common.py`` for the advection-diffusion
scripts, but with the higher-order operator switched to the bi-Laplacian
and the diffusion matrix replaced by a scalar hyper-viscosity coefficient.
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
    """Discretisation knobs for one bi-Laplacian solve."""
    n_interior: int
    n_boundary: int
    epsilon:    float
    polydeg:    int  = 4
    dt:         float = 0.01
    t_final:    float = 5.0


# Table 5 of the paper sweeps epsilon over four orders of magnitude.
DEFAULT_RESOLUTIONS    = (30, 50, 80)
DEFAULT_EPSILON_VALUES = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
DEFAULT_T_FINAL        = 5.0
DEFAULT_DT             = 0.01


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser(description: str) -> argparse.ArgumentParser:
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
        "--epsilon-sweep", action="store_true",
        help=f"Sweep over epsilon in {DEFAULT_EPSILON_VALUES}.",
    )
    p.add_argument(
        "--epsilon", type=float, default=1.0e-5,
        help="Hyper-viscosity coefficient (used unless --epsilon-sweep is given).",
    )
    p.add_argument("--dt",      type=float, default=DEFAULT_DT)
    p.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL)
    p.add_argument("--polydeg", type=int,   default=4)
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
    solver = NonlocalBalanceSolver(
        domain=domain,
        velocity=velocity,
        regularization="bilaplacian",
        mode="validation",
        n_interior=cfg.n_interior,
        n_boundary=cfg.n_boundary,
        polydeg=cfg.polydeg,
        epsilon=cfg.epsilon,
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
    """rows: [(n, dt, eps, err_inf, err_l2, wall), ...]"""
    print()
    print(f"{'n_interior':>10} {'dt':>8} {'epsilon':>10} "
          f"{'L_inf rel err':>14} {'L_2 rel err':>14} {'wall (s)':>10}")
    print("-" * 70)
    for n, dt, eps, ei, el, wt in rows:
        print(f"{n:>10d} {dt:>8.4f} {eps:>10.0e} "
              f"{ei:>14.3e} {el:>14.3e} {wt:>10.2f}")
    print()


def print_epsilon_sweep_table(rows: List[tuple]) -> None:
    """rows: [(eps, n, err_inf, err_l2, wall), ...]"""
    print()
    print(f"{'epsilon':>10} {'n_interior':>10} "
          f"{'L_inf rel err':>14} {'L_2 rel err':>14} {'wall (s)':>10}")
    print("-" * 65)
    for eps, n, ei, el, wt in rows:
        print(f"{eps:>10.0e} {n:>10d} {ei:>14.3e} {el:>14.3e} {wt:>10.2f}")
    print()


# ---------------------------------------------------------------------------
# High-level entry point used by both scripts
# ---------------------------------------------------------------------------

def run_pure_convection_example(
    *,
    domain: str,
    velocity: str,
    figure_subdir: str,
    description: str,
    argv: Sequence[str] | None = None,
) -> None:
    parser = build_argparser(description)
    args = parser.parse_args(argv)
    progress = not args.no_progress

    if args.epsilon_sweep:
        n = (args.n_interior or [80])[0]
        rows = []
        last_solver, last_result = None, None
        for eps in DEFAULT_EPSILON_VALUES:
            cfg = RunConfig(
                n_interior=n, n_boundary=n, epsilon=eps,
                polydeg=args.polydeg, dt=args.dt, t_final=args.t_final,
            )
            print(f"\n=== epsilon = {eps:.0e}  (n_interior = {n}) ===")
            s, r, wt = run_one(cfg, domain=domain, velocity=velocity, progress=progress)
            rows.append((eps, n, r.final_err_inf, r.final_err_l2, wt))
            last_solver, last_result = s, r
        print_epsilon_sweep_table(rows)

    elif args.n_interior is not None and len(args.n_interior) > 1:
        rows = []
        last_solver, last_result = None, None
        for n in args.n_interior:
            cfg = RunConfig(
                n_interior=n, n_boundary=n, epsilon=args.epsilon,
                polydeg=args.polydeg, dt=args.dt, t_final=args.t_final,
            )
            print(f"\n=== n_interior = {n}  (epsilon = {args.epsilon:.0e}) ===")
            s, r, wt = run_one(cfg, domain=domain, velocity=velocity, progress=progress)
            rows.append((n, args.dt, args.epsilon, r.final_err_inf, r.final_err_l2, wt))
            last_solver, last_result = s, r
        print_resolution_table(rows)

    else:
        n = (args.n_interior or [80])[0]
        cfg = RunConfig(
            n_interior=n, n_boundary=n, epsilon=args.epsilon,
            polydeg=args.polydeg, dt=args.dt, t_final=args.t_final,
        )
        print(f"\n=== Single run: n_interior = {n}, epsilon = {args.epsilon:.0e} ===")
        last_solver, last_result, wt = run_one(
            cfg, domain=domain, velocity=velocity, progress=progress,
        )
        print(f"\nFinal L_inf rel error: {last_result.final_err_inf:.3e}")
        print(f"Final L_2   rel error: {last_result.final_err_l2:.3e}")
        print(f"Wall time:             {wt:.2f} s")

    if args.plot and last_solver is not None:
        save_dir = os.path.join("figures", figure_subdir)
        print(f"\nSaving figures to {save_dir}/")
        last_solver.plot_solution(last_result, save_dir=save_dir, which="final")