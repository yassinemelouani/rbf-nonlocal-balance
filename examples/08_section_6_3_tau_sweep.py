#!/usr/bin/env python
"""
Section 6.3 reproducer -- shape-parameter sweep with the radial basis
function under tension.

Reproduces Table 5 (relative L-infinity / L^2 errors and condition numbers
at the empirically optimal tau) and the data behind Figure 6 (the U-curve
panel) of the paper.

The script sweeps the number of interior collocation points
n = n' in {30, 50, 80, 120} and the shape parameter
tau in {1, 2, 3, 5, 8, 12, 20, 35, 60}. For each (n, tau) it solves the
two-component advection-diffusion problem on the flower domain with the
exponential density-dependent velocity, the default manufactured solution,
and records:

    * the relative L-infinity error at t = T,
    * the relative L^2 error at t = T,
    * the spectral condition number of the augmented collocation matrix.

Outputs
-------
* ``results/exp_convergence_sweep.csv``  (one row per (n, tau))
* A pretty-printed table on stdout with the empirically optimal tau*(n).

Usage
-----
    python examples/08_section_6_3_tau_sweep.py
    python examples/08_section_6_3_tau_sweep.py --n-grid 30 50 80
    python examples/08_section_6_3_tau_sweep.py --t-final 100 --dt 0.01
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List

import numpy as np
from scipy.spatial.distance import cdist

from rbf_nonlocal import NonlocalBalanceSolver
from rbf_nonlocal.basis import phi


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_GRID   = (30, 50, 80, 120)
DEFAULT_TAU_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0)
DEFAULT_T_FINAL  = 100.0
DEFAULT_DT       = 0.01
DEFAULT_NU       = 0.05      # diagonal entry of the nu matrix
DEFAULT_POLYDEG  = 4


# ---------------------------------------------------------------------------
# Augmented-matrix condition number
# ---------------------------------------------------------------------------
def _polynomial_basis(points: np.ndarray, polydeg: int) -> np.ndarray:
    """Same monomial basis as :func:`rbf_nonlocal.matrices._polynomial_basis`."""
    x, y = points[:, 0], points[:, 1]
    cols: List[np.ndarray] = []
    for total in range(polydeg + 1):
        for i in range(total + 1):
            j = total - i
            cols.append(x ** i * y ** j)
    return np.column_stack(cols)


def augmented_matrix_condition_number(
    interior_points: np.ndarray,
    boundary_points: np.ndarray,
    tau: float,
    polydeg: int,
    kind: str = "tension",
) -> float:
    """spectral condition number of [[K, Q], [Q^T, 0]]."""
    all_pts = np.vstack([interior_points, boundary_points])
    dists = cdist(all_pts, all_pts)
    K = phi(dists, kind=kind, tau=tau)
    Q = _polynomial_basis(all_pts, polydeg)
    dm = Q.shape[1]
    A_blk = np.block([[K,    Q                  ],
                      [Q.T,  np.zeros((dm, dm))]])
    return float(np.linalg.cond(A_blk))


# ---------------------------------------------------------------------------
# Single (n, tau) run
# ---------------------------------------------------------------------------
def run_one(n: int, tau: float, *, t_final: float, dt: float,
            nu_diag: float, polydeg: int) -> dict:
    nu = np.array([[nu_diag, 0.0], [0.0, nu_diag]], dtype=float)
    solver = NonlocalBalanceSolver(
        domain="flower",
        velocity="exp_density",
        regularization="laplacian",
        mode="validation",
        n_interior=n,
        n_boundary=n,
        nu=nu,
        dt=dt,
        t_final=t_final,
        polydeg=polydeg,
        tau=tau,
    )
    t0 = time.perf_counter()
    result = solver.run()
    wall = time.perf_counter() - t0

    kappa = augmented_matrix_condition_number(
        solver.mats.interior_points,
        solver.mats.boundary_points,
        tau=tau,
        polydeg=polydeg,
        kind=solver.rbf_kind,
    )
    return dict(
        n=n,
        tau=tau,
        E_inf=result.final_err_inf,
        E_2=result.final_err_l2,
        kappa=kappa,
        wall=wall,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Section 6.3 tau sweep on the flower domain "
                    "(reproduces Table 5 / Figure 6 panel a).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-grid",   type=int,   nargs="+", default=list(DEFAULT_N_GRID))
    p.add_argument("--tau-grid", type=float, nargs="+", default=list(DEFAULT_TAU_GRID))
    p.add_argument("--t-final",  type=float, default=DEFAULT_T_FINAL)
    p.add_argument("--dt",       type=float, default=DEFAULT_DT)
    p.add_argument("--nu",       type=float, default=DEFAULT_NU)
    p.add_argument("--polydeg",  type=int,   default=DEFAULT_POLYDEG)
    p.add_argument("--out-csv",  type=str,   default="results/exp_convergence_sweep.csv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rows: List[dict] = []
    print(f"\nSection 6.3 -- tau sweep on flower domain")
    print(f"  T = {args.t_final}, dt = {args.dt}, nu = {args.nu}, polydeg = {args.polydeg}")
    print(f"  n in {args.n_grid}")
    print(f"  tau in {args.tau_grid}\n")

    for n in args.n_grid:
        print(f"  --- n = {n} ---")
        for tau in args.tau_grid:
            try:
                r = run_one(
                    n=n, tau=tau,
                    t_final=args.t_final, dt=args.dt,
                    nu_diag=args.nu, polydeg=args.polydeg,
                )
                rows.append(r)
                print(f"    tau = {tau:6.2f}  E_inf = {r['E_inf']:.3e}  "
                      f"E_2 = {r['E_2']:.3e}  kappa = {r['kappa']:.2e}  "
                      f"({r['wall']:.1f} s)")
            except Exception as exc:
                rows.append(dict(n=n, tau=tau, E_inf=np.nan, E_2=np.nan,
                                 kappa=np.nan, wall=np.nan))
                print(f"    tau = {tau:6.2f}  FAILED: {exc}")

    # ----- write CSV -----
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["n", "tau", "E_inf", "E_2",
                                                "kappa", "wall"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {args.out_csv}")

    # ----- empirical tau*(n) -----
    print("\nEmpirical optima:")
    print(f"  {'n':>4}  {'tau*':>5}  {'E_inf*':>10}  {'E_2*':>10}  {'kappa*':>10}")
    for n in args.n_grid:
        sub = [r for r in rows if r["n"] == n and np.isfinite(r["E_inf"])]
        if not sub:
            continue
        best = min(sub, key=lambda r: r["E_inf"])
        print(f"  {n:>4}  {best['tau']:>5.0f}  "
              f"{best['E_inf']:.3e}  {best['E_2']:.3e}  {best['kappa']:.2e}")


if __name__ == "__main__":
    main()
