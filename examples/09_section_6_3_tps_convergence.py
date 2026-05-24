#!/usr/bin/env python
"""
Section 6.3 reproducer -- convergence with the parameter-free thin plate
spline kernel.

Reproduces Table 6 (relative L-infinity / L^2 errors and condition numbers
with phi(r) = r^4 ln r) and the second data series of Figure 7 of the paper.

The script solves the two-component advection-diffusion problem on the
flower domain with the same exponential density-dependent velocity, the
same diffusion (nu = 0.05), and the default manufactured solution, but
swaps the radial kernel from the radial basis function under tension to
the parameter-free thin plate spline. The script sweeps the number of
interior collocation points n = n' in {30, 50, 80, 120} and records the
relative L-infinity error, the relative L^2 error, and the spectral
condition number of the augmented collocation matrix.

Outputs
-------
* ``results/exp_convergence_tps.csv``  (one row per n)
* A pretty-printed table on stdout.

Usage
-----
    python examples/09_section_6_3_tps_convergence.py
    python examples/09_section_6_3_tps_convergence.py --n-grid 30 50 80
    python examples/09_section_6_3_tps_convergence.py --t-final 100 --dt 0.01
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List

import numpy as np

from rbf_nonlocal import NonlocalBalanceSolver

from scipy.spatial.distance import cdist
from rbf_nonlocal.basis import phi


def _polynomial_basis(points: np.ndarray, polydeg: int) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    cols = []
    for total in range(polydeg + 1):
        for i in range(total + 1):
            j = total - i
            cols.append(x ** i * y ** j)
    return np.column_stack(cols)


def augmented_matrix_condition_number(
    interior_points: np.ndarray,
    boundary_points: np.ndarray,
    polydeg: int,
    kind: str,
    tau: float,
) -> float:
    all_pts = np.vstack([interior_points, boundary_points])
    dists = cdist(all_pts, all_pts)
    K = phi(dists, kind=kind, tau=tau)
    Q = _polynomial_basis(all_pts, polydeg)
    dm = Q.shape[1]
    A_blk = np.block([[K,    Q                  ],
                      [Q.T,  np.zeros((dm, dm))]])
    return float(np.linalg.cond(A_blk))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_GRID  = (30, 50, 80, 120)
DEFAULT_T_FINAL = 100.0
DEFAULT_DT      = 0.01
DEFAULT_NU      = 0.05
DEFAULT_POLYDEG = 4


def run_one_tps(n: int, *, t_final: float, dt: float,
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
        # Thin plate spline (parameter-free) with the laplacian regulariser.
        # The ``rbf_kind`` override is what makes this combination possible
        # without changing the rest of the API; ``tau`` is then ignored.
        rbf_kind="thin_plate",
        tau=1.0,
    )
    t0 = time.perf_counter()
    result = solver.run()
    wall = time.perf_counter() - t0

    kappa = augmented_matrix_condition_number(
        solver.mats.interior_points,
        solver.mats.boundary_points,
        polydeg=polydeg,
        kind=solver.rbf_kind,
        tau=1.0,
    )
    return dict(
        n=n,
        E_inf=result.final_err_inf,
        E_2=result.final_err_l2,
        kappa=kappa,
        wall=wall,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Section 6.3 thin-plate-spline convergence on the "
                    "flower domain (reproduces Table 6 / Figure 7 TPS series).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-grid",  type=int,   nargs="+", default=list(DEFAULT_N_GRID))
    p.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL)
    p.add_argument("--dt",      type=float, default=DEFAULT_DT)
    p.add_argument("--nu",      type=float, default=DEFAULT_NU)
    p.add_argument("--polydeg", type=int,   default=DEFAULT_POLYDEG)
    p.add_argument("--out-csv", type=str,   default="results/exp_convergence_tps.csv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    print(f"\nSection 6.3 -- TPS convergence on flower domain")
    print(f"  kernel = phi(r) = r^4 ln r")
    print(f"  T = {args.t_final}, dt = {args.dt}, nu = {args.nu}, polydeg = {args.polydeg}")
    print(f"  n in {args.n_grid}\n")

    rows: List[dict] = []
    for n in args.n_grid:
        try:
            r = run_one_tps(
                n=n,
                t_final=args.t_final, dt=args.dt,
                nu_diag=args.nu, polydeg=args.polydeg,
            )
            rows.append(r)
            print(f"  n = {n:3d}  E_inf = {r['E_inf']:.3e}  "
                  f"E_2 = {r['E_2']:.3e}  kappa = {r['kappa']:.2e}  "
                  f"({r['wall']:.1f} s)")
        except Exception as exc:
            rows.append(dict(n=n, E_inf=np.nan, E_2=np.nan,
                             kappa=np.nan, wall=np.nan))
            print(f"  n = {n:3d}  FAILED: {exc}")

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["n", "E_inf", "E_2", "kappa", "wall"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
