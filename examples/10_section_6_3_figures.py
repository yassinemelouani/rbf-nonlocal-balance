#!/usr/bin/env python
"""
Section 6.3 figure generator -- reads the two CSV files produced by
scripts 08 and 09 and writes Figure 6 (tau-sweep U-curves) and Figure 7
(kernel comparison + accuracy ceiling) as PDF and PNG.

Reads
-----
* ``results/exp_convergence_sweep.csv``  (from ``08_section_6_3_tau_sweep.py``)
* ``results/exp_convergence_tps.csv``    (from ``09_section_6_3_tps_convergence.py``)

Writes
------
* ``figures/section_6_3/tau_sweep.{pdf,png}``        -- Figure 6
* ``figures/section_6_3/convergence.{pdf,png}``      -- Figure 7

Usage
-----
    python examples/10_section_6_3_figures.py
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter


# ---------------------------------------------------------------------------
# Publication-quality matplotlib settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":     "cm",
    "font.size":            11,
    "axes.labelsize":       12,
    "axes.titlesize":       12,
    "axes.linewidth":       0.8,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.major.size":     4.0,
    "ytick.major.size":     4.0,
    "xtick.minor.size":     2.0,
    "ytick.minor.size":     2.0,
    "xtick.top":            True,
    "ytick.right":          True,
    "legend.fontsize":      10,
    "legend.frameon":       True,
    "legend.framealpha":    0.95,
    "legend.edgecolor":     "0.4",
    "legend.fancybox":      False,
    "legend.borderpad":     0.4,
    "figure.dpi":           120,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.04,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

# Color-blind-safe palette (Wong 2011).
N_COLORS  = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
N_MARKERS = ["o", "s", "^", "D"]


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------
def _load_csv(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing CSV: {path}. Run the matching sweep script first."
        )
    with open(path, "r", newline="") as fh:
        return [
            {k: float(v) if v not in ("", "nan") else float("nan")
             for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def _scientific_log_label(val, _):
    if val <= 0:
        return ""
    exp = int(round(np.log10(val)))
    if abs(val - 10.0 ** exp) / val > 1e-6:
        return ""
    return r"$10^{{{0}}}$".format(exp)


def _style_log_axes(ax):
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(_scientific_log_label))
    ax.yaxis.set_major_formatter(FuncFormatter(_scientific_log_label))
    ax.grid(True, which="major", alpha=0.30, linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.4)


# ---------------------------------------------------------------------------
# Figure 6 -- tau sweep, one curve per n
# ---------------------------------------------------------------------------
def figure_tau_sweep(rows: list, out_pdf: str, out_png: str) -> None:
    grouped: dict = defaultdict(list)
    for r in rows:
        grouped[int(r["n"])].append(r)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for color, marker, (n, items) in zip(N_COLORS, N_MARKERS,
                                         sorted(grouped.items())):
        items.sort(key=lambda r: r["tau"])
        taus  = [r["tau"]  for r in items if np.isfinite(r["E_inf"])]
        e_inf = [r["E_inf"] for r in items if np.isfinite(r["E_inf"])]
        ax.loglog(taus, e_inf, marker=marker, color=color, markersize=6.0,
                  linewidth=1.4, markeredgecolor="white", markeredgewidth=0.5,
                  label=fr"$n={n}$")
    ax.set_xlabel(r"shape parameter $\tau$")
    ax.set_ylabel(r"relative $L^{\infty}$ error")
    ax.set_xlim(0.8, 80)
    ax.set_ylim(1e-4, 1e-2)
    _style_log_axes(ax)
    ax.legend(loc="upper left", ncol=2)
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close(fig)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
# Figure 7 -- ceiling comparison: tension(best tau) vs TPS
# ---------------------------------------------------------------------------
def figure_kernel_comparison(rows_tension: list, rows_tps: list,
                             out_pdf: str, out_png: str) -> None:
    # Best tau* per n in the tension sweep.
    grouped: dict = defaultdict(list)
    for r in rows_tension:
        grouped[int(r["n"])].append(r)

    n_tension, best_tension_inf = [], []
    for n in sorted(grouped):
        items = [r for r in grouped[n] if np.isfinite(r["E_inf"])]
        if not items:
            continue
        best = min(items, key=lambda r: r["E_inf"])
        n_tension.append(n)
        best_tension_inf.append(best["E_inf"])

    rows_tps_sorted = sorted(rows_tps, key=lambda r: r["n"])
    n_tps   = [int(r["n"])    for r in rows_tps_sorted if np.isfinite(r["E_inf"])]
    e_inf_tps = [r["E_inf"]   for r in rows_tps_sorted if np.isfinite(r["E_inf"])]

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.loglog(n_tension, best_tension_inf, marker="o", color=N_COLORS[0],
              markersize=8, linewidth=1.6, markeredgecolor="white",
              markeredgewidth=0.6, label=r"tension, optimal $\tau^{*}(n)$")
    ax.loglog(n_tps, e_inf_tps, marker="s", color=N_COLORS[2], markersize=8,
              linewidth=1.6, markeredgecolor="white", markeredgewidth=0.6,
              label=r"thin plate spline")
    ax.axhline(2e-4, color="0.35", linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(140, 1.45e-4,
            r"accuracy ceiling $\sim 2\!\times\!10^{-4}$",
            ha="right", va="center", fontsize=9, color="0.25")
    ax.set_xlabel(r"number of interior points $n$")
    ax.set_ylabel(r"relative $L^{\infty}$ error")
    ax.set_xlim(25, 145)
    ax.set_ylim(1e-4, 1e-2)
    _style_log_axes(ax)
    ax.legend(loc="upper right")
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close(fig)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate Figures 6 and 7 of Section 6.3 from the "
                    "two CSV files produced by scripts 08 and 09.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tau-csv", type=str,
                   default="results/exp_convergence_sweep.csv")
    p.add_argument("--tps-csv", type=str,
                   default="results/exp_convergence_tps.csv")
    p.add_argument("--out-dir", type=str,
                   default="figures/section_6_3")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Reading sweep results...")
    rows_tension = _load_csv(args.tau_csv)
    rows_tps     = _load_csv(args.tps_csv)
    print(f"  {len(rows_tension)} rows from {args.tau_csv}")
    print(f"  {len(rows_tps)} rows from {args.tps_csv}")

    print("\nFigure 6 (tau sweep):")
    figure_tau_sweep(
        rows_tension,
        out_pdf=os.path.join(args.out_dir, "tau_sweep.pdf"),
        out_png=os.path.join(args.out_dir, "tau_sweep.png"),
    )

    print("\nFigure 7 (kernel comparison):")
    figure_kernel_comparison(
        rows_tension, rows_tps,
        out_pdf=os.path.join(args.out_dir, "convergence.pdf"),
        out_png=os.path.join(args.out_dir, "convergence.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
