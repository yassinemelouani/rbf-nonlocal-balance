# rbf-nonlocal-balance

A meshless radial basis function (RBF) + BDF solver for nonlocal balance equations on irregular two-dimensional domains.

This package implements the numerical scheme described in:

> Y. Melouani, A. Bouhamidi, I. El Harraki,
> *A Meshless Radial Basis Function Method for Nonlocal Balance Equations*, 2026.

---

## Overview

The package solves the two-component system

$$\partial_t u + \nabla \cdot (V(\mathcal{W}_u) \otimes u) + \mathcal{L} u = \mathcal{N}(u) + f$$

on irregular 2-D domains, where

- $V$ is an exponential density-dependent velocity with cross-inhibition,
- $\mathcal{W}_u$ is a nonlocal weighted average of $u$,
- $\mathcal{L}$ is either $-\nu\,\Delta$ (advection-diffusion) or $\varepsilon\,\Delta^2$ (pure nonlocal convection with hyper-viscosity),
- $\mathcal{N}$ is a quadratic reaction term.

**Space discretisation:** RBF + polynomial collocation (tension RBF or thin-plate spline), with Gaussian nonlocal kernels integrated by 2-D Gauss-Legendre quadrature.

**Time discretisation:** Variable-order BDF (orders 1–6), bootstrapped implicitly by using BDF-$k$ at step $k$.

**Linear algebra:** Newton–Krylov with GMRES at each step; optional ILU preconditioning for small meshes.

**Domains:** flower ($r(\theta)=a+b\cos(n_\text{petals}\,\theta)$) and gear (cosine-profile teeth), with quasi-random Halton collocation points.

---

## Installation

```bash
git clone https://github.com/yassinemelouani/rbf-nonlocal-balance.git
cd rbf-nonlocal-balance
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
pip install -e .
```

Dependencies: `numpy>=1.22`, `scipy>=1.10`, `matplotlib>=3.5`, `joblib>=1.2`.

---

## Quick start

```python
from rbf_nonlocal import NonlocalBalanceSolver

solver = NonlocalBalanceSolver(
    domain="flower",
    velocity="exp_density",
    regularization="laplacian",
    mode="validation",          # compare to manufactured solution
    n_interior=80,
    n_boundary=80,
    nu=[[0.01, 0.0], [0.0, 0.01]],
    dt=0.01,
    t_final=5.0,
    polydeg=4,
    tau=10.0,
)
result = solver.run()
print(f"Final L∞ relative error: {result.final_err_inf:.3e}")
print(f"Final L²  relative error: {result.final_err_l2:.3e}")

# Save publication-quality figures
solver.plot_solution(result, save_dir="figures/run1")
```

### Simulation mode (user-supplied problem)

```python
import numpy as np
from rbf_nonlocal import NonlocalBalanceSolver

def my_ic(x, y):
    bump = np.exp(-30.0 * ((x - 0.5)**2 + (y - 0.5)**2))
    return np.column_stack([bump, 0.5 * bump])   # shape (n, 2)

def my_bc(t, x, y):
    return np.zeros((x.size, 2))                 # homogeneous Dirichlet

solver = NonlocalBalanceSolver(
    domain="flower",
    velocity="exp_density",
    regularization="laplacian",
    mode="simulation",          # no manufactured solution needed
    n_interior=80,
    n_boundary=80,
    nu=[[0.01, 0.0], [0.0, 0.01]],
    dt=0.01,
    t_final=2.0,
    initial_condition=my_ic,
    boundary_condition=my_bc,
)
result = solver.run()
solver.plot_solution(result, save_dir="figures/simulation")
```

---

## Configuration flags

| Flag | Values | Effect |
|------|--------|--------|
| `domain` | `"flower"` / `"gear"` | geometry and collocation-point generator |
| `velocity` | `"exp_density"` | density-dependent velocity $V$, its Jacobian and Hessian |
| `regularization` | `"laplacian"` / `"bilaplacian"` | $-\nu\Delta$ (advection-diffusion) vs $\varepsilon\Delta^2$ (hyper-viscosity) |
| `rbf_kind` (optional) | `"tension"` / `"thin_plate"` | override the RBF kernel; by default `"tension"` is paired with the Laplacian and `"thin_plate"` with the bi-Laplacian |
| `mode` | `"validation"` / `"simulation"` | compare to manufactured solution vs run user-supplied problem |

---

## Example scripts

Eight runnable scripts under `examples/`:

| Script | What it does |
|--------|--------------|
| `01_advdiff_flower.py` | Flower advection-diffusion, validation. Default single run; `--n-interior 30 50 80` reproduces Table 3; `--nu-sweep` reproduces Table 7. |
| `03_advdiff_gear.py` | Gear advection-diffusion, validation. `--n-interior 30 50 80` reproduces Table 4. |
| `05_pure_convection_flower.py` | Flower, hyper-viscosity ($\varepsilon\Delta^2$). `--eps-sweep` reproduces Table 8. |
| `06_pure_convection_gear.py` | Gear, hyper-viscosity. |
| `07_simulation_only.py` | User-supplied IC/BC, no manufactured solution. |
| `08_section_6_3_tau_sweep.py` | Section 6.3 $\tau$ sweep with the tension kernel, produces `results/exp_convergence_sweep.csv` (Table 5). |
| `09_section_6_3_tps_convergence.py` | Section 6.3 convergence with the parameter-free thin plate spline, produces `results/exp_convergence_tps.csv` (Table 6). |
| `10_section_6_3_figures.py` | Reads the two CSVs above and writes Figures 6 and 7 (U-curves and the kernel-comparison ceiling plot). |

All advection-diffusion validation scripts share a common CLI:

```bash
# Default single run at n=80
python examples/01_advdiff_flower.py

# Convergence table (reproduces paper Table 3 for the flower / Table 4 for the gear)
python examples/01_advdiff_flower.py --n-interior 30 50 80

# Nu sweep (reproduces paper Table 7)
python examples/01_advdiff_flower.py --nu-sweep

# Save figures
python examples/01_advdiff_flower.py --plot
```

The simulation script has an extended CLI:

```bash
python examples/07_simulation_only.py --domain gear --plot
```

---

## Reproducing every result in the paper

Each row below maps one paper artefact to the exact command(s) that produce it.
All commands are run from the repository root, with the package installed
in editable mode (`pip install -e .`).

| Paper artefact | Command |
|----------------|---------|
| **Table 3** -- flower convergence at $\tau=10$, $T=500$ | `python examples/01_advdiff_flower.py --n-interior 30 50 80 --t-final 500 --dt 0.01` |
| **Table 4** -- gear convergence at $\tau=10$, $T=500$ | `python examples/03_advdiff_gear.py  --n-interior 30 50 80 --t-final 500 --dt 0.01` |
| **Table 5** -- $\tau$ sweep (Section 6.3) | `python examples/08_section_6_3_tau_sweep.py` |
| **Table 6** -- thin-plate-spline convergence (Section 6.3) | `python examples/09_section_6_3_tps_convergence.py` |
| **Table 7** -- $\nu$ sweep on flower, $n=80$ | `python examples/01_advdiff_flower.py --nu-sweep --t-final 500 --dt 0.01` |
| **Table 8** -- hyper-viscosity $\varepsilon$ sweep | `python examples/05_pure_convection_flower.py --eps-sweep --t-final 500 --dt 0.01` |
| **Figure 1** -- collocation point distributions | `python examples/01_advdiff_flower.py --plot` and `python examples/03_advdiff_gear.py --plot` |
| **Figures 2--4** -- solutions / error contours on flower & gear | `python examples/01_advdiff_flower.py --plot --t-final 500 --dt 0.01` and `python examples/03_advdiff_gear.py --plot --t-final 500 --dt 0.01` |
| **Figure 5** -- $E_\infty$ vs $\nu$ | `python examples/01_advdiff_flower.py --nu-sweep --plot --t-final 500 --dt 0.01` |
| **Figure 6** -- U-curves ($E_\infty$ vs $\tau$) | `python examples/10_section_6_3_figures.py` (after running 08) |
| **Figure 7** -- kernel comparison ceiling plot | `python examples/10_section_6_3_figures.py` (after running 08 and 09) |
| **Figures 8--10** -- hyper-viscosity contour and 3D plots | `python examples/05_pure_convection_flower.py --plot --t-final 500 --dt 0.01` |

The short-time defaults in the scripts (typically $T=5$ or $T=20$ with $\Delta t$ around $0.025$) are chosen so a single run takes one to a few minutes; pass `--t-final 500 --dt 0.01` to reproduce the long-time tables of the paper exactly. The Section 6.3 sweep scripts run at the short-time defaults because the spatial error is what is being measured there and the temporal error of BDF-6 is negligible (see Section 6.3 of the paper).

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

The test suite (~25 tests) covers per-module unit checks (FD verification of every analytical derivative) and four end-to-end smoke runs. Total runtime is under one minute.

---

## Package structure

```
rbf_nonlocal/
├── basis.py            RBF kernels (tension, thin-plate) and derivatives
├── domains.py          FlowerDomain, GearDomain, Halton point generation
├── matrices.py         RBF collocation matrices, Laplacian/bi-Laplacian assembly
├── nonlocal_ops.py     Nonlocal operator matrices Ka, Kb and spatial gradients
├── velocity.py         VelocityExpDensity
├── reaction.py         DefaultReaction
├── manufactured.py     DefaultManufacturedSolution and source-term precomputation
├── rhs.py              Semi-discrete RHS and Fréchet derivative
├── time_integration.py BDF1–6 + Newton–Krylov with GMRES
├── solver.py           NonlocalBalanceSolver (the front-door API)
└── plotting.py         Publication-quality figures (grid-based RBF interpolation)
```

---

## Citation

If you use this software, please cite the accompanying paper:

```bibtex
@article{melouani2025meshless,
  author  = {Melouani, Yassine and Bouhamidi, Abderrahman and {El Harraki}, Imad},
  title   = {A Meshless Radial Basis Function Method for Nonlocal Balance Equations},
  year    = {2026},
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
