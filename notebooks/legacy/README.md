# Legacy notebooks

The three Jupyter notebooks in this folder are the **original, self-contained
implementations** that produced the numerical results reported in the paper
*A Meshless Radial Basis Function Method for Nonlocal Balance Equations*:

| Notebook | What it produced |
|----------|------------------|
| `Visc_flower.ipynb`      | Original setup for Tables 3 and 7 (flower domain, $-\nu\Delta$). |
| `Visc_gear.ipynb`        | Original setup for Table 4 (gear domain, $-\nu\Delta$). |
| `Hypervisc_flower.ipynb` | Original setup for Table 8 (flower domain, $\varepsilon\Delta^2$). |

They are kept here for historical reproducibility. Each notebook redefines
its own copies of the RBF basis, Halton point generator, matrix assembly,
Newton–Krylov solver, and plotting routines — these are no longer maintained,
and minor differences from the current `rbf_nonlocal` package (in particular
the off-diagonal entries of the velocity Hessian) are not propagated here.

**For new work, use the cleaned `rbf_nonlocal` package and the scripts under
`../examples/` instead.** The runnable example scripts reproduce every
table and figure of the paper; see the *Reproducing every result in the
paper* table in the top-level [README](../../README.md).
