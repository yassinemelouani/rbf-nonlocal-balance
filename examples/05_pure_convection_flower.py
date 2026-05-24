#!/usr/bin/env python
"""
Experiment 2: pure nonlocal convection with bi-Laplacian hyper-viscosity
on the flower domain.

Reproduces (with --n-interior 30 50 80) the convergence rows of Table 5
on the flower geometry, and (with --epsilon-sweep) the epsilon sweep at
fixed resolution.

Usage
-----
    python examples/05_pure_convection_flower.py
    python examples/05_pure_convection_flower.py --n-interior 30 50 80
    python examples/05_pure_convection_flower.py --epsilon-sweep
    python examples/05_pure_convection_flower.py --plot
"""
from _common_pure import run_pure_convection_example

if __name__ == "__main__":
    run_pure_convection_example(
        domain="flower",
        velocity="exp_density",
        figure_subdir="05_pure_convection_flower",
        description="Pure nonlocal convection with bi-Laplacian "
                    "hyper-viscosity on the flower domain.",
    )