#!/usr/bin/env python
"""
Experiment 1: advection-diffusion on the flower domain.

Reproduces Table 3 of the paper (with ``--n-interior 30 50 80``) and
Table 7 (with ``--nu-sweep``).

Usage
-----
    python examples/01_advdiff_flower.py
    python examples/01_advdiff_flower.py --n-interior 30 50 80
    python examples/01_advdiff_flower.py --nu-sweep
    python examples/01_advdiff_flower.py --plot
"""
from _common import run_advdiff_example

if __name__ == "__main__":
    run_advdiff_example(
        domain="flower",
        velocity="exp_density",
        figure_subdir="01_advdiff_flower",
        description="Advection-diffusion on the flower domain.",
    )
