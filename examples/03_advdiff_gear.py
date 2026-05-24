#!/usr/bin/env python
"""
Experiment 3: advection-diffusion on the gear domain.

Reproduces Table 4 of the paper (with ``--n-interior 30 50 80``).

Usage
-----
    python examples/03_advdiff_gear.py
    python examples/03_advdiff_gear.py --n-interior 30 50 80
    python examples/03_advdiff_gear.py --plot
"""
from _common import run_advdiff_example

if __name__ == "__main__":
    run_advdiff_example(
        domain="gear",
        velocity="exp_density",
        figure_subdir="03_advdiff_gear",
        description="Advection-diffusion on the gear domain.",
    )
