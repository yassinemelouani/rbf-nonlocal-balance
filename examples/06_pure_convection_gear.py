#!/usr/bin/env python
"""
Experiment 2 (gear variant): pure nonlocal convection with bi-Laplacian
hyper-viscosity on the gear domain.

The original notebook only had the flower geometry for this regime; this
script adds the gear case requested for the public release. Same CLI as
05_*; pass --help for options.
"""
from _common_pure import run_pure_convection_example

if __name__ == "__main__":
    run_pure_convection_example(
        domain="gear",
        velocity="exp_density",
        figure_subdir="06_pure_convection_gear",
        description="Pure nonlocal convection with bi-Laplacian "
                    "hyper-viscosity on the gear domain.",
    )