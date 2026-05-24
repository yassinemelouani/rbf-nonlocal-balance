"""
Nonlinear reaction term N(u) used in the paper.

Currently only one reaction is provided, :class:`DefaultReaction`, which
matches the manufactured-solution experiments in Section 6 of the paper:

.. math::
    \\mathcal{N}_1(u) &= 0.15\\,\\tanh u_1 - 0.05\\,\\tanh(u_2^2),\\\\
    \\mathcal{N}_2(u) &= 0.10\\,\\tanh u_2 + 0.05\\, u_1\\, e^{-u_1^2}.

The class also exposes :meth:`jacobian` returning the analytical
:math:`J_{\\mathcal{N}}(u)`, which is used by the Frechet-derivative-based
Newton-Krylov solver. New reactions can be added by subclassing
:class:`Reaction` and registering them in :data:`_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Reaction",
    "DefaultReaction",
    "make_reaction",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Reaction:
    """
    Abstract base class for a nonlinear reaction term
    :math:`\\mathcal{N} : \\mathbb{R}^p \\to \\mathbb{R}^p`.

    Concrete subclasses implement:

    * :meth:`field(u)` — value :math:`\\mathcal{N}(u)`, shape ``u.shape``.
    * :meth:`jacobian(u)` — :math:`(J_{\\mathcal{N}})_{ij} =
      \\partial \\mathcal{N}_i / \\partial u_j`, shape ``u.shape + (p,)``.

    Both methods must be vectorised: ``u`` may be ``shape == (p,)`` or
    ``shape == (..., p)``.
    """

    name: str = "abstract"

    def field(self, u):  # pragma: no cover
        raise NotImplementedError

    def jacobian(self, u):  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The reaction used throughout the paper
# ---------------------------------------------------------------------------

@dataclass
class DefaultReaction(Reaction):
    r"""
    Two-component reaction with tanh saturation and a Gaussian-modulated
    cross-coupling term.

    .. math::
        \mathcal{N}_1(u) &= 0.15\,\tanh u_1 - 0.05\,\tanh(u_2^2),\\
        \mathcal{N}_2(u) &= 0.10\,\tanh u_2 + 0.05\, u_1\, e^{-u_1^2}.

    Bounded, smooth, and Lipschitz on every compact set; remains so on
    the manufactured solution's image :math:`u_1 \in [0, 2]`,
    :math:`u_2 \in [0.5, 1.5]`.
    """

    name: str = "default"

    def field(self, u):
        u = np.asarray(u, dtype=float)
        N = np.empty(u.shape)
        N[..., 0] = 0.15 * np.tanh(u[..., 0]) - 0.05 * np.tanh(u[..., 1] ** 2)
        N[..., 1] = (0.10 * np.tanh(u[..., 1])
                     + 0.05 * u[..., 0] * np.exp(-u[..., 0] ** 2))
        return N

    def jacobian(self, u):
        u = np.asarray(u, dtype=float)
        # Component-wise temporaries
        sech2_u0    = 1.0 - np.tanh(u[..., 0]) ** 2          # sech^2(u_1)
        sech2_u1    = 1.0 - np.tanh(u[..., 1]) ** 2          # sech^2(u_2)
        sech2_u1sq  = 1.0 - np.tanh(u[..., 1] ** 2) ** 2     # sech^2(u_2^2)
        exp_neg_u0sq = np.exp(-u[..., 0] ** 2)

        J = np.empty(u.shape + (2,))
        # dN1/du1 = 0.15 sech^2(u1)
        J[..., 0, 0] =  0.15 * sech2_u0
        # dN1/du2 = -0.05 * 2 u2 * sech^2(u2^2)
        J[..., 0, 1] = -0.10 * u[..., 1] * sech2_u1sq
        # dN2/du1 = 0.05 * exp(-u1^2) * (1 - 2 u1^2)
        J[..., 1, 0] =  0.05 * exp_neg_u0sq * (1.0 - 2.0 * u[..., 0] ** 2)
        # dN2/du2 = 0.10 sech^2(u2)
        J[..., 1, 1] =  0.10 * sech2_u1

        return J


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "default": DefaultReaction,
}


def make_reaction(name: str = "default", **kwargs) -> Reaction:
    """
    Construct a reaction term by name. Currently only ``"default"`` is
    registered; pass ``**kwargs`` if a future variant accepts parameters.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown reaction {name!r}. Available: {tuple(_REGISTRY)}."
        )
    return _REGISTRY[key](**kwargs)