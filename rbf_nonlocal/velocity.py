"""
Velocity field V(W_u) used in the paper.

* :class:`VelocityExpDensity` — the exponential density-dependent velocity
  with cross-inhibition,

  .. math::
      V_1(\\mathcal{W}_u) &= 0.5\\, e^{-1.5\\,\\mathcal{W}_{u,1}}
                              \\,(1 - 0.3\\,\\mathcal{W}_{u,2}),\\\\
      V_2(\\mathcal{W}_u) &= 0.4\\, e^{-1.8\\,\\mathcal{W}_{u,2}}
                              \\,(1 - 0.25\\,\\mathcal{W}_{u,1}).

  Used throughout Sections 6.2-6.5 of the paper.

The class exposes :meth:`field`, :meth:`jacobian`, and :meth:`hessian`,
all vectorised in the leading axes of the input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Velocity",
    "VelocityExpDensity",
    "make_velocity",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Velocity:
    """
    Abstract base class for a velocity field :math:`V : \\mathbb{R}^p \\to \\mathbb{R}^d`.

    Concrete subclasses must implement:

    * :meth:`field(Wu)` — value :math:`V(W_u)`,  shape ``Wu.shape``.
    * :meth:`jacobian(Wu)` — :math:`(J_V)_{dc} = \\partial V_d / \\partial W_c`,
      shape ``Wu.shape + (p,)``.
    * :meth:`hessian(Wu)` — :math:`(H_V)_{dco} = \\partial^2 V_d /
      \\partial W_c \\partial W_o`, shape ``Wu.shape + (p, p)``.

    All methods must be vectorised: ``Wu`` may be a single vector
    ``shape == (p,)`` or a batch ``shape == (..., p)``.
    """

    name: str = "abstract"

    def field(self, Wu):  # pragma: no cover
        raise NotImplementedError

    def jacobian(self, Wu):  # pragma: no cover
        raise NotImplementedError

    def hessian(self, Wu):  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Exponential density-dependent velocity
# ---------------------------------------------------------------------------

@dataclass
class VelocityExpDensity(Velocity):
    r"""
    Exponential density-dependent velocity with cross-inhibition.

    .. math::
        V_1(W) &= 0.5\, e^{-1.5\, W_1}\,(1 - 0.3\, W_2),\\
        V_2(W) &= 0.4\, e^{-1.8\, W_2}\,(1 - 0.25\, W_1).

    Used in the convergence study (Tables 2, 3), the diffusion-coefficient
    sweep (Table 4), and the hyper-viscosity experiments (Table 5).
    """

    name: str = "exp_density"

    # -- field --
    def field(self, Wu):
        Wu = np.asarray(Wu, dtype=float)
        e0 = np.exp(-1.5 * Wu[..., 0])
        e1 = np.exp(-1.8 * Wu[..., 1])
        f0 = 1.0 - 0.30 * Wu[..., 1]
        f1 = 1.0 - 0.25 * Wu[..., 0]
        V = np.empty(Wu.shape)
        V[..., 0] = 0.5 * e0 * f0
        V[..., 1] = 0.4 * e1 * f1
        return V

    # -- Jacobian J[d, c] = dV[d] / dW[c] --
    def jacobian(self, Wu):
        Wu = np.asarray(Wu, dtype=float)
        e0 = np.exp(-1.5 * Wu[..., 0])
        e1 = np.exp(-1.8 * Wu[..., 1])
        f0 = 1.0 - 0.30 * Wu[..., 1]
        f1 = 1.0 - 0.25 * Wu[..., 0]
        J = np.empty(Wu.shape + (2,))
        J[..., 0, 0] = -0.75 * e0 * f0     # dV1/dW1
        J[..., 0, 1] = -0.15 * e0          # dV1/dW2
        J[..., 1, 0] = -0.10 * e1          # dV2/dW1
        J[..., 1, 1] = -0.72 * e1 * f1     # dV2/dW2
        return J

    # -- Hessian H[d, c, o] = d^2 V[d] / dW[c] dW[o] --
    def hessian(self, Wu):
        Wu = np.asarray(Wu, dtype=float)
        e0 = np.exp(-1.5 * Wu[..., 0])
        e1 = np.exp(-1.8 * Wu[..., 1])
        f0 = 1.0 - 0.30 * Wu[..., 1]
        f1 = 1.0 - 0.25 * Wu[..., 0]

        H = np.zeros(Wu.shape + (2, 2))

        # V1 = 0.5 * exp(-1.5 W1) * (1 - 0.3 W2)
        H[..., 0, 0, 0] =  1.125 * e0 * f0    # d2V1/dW1^2
        H[..., 0, 0, 1] =  0.225 * e0         # d2V1/dW1 dW2  (NOTE: non-zero!)
        H[..., 0, 1, 0] =  0.225 * e0         # symmetric
        H[..., 0, 1, 1] =  0.0                # V1 is linear in W2

        # V2 = 0.4 * exp(-1.8 W2) * (1 - 0.25 W1)
        H[..., 1, 0, 0] =  0.0                # V2 is linear in W1
        H[..., 1, 0, 1] =  0.18 * e1          # d2V2/dW1 dW2  (NOTE: non-zero!)
        H[..., 1, 1, 0] =  0.18 * e1          # symmetric
        H[..., 1, 1, 1] =  1.296 * e1 * f1    # d2V2/dW2^2

        return H


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "exp_density":   VelocityExpDensity,
}


def make_velocity(name: str, **kwargs) -> Velocity:
    """
    Construct a velocity field by name.

    Parameters
    ----------
    name : str
        ``"exp_density"`` (case-insensitive).

    Returns
    -------
    Velocity
        A concrete velocity field.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown velocity {name!r}. Available: {tuple(_REGISTRY)}."
        )
    return _REGISTRY[key](**kwargs)