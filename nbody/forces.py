"""Direct-sum gravitational accelerations with Plummer softening."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from nbody.constants import G as G_DEFAULT

Array = NDArray[np.float64]
AccelFn = Callable[[Array], Array]
"""A function that maps positions (N, 3) to accelerations (N, 3)."""


def direct_accel(
    pos: Array,
    mass: Array,
    softening: float = 0.0,
    G: float = G_DEFAULT,
    block: int | None = None,
) -> Array:
    """Acceleration on every body from every other body, summed exactly in O(N^2).

    a_i = G * sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2), with j != i.
    Rows are processed in blocks so temporary arrays stay around 2 million pairs (tens of MB).
    """
    n = len(mass)
    block = block or max(1, 2_000_000 // max(n, 1))
    acc = np.zeros_like(pos)
    for start in range(0, n, block):
        stop = min(start + block, n)
        diff = pos[None, :, :] - pos[start:stop, None, :]  # (B, N, 3): r_j - r_i
        r2 = np.einsum("bnk,bnk->bn", diff, diff) + softening**2
        rows = np.arange(start, stop)
        r2[rows - start, rows] = np.inf  # a body exerts no force on itself: 1 / inf = 0
        inv_r3 = 1.0 / (r2 * np.sqrt(r2))
        acc[start:stop] = G * np.einsum("bn,bnk->bk", inv_r3 * mass[None, :], diff)
    return acc


def make_direct_accel(mass: Array, softening: float = 0.0, G: float = G_DEFAULT) -> AccelFn:
    """Bind masses, softening and G so integrators only need to pass positions."""

    def accel(pos: Array) -> Array:
        return direct_accel(pos, mass, softening, G)

    return accel
