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
    block: int = 1024,
) -> Array:
    """Acceleration on every body from every other body, summed exactly in O(N^2).

    a_i = G * sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2), with j != i.
    Rows are processed in blocks so memory stays O(block * N) instead of O(N^2).
    """
    n = len(mass)
    acc = np.zeros_like(pos)
    for start in range(0, n, block):
        stop = min(start + block, n)
        diff = pos[None, :, :] - pos[start:stop, None, :]  # (B, N, 3): r_j - r_i
        r2 = np.einsum("bnk,bnk->bn", diff, diff) + softening**2
        inv_r3 = np.zeros_like(r2)
        nonzero = r2 > 0.0
        inv_r3[nonzero] = r2[nonzero] ** -1.5
        rows = np.arange(start, stop)
        inv_r3[rows - start, rows] = 0.0  # no self-force
        acc[start:stop] = G * np.einsum("bn,bnk->bk", inv_r3 * mass[None, :], diff)
    return acc


def make_direct_accel(mass: Array, softening: float = 0.0, G: float = G_DEFAULT) -> AccelFn:
    """Bind masses, softening and G so integrators only need to pass positions."""

    def accel(pos: Array) -> Array:
        return direct_accel(pos, mass, softening, G)

    return accel
