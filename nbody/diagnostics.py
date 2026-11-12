"""Conserved quantities used to check that a simulation is physically correct."""

import numpy as np

from nbody.constants import G as G_DEFAULT
from nbody.forces import Array


def kinetic_energy(vel: Array, mass: Array) -> float:
    """Total kinetic energy, sum of (1/2) m v^2."""
    return float(0.5 * np.sum(mass * np.einsum("nk,nk->n", vel, vel)))


def potential_energy(pos: Array, mass: Array, softening: float = 0.0, G: float = G_DEFAULT) -> float:
    """Total potential energy, -G sum over pairs i<j of m_i m_j / sqrt(r_ij^2 + eps^2).

    This is the potential whose gradient is the softened force in forces.py, so the
    sum of kinetic and potential energy is exactly conserved by the true dynamics.
    """
    total = 0.0
    for i in range(len(mass) - 1):
        diff = pos[i + 1 :] - pos[i]
        r = np.sqrt(np.einsum("nk,nk->n", diff, diff) + softening**2)
        total -= G * mass[i] * np.sum(mass[i + 1 :] / r)
    return float(total)


def total_energy(pos: Array, vel: Array, mass: Array, softening: float = 0.0, G: float = G_DEFAULT) -> float:
    """Kinetic plus potential energy."""
    return kinetic_energy(vel, mass) + potential_energy(pos, mass, softening, G)


def angular_momentum(pos: Array, vel: Array, mass: Array) -> Array:
    """Total angular momentum vector, sum of m (r x v)."""
    return np.sum(mass[:, None] * np.cross(pos, vel), axis=0)


def linear_momentum(vel: Array, mass: Array) -> Array:
    """Total linear momentum vector, sum of m v."""
    return np.sum(mass[:, None] * vel, axis=0)
