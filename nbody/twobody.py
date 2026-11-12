"""Exact two-body (Kepler) solution, used to test the integrators against a known answer."""

import math

import numpy as np

from nbody.constants import G as G_DEFAULT
from nbody.forces import Array
from nbody.simulation import System


def kepler_period(a: float, total_mass: float, G: float = G_DEFAULT) -> float:
    """Kepler's third law: T = 2 pi sqrt(a^3 / (G M)). With G = 4 pi^2 this is a^1.5 / sqrt(M) years."""
    return 2.0 * math.pi * math.sqrt(a**3 / (G * total_mass))


def two_body_system(m1: float, m2: float, a: float, e: float, G: float = G_DEFAULT) -> System:
    """Two bodies starting at periapsis of an orbit with semi-major axis a and eccentricity e.

    The relative speed at periapsis comes from the vis-viva equation
    v^2 = G M (2/r - 1/a), and both bodies are placed in the centre-of-mass frame.
    """
    total = m1 + m2
    r_peri = a * (1.0 - e)
    v_peri = math.sqrt(G * total * (2.0 / r_peri - 1.0 / a))
    rel_pos = np.array([r_peri, 0.0, 0.0])
    rel_vel = np.array([0.0, v_peri, 0.0])
    pos = np.array([-m2 / total * rel_pos, m1 / total * rel_pos])
    vel = np.array([-m2 / total * rel_vel, m1 / total * rel_vel])
    return System(pos, vel, np.array([m1, m2]))


def solve_kepler_equation(mean_anomaly: float, e: float, tol: float = 1e-15) -> float:
    """Solve E - e sin E = M for the eccentric anomaly E by Newton's method."""
    E = mean_anomaly if e < 0.8 else math.pi
    for _ in range(100):
        step = (E - e * math.sin(E) - mean_anomaly) / (1.0 - e * math.cos(E))
        E -= step
        if abs(step) < tol:
            break
    return E


def relative_position(t: float, total_mass: float, a: float, e: float, G: float = G_DEFAULT) -> Array:
    """Exact position of body 2 relative to body 1 at time t after periapsis."""
    n = 2.0 * math.pi / kepler_period(a, total_mass, G)  # mean motion
    E = solve_kepler_equation(n * t, e)
    return np.array([a * (math.cos(E) - e), a * math.sqrt(1.0 - e**2) * math.sin(E), 0.0])


def measured_period(times: Array, rel_pos: Array) -> float:
    """Time for the relative position vector to sweep one full turn (2 pi), linearly interpolated."""
    angle = np.unwrap(np.arctan2(rel_pos[:, 1], rel_pos[:, 0]))
    swept = angle - angle[0]
    k = int(np.argmax(swept >= 2.0 * math.pi))
    if swept[k] < 2.0 * math.pi:
        raise ValueError("orbit did not complete a full turn")
    frac = (2.0 * math.pi - swept[k - 1]) / (swept[k] - swept[k - 1])
    return float(times[k - 1] + frac * (times[k] - times[k - 1]))
