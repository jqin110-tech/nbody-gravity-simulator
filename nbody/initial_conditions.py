"""Initial conditions drawn from published models, in N-body units (G = 1)."""

import numpy as np

from nbody.forces import Array
from nbody.simulation import System


def _random_directions(n: int, rng: np.random.Generator) -> Array:
    """n unit vectors spread uniformly over the sphere."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def plummer_sphere(n: int, rng: np.random.Generator, total_mass: float = 1.0, scale: float = 1.0) -> System:
    """Equal-mass Plummer (1911) sphere in equilibrium, with G = 1.

    Sampling follows Aarseth, Henon & Wielen (1974, A&A 37, 183): radii by inverting the
    cumulative mass M(r) = r^3 / (1 + r^2)^(3/2), speeds by rejection sampling of
    g(q) = q^2 (1 - q^2)^(7/2) with q = v / v_escape.
    """
    x = rng.uniform(1e-10, 1.0, size=n)  # fraction of mass inside r; avoid exactly 0
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    q = np.empty(n)
    filled = 0
    while filled < n:
        trial_q, trial_g = rng.uniform(0, 1, size=n), rng.uniform(0, 0.1, size=n)
        accepted = trial_q[trial_g < trial_q**2 * (1 - trial_q**2) ** 3.5]
        take = min(len(accepted), n - filled)
        q[filled : filled + take] = accepted[:take]
        filled += take
    v_escape = np.sqrt(2.0) * (1.0 + r**2) ** -0.25
    pos = scale * r[:, None] * _random_directions(n, rng)
    vel = np.sqrt(total_mass / scale) * (q * v_escape)[:, None] * _random_directions(n, rng)
    return System(pos, vel, np.full(n, total_mass / n))

