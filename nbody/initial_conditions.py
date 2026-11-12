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


def exponential_disc_positions(n: int, rng: np.random.Generator, scale_length: float,
                               scale_height: float, r_max: float) -> Array:
    """Positions in a thin disc with surface density proportional to exp(-R / R_d) (Freeman 1970, ApJ 160, 811).

    The radius of a random particle then has density R exp(-R / R_d), a Gamma(2, R_d) distribution.
    Particles beyond r_max are redrawn; heights are Gaussian with standard deviation scale_height.
    """
    radius = rng.gamma(2.0, scale_length, size=n)
    while np.any(far := radius > r_max):
        radius[far] = rng.gamma(2.0, scale_length, size=int(far.sum()))
    phi = rng.uniform(0, 2 * np.pi, size=n)
    z = rng.normal(0.0, scale_height, size=n)
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])


def circular_velocities(pos: Array, acc: Array) -> Array:
    """Velocities for circular, counter-clockwise orbits in the x-y plane: v^2 / R = inward radial acceleration.

    Using the simulation's own accelerations means the disc starts in balance with the
    forces the code will actually compute, not with an approximate formula.
    """
    radius = np.hypot(pos[:, 0], pos[:, 1])
    r_hat = np.column_stack([pos[:, 0], pos[:, 1]]) / radius[:, None]
    inward = -(acc[:, 0] * r_hat[:, 0] + acc[:, 1] * r_hat[:, 1])
    speed = np.sqrt(radius * np.maximum(inward, 0.0))
    return np.column_stack([-speed * r_hat[:, 1], speed * r_hat[:, 0], np.zeros(len(pos))])


def parabolic_orbit(total_mass: float, pericentre: float, separation: float, G: float = 1.0) -> tuple[Array, Array]:
    """Relative position and velocity of two bodies approaching on a parabolic (e = 1) orbit.

    Toomre & Toomre (1972, ApJ 178, 623) used parabolic encounters for colliding galaxies.
    For e = 1, the orbit equation is r = 2q / (1 + cos f), and the velocity in the orbital
    plane is sqrt(G M / p) * (-sin f, 1 + cos f) with p = 2q.
    """
    cos_f = 2.0 * pericentre / separation - 1.0
    f = -np.arccos(cos_f)  # negative true anomaly: still on the way in
    p = 2.0 * pericentre
    rel_pos = separation * np.array([np.cos(f), np.sin(f), 0.0])
    rel_vel = np.sqrt(G * total_mass / p) * np.array([-np.sin(f), 1.0 + np.cos(f), 0.0])
    return rel_pos, rel_vel


def rotation_x(angle: float) -> Array:
    """Matrix that rotates vectors by `angle` radians about the x axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
