"""Stage 5: the galaxy building blocks match their textbook properties."""

import numpy as np
import pytest

from nbody.diagnostics import kinetic_energy, potential_energy
from nbody.initial_conditions import (circular_velocities, exponential_disc_positions, parabolic_orbit,
                                      plummer_sphere)


def test_plummer_sphere_has_the_analytic_half_mass_radius() -> None:
    # M(r) = r^3 / (1 + r^2)^(3/2) = 1/2 gives r = 1 / sqrt(2^(2/3) - 1), about 1.305 scale radii.
    system = plummer_sphere(20_000, np.random.default_rng(0))
    radius = np.linalg.norm(system.pos, axis=1)
    assert np.median(radius) == pytest.approx(1.0 / np.sqrt(2 ** (2 / 3) - 1), rel=0.03)
    assert system.mass.sum() == pytest.approx(1.0)


def test_plummer_sphere_is_in_virial_equilibrium() -> None:
    # A system in equilibrium has 2 * kinetic energy = |potential energy| (virial theorem).
    system = plummer_sphere(3000, np.random.default_rng(1))
    ratio = 2 * kinetic_energy(system.vel, system.mass) / abs(potential_energy(system.pos, system.mass, G=1.0))
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_parabolic_orbit_has_zero_energy_and_the_requested_pericentre() -> None:
    total_mass, pericentre, separation = 2.0, 1.0, 10.0
    rel_pos, rel_vel = parabolic_orbit(total_mass, pericentre, separation)
    assert np.linalg.norm(rel_pos) == pytest.approx(separation)
    assert 0.5 * rel_vel @ rel_vel - total_mass / separation == pytest.approx(0.0, abs=1e-12)
    # Angular momentum per unit mass of a parabola with pericentre q is sqrt(2 G M q).
    assert np.cross(rel_pos, rel_vel)[2] == pytest.approx(np.sqrt(2 * total_mass * pericentre))
    assert rel_pos @ rel_vel < 0  # the galaxies start out approaching each other


def test_circular_velocity_around_a_point_mass() -> None:
    # Around a point mass M the inward acceleration is G M / R^2, so v = sqrt(G M / R).
    pos = np.array([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    radius = np.linalg.norm(pos, axis=1)
    acc = -pos / radius[:, None] ** 3  # G = M = 1
    vel = circular_velocities(pos, acc)
    assert np.allclose(np.linalg.norm(vel, axis=1), 1.0 / np.sqrt(radius))
    assert np.allclose(np.einsum("nk,nk->n", pos, vel), 0.0)  # velocity is perpendicular to radius
    assert np.all(np.cross(pos, vel)[:, 2] > 0)  # counter-clockwise


def test_exponential_disc_is_thin_and_has_the_right_mean_radius() -> None:
    # Radii follow Gamma(2, R_d), whose mean is 2 R_d (the cut at r_max = 20 R_d removes almost nothing).
    pos = exponential_disc_positions(50_000, np.random.default_rng(2), scale_length=0.5, scale_height=0.03, r_max=10.0)
    radius = np.hypot(pos[:, 0], pos[:, 1])
    assert radius.mean() == pytest.approx(1.0, rel=0.02)
    assert radius.max() <= 10.0
    assert pos[:, 2].std() == pytest.approx(0.03, rel=0.05)
