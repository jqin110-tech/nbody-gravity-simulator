"""Stage 3: the saved JPL / NASA data load correctly and give a sensible Earth year."""

import numpy as np
import pytest

from nbody.constants import DAYS_PER_YEAR, G
from nbody.diagnostics import linear_momentum
from nbody.forces import make_direct_accel
from nbody.integrators import leapfrog_step
from nbody.simulation import System, run
from nbody.solar_system import load_solar_system, read_reference_periods
from nbody.twobody import kepler_period, measured_period


def test_data_loads_at_the_recorded_epoch() -> None:
    system, jd = load_solar_system()
    assert jd == 2460676.5  # 2025-01-01 00:00 TDB
    assert system.n == 9
    assert system.mass[0] == 1.0
    assert system.mass[5] == pytest.approx(9.5479e-4, rel=1e-4)  # Jupiter system / Sun, from GM ratio


def test_positions_are_relative_to_the_barycentre() -> None:
    # Horizons gives barycentric vectors, so total momentum should be close to zero.
    system, _ = load_solar_system()
    momentum = linear_momentum(system.vel, system.mass)
    typical = system.mass[5] * np.linalg.norm(system.vel[5])
    assert np.linalg.norm(momentum) < 1e-3 * typical


def test_simulated_earth_year_matches_the_fact_sheet() -> None:
    system, _ = load_solar_system()
    dt = 0.5 / DAYS_PER_YEAR
    traj = run(system, leapfrog_step, make_direct_accel(system.mass), dt, int(1.1 / dt))
    period_days = measured_period(traj.times, traj.pos[:, 3] - traj.pos[:, 0]) * DAYS_PER_YEAR
    reference = read_reference_periods()["Earth"]["sidereal"]  # 365.256 days
    # One orbit from one starting point is nudged by the Moon and other planets, so allow 0.01%.
    assert period_days == pytest.approx(reference, rel=1e-4)


def test_reference_periods_are_sidereal_not_tropical() -> None:
    periods = read_reference_periods()
    assert set(periods) == {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"}
    assert all(p["sidereal"] > p["tropical"] for p in periods.values())


@pytest.mark.parametrize("planet", [1, 5, 8])  # Mercury, Jupiter, Neptune
def test_each_planet_alone_with_the_sun_obeys_kepler(planet: int) -> None:
    """With the other planets removed, the real initial conditions must give a Kepler orbit."""
    full, _ = load_solar_system()
    pair = System(full.pos[[0, planet]], full.vel[[0, planet]], full.mass[[0, planet]])
    rel_pos, rel_vel = pair.pos[1] - pair.pos[0], pair.vel[1] - pair.vel[0]
    total = pair.mass.sum()
    a = 1.0 / (2.0 / np.linalg.norm(rel_pos) - rel_vel @ rel_vel / (G * total))  # vis-viva
    T = kepler_period(a, total)
    n = 20_000
    traj = run(pair, leapfrog_step, make_direct_accel(pair.mass), T / n, int(1.05 * n))
    T_sim = measured_period(traj.times, traj.pos[:, 1] - traj.pos[:, 0])
    assert T_sim == pytest.approx(T, rel=1e-6)
