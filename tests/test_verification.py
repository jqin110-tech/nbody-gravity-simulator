"""Stage 2: the simulator against known physics (Kepler, conservation laws, convergence order)."""

import numpy as np
import pytest

from nbody.diagnostics import angular_momentum, linear_momentum, total_energy
from nbody.forces import make_direct_accel
from nbody.integrators import INTEGRATORS, euler_step, leapfrog_step, rk4_step
from nbody.simulation import System, run
from nbody.twobody import kepler_period, measured_period, relative_position, two_body_system


@pytest.mark.parametrize("a, m1, m2, e", [(1.0, 1.0, 1e-3, 0.0), (2.0, 1.0, 1e-3, 0.5), (1.0, 0.5, 0.5, 0.3)])
def test_two_body_period_matches_keplers_third_law(a: float, m1: float, m2: float, e: float) -> None:
    system = two_body_system(m1, m2, a, e)
    T = kepler_period(a, m1 + m2)
    n = 5000
    traj = run(system, leapfrog_step, make_direct_accel(system.mass), T / n, int(1.05 * n))
    T_sim = measured_period(traj.times, traj.pos[:, 1] - traj.pos[:, 0])
    assert abs(T_sim - T) / T < 2e-5


def test_one_au_orbit_around_one_solar_mass_takes_one_year() -> None:
    assert kepler_period(1.0, 1.0) == pytest.approx(1.0, rel=1e-15)


def _energy_errors(stepper, n_orbits: int = 30, steps_per_orbit: int = 200) -> np.ndarray:
    system = two_body_system(1.0, 1e-3, 1.0, 0.5)
    T = kepler_period(1.0, 1.001)
    traj = run(system, stepper, make_direct_accel(system.mass), T / steps_per_orbit,
               n_orbits * steps_per_orbit, record_every=steps_per_orbit // 4)
    E = np.array([total_energy(p, v, traj.mass) for p, v in zip(traj.pos, traj.vel)])
    return np.abs(E - E[0]) / abs(E[0])


def test_euler_energy_drifts_but_leapfrog_stays_bounded() -> None:
    euler, leap = _energy_errors(euler_step, steps_per_orbit=2000), _energy_errors(leapfrog_step, steps_per_orbit=2000)
    quarter = len(leap) // 4
    # Leapfrog: the worst error in the last quarter has not grown (10% slack for sampling).
    assert leap[-quarter:].max() <= 1.1 * leap[:quarter].max()
    # Euler: the error keeps growing and ends far larger than leapfrog's.
    assert euler[-quarter:].max() > 1.5 * euler[:quarter].max()
    assert euler[-1] > 100 * leap.max()


def test_leapfrog_conserves_angular_and_linear_momentum() -> None:
    rng = np.random.default_rng(2)
    system = System(rng.normal(size=(10, 3)), 0.3 * rng.normal(size=(10, 3)), rng.uniform(0.1, 1, 10))
    traj = run(system, leapfrog_step, make_direct_accel(system.mass, softening=0.05), 1e-3, 500)
    L0, L1 = angular_momentum(traj.pos[0], traj.vel[0], traj.mass), angular_momentum(traj.pos[-1], traj.vel[-1], traj.mass)
    P0, P1 = linear_momentum(traj.vel[0], traj.mass), linear_momentum(traj.vel[-1], traj.mass)
    assert np.allclose(L0, L1, atol=1e-12)
    assert np.allclose(P0, P1, atol=1e-12)


def test_softened_energy_is_conserved_through_a_head_on_collision() -> None:
    # A head-on collision: without softening the force would be infinite at r = 0. The softened
    # potential matches the softened force, so leapfrog with a small step still conserves energy.
    system = System([[-1, 0, 0], [1, 0, 0]], [[1, 0, 0], [-1, 0, 0]], [1.0, 1.0])
    eps = 0.05
    traj = run(system, leapfrog_step, make_direct_accel(system.mass, eps), 1e-5, 100_000, record_every=250)
    E = [total_energy(p, v, traj.mass, eps) for p, v in zip(traj.pos, traj.vel)]
    assert max(abs(e - E[0]) for e in E) / abs(E[0]) < 1e-3


@pytest.mark.parametrize("stepper, order, steps", [
    (euler_step, 1, [8000, 16000, 32000]),
    (leapfrog_step, 2, [500, 1000, 2000]),
    (rk4_step, 4, [250, 500, 1000]),
])
def test_convergence_order(stepper, order: int, steps: list[int]) -> None:
    """Halving dt should divide the error by about 2**order (2, 4, 16)."""
    m1, m2, a, e = 1.0, 1e-3, 1.0, 0.5
    system = two_body_system(m1, m2, a, e)
    T = kepler_period(a, m1 + m2)
    exact = relative_position(T, m1 + m2, a, e)
    errors = []
    for n in steps:
        traj = run(system, stepper, make_direct_accel(system.mass), T / n, n, record_every=n)
        errors.append(np.linalg.norm(traj.pos[-1, 1] - traj.pos[-1, 0] - exact))
    measured = np.log2(errors[-2] / errors[-1])
    assert measured == pytest.approx(order, abs=0.15)
