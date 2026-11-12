"""Stage 1: the force law and the integrator interface."""

import numpy as np
import pytest

from nbody.constants import G
from nbody.forces import direct_accel, make_direct_accel
from nbody.integrators import INTEGRATORS
from nbody.simulation import System, run


def naive_accel(pos: np.ndarray, mass: np.ndarray, eps: float) -> np.ndarray:
    """Textbook double loop, used only as a reference for the vectorised version."""
    acc = np.zeros_like(pos)
    for i in range(len(mass)):
        for j in range(len(mass)):
            if i != j:
                d = pos[j] - pos[i]
                acc[i] += G * mass[j] * d / (d @ d + eps**2) ** 1.5
    return acc


def test_two_body_force_matches_newton() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mass = np.array([1.0, 3.0])
    acc = direct_accel(pos, mass)
    # |a_0| = G m_1 / r^2 toward body 1, |a_1| = G m_0 / r^2 toward body 0.
    assert np.allclose(acc[0], [G * 3.0 / 4.0, 0, 0])
    assert np.allclose(acc[1], [-G * 1.0 / 4.0, 0, 0])


@pytest.mark.parametrize("eps", [0.0, 0.1])
@pytest.mark.parametrize("block", [1, 7, 1024])
def test_vectorised_matches_double_loop(eps: float, block: int) -> None:
    rng = np.random.default_rng(0)
    pos, mass = rng.normal(size=(20, 3)), rng.uniform(0.1, 1.0, size=20)
    assert np.allclose(direct_accel(pos, mass, eps, block=block), naive_accel(pos, mass, eps))


def test_forces_obey_newtons_third_law() -> None:
    rng = np.random.default_rng(1)
    pos, mass = rng.normal(size=(50, 3)), rng.uniform(0.1, 1.0, size=50)
    net_force = np.sum(mass[:, None] * direct_accel(pos, mass, 0.05), axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-10)


def test_softening_caps_close_encounter_force() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1e-4, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    hard = np.linalg.norm(direct_accel(pos, mass, 0.0)[0])
    soft = np.linalg.norm(direct_accel(pos, mass, 0.01)[0])
    assert soft < hard / 1e3


@pytest.mark.parametrize("name", INTEGRATORS)
def test_free_particles_move_in_straight_lines(name: str) -> None:
    # With G = 0 there is no force, so every integrator must be exact.
    system = System(pos=[[0, 0, 0], [1, 1, 1]], vel=[[1, 2, 3], [-1, 0, 0.5]], mass=[1, 1])
    accel = make_direct_accel(system.mass, G=0.0)
    traj = run(system, INTEGRATORS[name], accel, dt=0.1, n_steps=10)
    assert np.allclose(traj.pos[-1], system.pos + 1.0 * system.vel)
    assert traj.pos.shape == (11, 2, 3)


def test_system_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        System(pos=np.zeros((2, 2)), vel=np.zeros((2, 3)), mass=np.ones(2))
