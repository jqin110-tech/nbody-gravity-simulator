"""Stage 4: Barnes-Hut against the exact direct sum."""

import numpy as np
import pytest

from nbody.barnes_hut import NUMBA, bh_accel, build, direct_accel_compiled
from nbody.forces import direct_accel
from nbody.initial_conditions import plummer_sphere

BACKENDS = ["python"] + (["numba"] if NUMBA is not None else [])


def _cluster(n: int, seed: int = 0):
    s = plummer_sphere(n, np.random.default_rng(seed))
    return s.pos, s.mass


def _relative_errors(approx: np.ndarray, exact: np.ndarray) -> np.ndarray:
    return np.linalg.norm(approx - exact, axis=1) / np.linalg.norm(exact, axis=1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_theta_zero_is_the_exact_direct_sum(backend: str) -> None:
    pos, mass = _cluster(300)
    exact = direct_accel(pos, mass, 0.01, G=1.0)
    assert np.allclose(bh_accel(pos, mass, 0.0, 0.01, G=1.0, backend=backend), exact, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_compiled_direct_sum_matches_numpy(backend: str) -> None:
    pos, mass = _cluster(200)
    assert np.allclose(direct_accel_compiled(pos, mass, 0.01, 1.0, backend), direct_accel(pos, mass, 0.01, 1.0))


def test_force_error_is_small_and_shrinks_with_theta() -> None:
    pos, mass = _cluster(2000)
    exact = direct_accel(pos, mass, 0.01, G=1.0)
    backend = BACKENDS[-1]
    median = {t: np.median(_relative_errors(bh_accel(pos, mass, t, 0.01, 1.0, backend), exact)) for t in (0.3, 0.5, 0.8)}
    assert median[0.3] < median[0.5] < median[0.8]
    assert median[0.5] < 1e-2


@pytest.mark.skipif(NUMBA is None, reason="numba not installed")
def test_python_and_numba_give_identical_answers() -> None:
    pos, mass = _cluster(500)
    a_py = bh_accel(pos, mass, 0.6, 0.01, 1.0, backend="python")
    a_nb = bh_accel(pos, mass, 0.6, 0.01, 1.0, backend="numba")
    assert np.allclose(a_py, a_nb, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
def test_root_node_holds_total_mass_at_centre_of_mass(backend: str) -> None:
    pos, mass = _cluster(400)
    center, half, kind, children, node_mass, com = build(pos, mass, backend)
    assert node_mass[0] == pytest.approx(mass.sum())
    assert np.allclose(com[0], (mass[:, None] * pos).sum(0) / mass.sum())
    leaf_nodes = np.flatnonzero(kind >= 0)
    bodies = kind[leaf_nodes]
    assert sorted(bodies.tolist()) == list(range(len(mass)))  # every body sits in exactly one leaf
    # ... and that leaf's cube really contains it.
    offset = np.abs(pos[bodies] - center[leaf_nodes])
    assert np.all(offset <= half[leaf_nodes, None] * (1 + 1e-12))


def test_identical_positions_raise_a_clear_error() -> None:
    pos = np.zeros((2, 3))
    with pytest.raises(ValueError):
        bh_accel(pos, np.ones(2), backend="python")


@pytest.mark.parametrize("backend", BACKENDS)
def test_compiled_potential_matches_numpy_version(backend: str) -> None:
    from nbody.barnes_hut import potential_energy_compiled
    from nbody.diagnostics import potential_energy

    pos, mass = _cluster(300)
    assert potential_energy_compiled(pos, mass, 0.01, 1.0, backend) == pytest.approx(potential_energy(pos, mass, 0.01, 1.0))
