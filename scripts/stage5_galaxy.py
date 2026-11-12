"""Stage 5a: two disc galaxies colliding, simulated with Barnes-Hut and leapfrog.

Model (N-body units, G = 1, each galaxy has total mass 1):
  * a Plummer-sphere halo (Plummer 1911; sampled as in Aarseth, Henon & Wielen 1974),
  * an exponential disc (Freeman 1970) whose particles start on circular orbits
    computed from the code's own forces,
  * the two galaxies approach on a parabolic orbit, as in Toomre & Toomre (1972).
The sizes, mass split, pericentre and tilt below are modelling choices, not measured
constants; they are picked to give a clear encounter in a short run.

Run: python -m scripts.stage5_galaxy            (about 100k bodies; tens of minutes)
     python -m scripts.stage5_galaxy --quick    (small version for checking the pipeline)
Writes results/stage5_galaxy.npz (snapshots) and results/stage5_galaxy.json (diagnostics).
"""

import argparse
import time

import numpy as np

from nbody.barnes_hut import bh_accel, potential_energy_compiled
from nbody.diagnostics import angular_momentum, kinetic_energy, linear_momentum
from nbody.initial_conditions import (circular_velocities, exponential_disc_positions, parabolic_orbit,
                                      plummer_sphere, rotation_x)
from nbody.integrators import leapfrog_step
from nbody.simulation import System
from scripts._common import RESULTS, save_json

HALO_FRACTION = 0.75  # share of each galaxy's mass in the halo
HALO_SCALE = 1.0  # Plummer scale radius
DISC_SCALE_LENGTH, DISC_SCALE_HEIGHT, DISC_R_MAX = 0.5, 0.03, 3.0
PERICENTRE, START_SEPARATION = 1.0, 10.0
TILT_DEGREES = 60.0  # second galaxy's disc is tilted about the x axis
THETA, SOFTENING, DT = 0.7, 0.03, 0.01
T_END = 30.0
RECORD_EVERY = 10
SEED = 2026


def make_galaxy(n_halo: int, n_disc: int, rng: np.random.Generator) -> tuple[System, np.ndarray]:
    """One galaxy at rest at the origin; returns the system and a boolean mask marking disc particles."""
    halo = plummer_sphere(n_halo, rng, total_mass=HALO_FRACTION, scale=HALO_SCALE)
    disc_pos = exponential_disc_positions(n_disc, rng, DISC_SCALE_LENGTH, DISC_SCALE_HEIGHT, DISC_R_MAX)
    disc_mass = np.full(n_disc, (1.0 - HALO_FRACTION) / n_disc)
    pos = np.vstack([halo.pos, disc_pos])
    mass = np.concatenate([halo.mass, disc_mass])
    acc = bh_accel(pos, mass, 0.3, SOFTENING, G=1.0)  # accurate theta for setting up orbits
    vel = np.vstack([halo.vel, circular_velocities(disc_pos, acc[n_halo:])])
    is_disc = np.concatenate([np.zeros(n_halo, bool), np.ones(n_disc, bool)])
    return System(pos, vel, mass), is_disc


def make_encounter(n_total: int, rng: np.random.Generator) -> tuple[System, np.ndarray]:
    """Two equal galaxies on a parabolic orbit in the x-y plane, in the centre-of-mass frame."""
    n_galaxy = n_total // 2
    n_disc = int(round(0.4 * n_galaxy))  # 40% of particles in the disc so it shows up well in images
    g1, disc1 = make_galaxy(n_galaxy - n_disc, n_disc, rng)
    g2, disc2 = make_galaxy(n_galaxy - n_disc, n_disc, rng)
    tilt = rotation_x(np.radians(TILT_DEGREES))
    rel_pos, rel_vel = parabolic_orbit(2.0, PERICENTRE, START_SEPARATION)
    pos = np.vstack([g1.pos - 0.5 * rel_pos, g2.pos @ tilt.T + 0.5 * rel_pos])
    vel = np.vstack([g1.vel - 0.5 * rel_vel, g2.vel @ tilt.T + 0.5 * rel_vel])
    mass = np.concatenate([g1.mass, g2.mass])
    return System(pos, vel, mass), np.concatenate([disc1, disc2])


def main(n_total: int, t_end: float, out_name: str) -> None:
    rng = np.random.default_rng(SEED)
    system, is_disc = make_encounter(n_total, rng)
    accel = lambda p: bh_accel(p, system.mass, THETA, SOFTENING, G=1.0)  # noqa: E731
    n_steps = int(round(t_end / DT))
    pos, vel = system.pos.copy(), system.vel.copy()
    acc = accel(pos)
    snapshots, times, diag = [pos.astype(np.float32)], [0.0], []
    start = time.perf_counter()
    for step in range(1, n_steps + 1):
        pos, vel, acc = leapfrog_step(pos, vel, acc, DT, accel)
        if step % RECORD_EVERY == 0:
            snapshots.append(pos.astype(np.float32))
            times.append(step * DT)
            elapsed = time.perf_counter() - start
            print(f"step {step}/{n_steps}  t={step * DT:.2f}  {elapsed / step:.3f} s/step", flush=True)
        if step % (n_steps // 10) == 0 or step == n_steps:
            # Kinetic energy and momenta are cheap; the O(N^2) potential is only computed at start and end.
            diag.append({"t": step * DT, "kinetic": kinetic_energy(vel, system.mass),
                         "momentum": linear_momentum(vel, system.mass).tolist(),
                         "angular_momentum": angular_momentum(pos, vel, system.mass).tolist()})
    wall = time.perf_counter() - start
    E0 = kinetic_energy(system.vel, system.mass) + potential_energy_compiled(system.pos, system.mass, SOFTENING, 1.0)
    E1 = kinetic_energy(vel, system.mass) + potential_energy_compiled(pos, system.mass, SOFTENING, 1.0)
    L0 = angular_momentum(system.pos, system.vel, system.mass)
    L1 = angular_momentum(pos, vel, system.mass)
    np.savez_compressed(RESULTS / f"{out_name}.npz", pos=np.array(snapshots), times=np.array(times),
                        is_disc=is_disc, mass=system.mass)
    save_json(out_name, {
        "n_bodies": system.n, "n_disc": int(is_disc.sum()), "theta": THETA, "softening": SOFTENING,
        "dt": DT, "t_end": t_end, "n_steps": n_steps, "integrator": "leapfrog", "seed": SEED,
        "wall_seconds": wall, "seconds_per_step": wall / n_steps,
        "energy_initial": E0, "energy_final": E1, "energy_rel_change": abs(E1 - E0) / abs(E0),
        "angular_momentum_rel_change": float(np.linalg.norm(L1 - L0) / np.linalg.norm(L0)),
        "final_momentum_magnitude": float(np.linalg.norm(linear_momentum(vel, system.mass))),
        "diagnostics": diag})
    print(f"energy change |dE/E| = {abs(E1 - E0) / abs(E0):.2e}")
    print(f"done: {system.n} bodies, {n_steps} steps in {wall:.0f} s ({wall / n_steps:.2f} s/step)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="10k bodies, short run")
    args = parser.parse_args()
    if args.quick:
        main(10_000, 2.0, "stage5_galaxy_quick")
    else:
        main(100_000, T_END, "stage5_galaxy")
