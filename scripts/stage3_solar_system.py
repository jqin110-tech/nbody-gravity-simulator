"""Stage 3: the real Sun and eight planets from JPL Horizons; measure each simulated orbital period.

Inputs (saved unmodified in data/raw/, see manifest.json for URLs and retrieval time):
  * state vectors at 2025-01-01 00:00 TDB from JPL Horizons (Solar System barycentre,
    ecliptic J2000), for the Sun and the barycentres of the eight planetary systems;
  * GM values from JPL's Astrodynamic Parameters page (DE440);
  * reference sidereal and tropical periods from the NASA NSSDC fact sheet of each planet.

Run: python -m scripts.stage3_solar_system
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nbody.constants import DAYS_PER_YEAR
from nbody.diagnostics import total_energy
from nbody.forces import make_direct_accel
from nbody.integrators import leapfrog_step
from nbody.simulation import run
from nbody.solar_system import BODIES, load_solar_system, read_reference_periods
from scripts._common import RESULTS, save_json

YEARS = 200.0  # long enough for one full Neptune orbit (~165 yr)
DT_DAYS = [0.125, 0.0625]  # run twice: if the periods barely change, the time step is not the limit
RECORD_DAYS = 1.0  # spacing of recorded snapshots


def mean_period(times: np.ndarray, rel_pos: np.ndarray) -> tuple[float, int]:
    """Average sidereal period over all complete turns of the heliocentric position in the ecliptic plane.

    Returns (period in years, number of complete turns used).
    """
    angle = np.unwrap(np.arctan2(rel_pos[:, 1], rel_pos[:, 0]))
    swept = angle - angle[0]
    turns = int(swept[-1] // (2 * np.pi))
    target = turns * 2 * np.pi
    k = int(np.argmax(swept >= target))
    frac = (target - swept[k - 1]) / (swept[k] - swept[k - 1])
    t_end = times[k - 1] + frac * (times[k] - times[k - 1])
    return float(t_end / turns), turns


def simulate(dt_days: float) -> dict:
    system, jd = load_solar_system()
    dt = dt_days / DAYS_PER_YEAR
    n_steps = int(round(YEARS / dt))
    start = time.perf_counter()
    traj = run(system, leapfrog_step, make_direct_accel(system.mass), dt, n_steps,
               record_every=int(round(RECORD_DAYS / dt_days)))
    wall = time.perf_counter() - start
    periods = {}
    for i, name in enumerate(BODIES[1:], start=1):
        T, turns = mean_period(traj.times, traj.pos[:, i] - traj.pos[:, 0])
        periods[name] = {"period_days": T * DAYS_PER_YEAR, "turns": turns}
    E = np.array([total_energy(p, v, traj.mass) for p, v in zip(traj.pos[::30], traj.vel[::30])])
    return {"dt_days": dt_days, "n_steps": n_steps, "wall_seconds": wall, "epoch_jd": jd, "periods": periods,
            "max_rel_energy_error": float(np.max(np.abs(E - E[0]) / abs(E[0]))), "traj": traj}


def main() -> None:
    reference = read_reference_periods()
    runs = [simulate(dt) for dt in DT_DAYS]
    fine, coarse = runs[-1], runs[0]
    rows = []
    print(f"{'planet':8s} {'simulated (d)':>14s} {'NASA sidereal':>14s} {'rel. error':>11s} "
          f"{'NASA tropical':>14s} {'vs tropical':>11s} {'turns':>6s} {'dt change':>10s}")
    for name in BODIES[1:]:
        sim = fine["periods"][name]["period_days"]
        sidereal, tropical = reference[name]["sidereal"], reference[name]["tropical"]
        dt_change = abs(sim - coarse["periods"][name]["period_days"]) / sim
        row = {"planet": name, "simulated_days": sim, "nasa_sidereal_days": sidereal,
               "rel_error_vs_sidereal": (sim - sidereal) / sidereal, "nasa_tropical_days": tropical,
               "rel_error_vs_tropical": (sim - tropical) / tropical, "turns": fine["periods"][name]["turns"],
               "rel_change_when_dt_halved": dt_change}
        rows.append(row)
        print(f"{name:8s} {sim:14.3f} {sidereal:14.3f} {row['rel_error_vs_sidereal']:+11.2e} "
              f"{tropical:14.3f} {row['rel_error_vs_tropical']:+11.2e} {row['turns']:6d} {dt_change:10.1e}")
    for r in runs:
        print(f"dt = {r['dt_days']} d: {r['n_steps']} steps in {r['wall_seconds']:.1f} s, "
              f"max |dE/E| = {r['max_rel_energy_error']:.2e}")
    save_json("stage3_solar_system", {
        "epoch_jd_tdb": fine["epoch_jd"], "years_simulated": YEARS, "integrator": "leapfrog",
        "runs": [{k: v for k, v in r.items() if k not in ("traj", "periods")} for r in runs],
        "planets": rows, "max_abs_rel_error_vs_sidereal": max(abs(r["rel_error_vs_sidereal"]) for r in rows)})

    traj = fine["traj"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, name in enumerate(BODIES[1:], start=1):
        rel = traj.pos[:, i] - traj.pos[:, 0]
        ax.plot(rel[:, 0], rel[:, 1], lw=0.5, label=name)
    ax.plot(0, 0, "o", color="gold", ms=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU] (ecliptic J2000, heliocentric)")
    ax.set_ylabel("y [AU]")
    ax.set_title(f"{YEARS:.0f} years from JPL Horizons initial conditions")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "stage3_orbits.png", dpi=150)


if __name__ == "__main__":
    main()
