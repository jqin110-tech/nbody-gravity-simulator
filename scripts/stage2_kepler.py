"""Stage 2a: measure two-body orbital periods and compare them with Kepler's third law.

Run: python -m scripts.stage2_kepler
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nbody.forces import make_direct_accel
from nbody.integrators import leapfrog_step
from nbody.twobody import kepler_period, measured_period, two_body_system
from nbody.simulation import run
from scripts._common import RESULTS, save_json

# (semi-major axis in AU, central mass in M_sun, orbiting mass in M_sun, eccentricity).
# These are test cases, not real bodies; they span a factor 20 in a and 10 in mass.
CASES = [
    (0.5, 1.0, 1e-3, 0.0),
    (1.0, 1.0, 1e-3, 0.3),
    (2.0, 1.0, 1e-3, 0.6),
    (5.0, 1.0, 1e-3, 0.1),
    (10.0, 1.0, 1e-3, 0.2),
    (1.0, 0.5, 0.5, 0.4),
    (3.0, 2.0, 0.1, 0.0),
    (1.0, 10.0, 1.0, 0.5),
]
STEPS_PER_ORBIT = 20_000


def measure(a: float, m1: float, m2: float, e: float) -> dict:
    """Simulate a bit more than one orbit with leapfrog and measure its period."""
    system = two_body_system(m1, m2, a, e)
    T = kepler_period(a, m1 + m2)
    dt = T / STEPS_PER_ORBIT
    traj = run(system, leapfrog_step, make_direct_accel(system.mass), dt, int(1.05 * STEPS_PER_ORBIT))
    rel = traj.pos[:, 1] - traj.pos[:, 0]
    T_sim = measured_period(traj.times, rel)
    return {"a": a, "m1": m1, "m2": m2, "e": e, "T_kepler": T, "T_sim": T_sim, "rel_error": abs(T_sim - T) / T}


def main() -> None:
    rows = [measure(*case) for case in CASES]
    for r in rows:
        print(f"a={r['a']:5.1f} M={r['m1'] + r['m2']:6.3f} e={r['e']:.1f}  "
              f"T_kepler={r['T_kepler']:.8f}  T_sim={r['T_sim']:.8f}  rel.err={r['rel_error']:.2e}")
    worst = max(r["rel_error"] for r in rows)
    print(f"worst relative period error: {worst:.2e}")
    save_json("stage2_kepler", {"steps_per_orbit": STEPS_PER_ORBIT, "integrator": "leapfrog",
                                "cases": rows, "worst_rel_error": worst})

    x = np.array([r["a"] ** 3 / (r["m1"] + r["m2"]) for r in rows])
    y = np.array([r["T_sim"] ** 2 for r in rows])
    fig, ax = plt.subplots(figsize=(5, 4))
    line = np.geomspace(x.min(), x.max(), 50)
    ax.loglog(line, line, "k-", lw=1, label=r"Kepler: $T^2 = a^3/M$")
    ax.loglog(x, y, "o", color="tab:blue", label="simulated")
    ax.set_xlabel(r"$a^3 / M$  [AU$^3$ / M$_\odot$]")
    ax.set_ylabel(r"$T^2$  [yr$^2$]")
    ax.set_title(f"Kepler's third law (worst error {worst:.1e})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "stage2_kepler.png", dpi=150)


if __name__ == "__main__":
    main()
