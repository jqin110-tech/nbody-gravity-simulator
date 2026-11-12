"""Stage 2b: energy and angular-momentum conservation over many orbits for each integrator.

Run: python -m scripts.stage2_energy
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nbody.diagnostics import angular_momentum, total_energy
from nbody.forces import make_direct_accel
from nbody.integrators import INTEGRATORS
from nbody.simulation import run
from nbody.twobody import kepler_period, two_body_system
from scripts._common import RESULTS, save_json

# Test orbit: 1 M_sun star, 1e-3 M_sun planet, a = 1 AU, e = 0.5 (eccentric, to make it hard).
M1, M2, A, E = 1.0, 1e-3, 1.0, 0.5
N_ORBITS = 100
STEPS_PER_ORBIT = 500


def conservation_history(name: str, steps_per_orbit: int = STEPS_PER_ORBIT) -> dict:
    """Relative energy error and angular-momentum error recorded once every 1/10 orbit."""
    system = two_body_system(M1, M2, A, E)
    T = kepler_period(A, M1 + M2)
    traj = run(system, INTEGRATORS[name], make_direct_accel(system.mass), T / steps_per_orbit,
               N_ORBITS * steps_per_orbit, record_every=steps_per_orbit // 10)
    energy = np.array([total_energy(p, v, traj.mass) for p, v in zip(traj.pos, traj.vel)])
    L = np.array([angular_momentum(p, v, traj.mass)[2] for p, v in zip(traj.pos, traj.vel)])
    return {
        "orbits": (traj.times / T).tolist(),
        "dE_rel": (np.abs(energy - energy[0]) / abs(energy[0])).tolist(),
        "dL_rel": (np.abs(L - L[0]) / abs(L[0])).tolist(),
    }


def main() -> None:
    histories = {name: conservation_history(name) for name in INTEGRATORS}
    # RK4 makes 4 force evaluations per step and leapfrog makes 1, so the fair comparison
    # at equal cost gives leapfrog 4x as many steps.
    histories["leapfrog, 4x steps (same cost as rk4)"] = conservation_history("leapfrog", 4 * STEPS_PER_ORBIT)
    summary = {}
    for name, h in histories.items():
        dE, dL = np.array(h["dE_rel"]), np.array(h["dL_rel"])
        first, last = dE[: len(dE) // 10].max(), dE[-len(dE) // 10 :].max()
        summary[name] = {"max_dE_rel": float(dE.max()), "final_dE_rel": float(dE[-1]),
                         "max_dE_first_10pct": float(first), "max_dE_last_10pct": float(last),
                         "max_dL_rel": float(dL.max())}
        print(f"{name:38s} max|dE/E|={dE.max():.2e}  final|dE/E|={dE[-1]:.2e}  "
              f"(first 10%: {first:.2e}, last 10%: {last:.2e})  max|dL/L|={dL.max():.2e}")
    save_json("stage2_energy", {"orbit": {"m1": M1, "m2": M2, "a": A, "e": E},
                                "n_orbits": N_ORBITS, "steps_per_orbit": STEPS_PER_ORBIT,
                                "summary": summary})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for name, h in histories.items():
        axes[0].semilogy(h["orbits"][1:], h["dE_rel"][1:], label=name, lw=1)
        axes[1].semilogy(h["orbits"][1:], np.maximum(h["dL_rel"][1:], 1e-17), label=name, lw=1)
    axes[0].set_title(f"Energy error, e = {E}, {STEPS_PER_ORBIT} steps/orbit")
    axes[0].set_ylabel(r"$|E - E_0| / |E_0|$")
    axes[1].set_title("Angular momentum error")
    axes[1].set_ylabel(r"$|L - L_0| / |L_0|$")
    for ax in axes:
        ax.set_xlabel("orbits")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RESULTS / "stage2_energy.png", dpi=150)


if __name__ == "__main__":
    main()
