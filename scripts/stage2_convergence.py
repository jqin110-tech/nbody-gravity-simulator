"""Stage 2c: convergence order. Halve dt and see how much the error after one orbit shrinks.

Run: python -m scripts.stage2_convergence
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nbody.forces import make_direct_accel
from nbody.integrators import INTEGRATORS
from nbody.simulation import run
from nbody.twobody import kepler_period, relative_position, two_body_system
from scripts._common import RESULTS, save_json

M1, M2, A, E = 1.0, 1e-3, 1.0, 0.5
STEP_COUNTS = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
EXPECTED_ORDER = {"euler": 1, "leapfrog": 2, "rk4": 4}
# Each method's error only follows its asymptotic law in a window: Euler needs small steps
# before it gets there, and RK4 reaches floating-point round-off (~1e-12 AU) past ~4000 steps.
# The order is fitted inside these windows; every point is still shown on the plot.
FIT_WINDOW = {"euler": (4000, 32000), "leapfrog": (250, 32000), "rk4": (250, 4000)}


def position_error(name: str, n_steps: int) -> float:
    """Distance (AU) between simulated and exact relative position after one full period."""
    system = two_body_system(M1, M2, A, E)
    T = kepler_period(A, M1 + M2)
    traj = run(system, INTEGRATORS[name], make_direct_accel(system.mass), T / n_steps, n_steps,
               record_every=n_steps)
    rel = traj.pos[-1, 1] - traj.pos[-1, 0]
    return float(np.linalg.norm(rel - relative_position(T, M1 + M2, A, E)))


def fitted_order(steps: list[int], errors: list[float]) -> float:
    """Slope p of log(error) against log(dt) by least squares, so error ~ dt^p."""
    return float(-np.polyfit(np.log(steps), np.log(errors), 1)[0])


def main() -> None:
    T = kepler_period(A, M1 + M2)
    out = {}
    for name in INTEGRATORS:
        errors = [position_error(name, n) for n in STEP_COUNTS]
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        lo, hi = FIT_WINDOW[name]
        window = [(n, err) for n, err in zip(STEP_COUNTS, errors) if lo <= n <= hi]
        order = fitted_order([n for n, _ in window], [err for _, err in window])
        out[name] = {"errors": errors, "ratios_when_halving_dt": ratios, "fit_window_steps": [lo, hi],
                     "fitted_order": order, "expected_order": EXPECTED_ORDER[name],
                     "expected_ratio": 2 ** EXPECTED_ORDER[name]}
        print(f"{name:9s} fitted order {order:.3f} (expected {EXPECTED_ORDER[name]}); "
              f"error ratios when halving dt: " + ", ".join(f"{r:.2f}" for r in ratios))
    save_json("stage2_convergence", {"orbit": {"m1": M1, "m2": M2, "a": A, "e": E},
                                     "step_counts_per_orbit": STEP_COUNTS, "results": out})

    fig, ax = plt.subplots(figsize=(6, 4.5))
    dts = T / np.array(STEP_COUNTS)
    for name, r in out.items():
        ax.loglog(dts, r["errors"], "o-", label=f"{name} (fitted order {r['fitted_order']:.2f})")
    ax.set_xlabel("time step dt [yr]")
    ax.set_ylabel("position error after one orbit [AU]")
    ax.set_title(f"Convergence, two-body orbit with e = {E}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "stage2_convergence.png", dpi=150)


if __name__ == "__main__":
    main()
