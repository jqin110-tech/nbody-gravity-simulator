"""Stage 4b: Barnes-Hut force error and cost against the opening angle theta.

Run: python -m scripts.stage4_theta
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nbody.barnes_hut import bh_accel, direct_accel_compiled
from nbody.initial_conditions import plummer_sphere
from scripts._common import RESULTS, save_json

N, EPS, SEED = 20_000, 0.01, 42  # Plummer sphere in N-body units (G = M = scale = 1)
THETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]


def timed(fn, *args, **kwargs) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def main() -> None:
    s = plummer_sphere(N, np.random.default_rng(SEED))
    bh_accel(s.pos[:100], s.mass[:100], 0.5, EPS, 1.0)  # compile before timing
    exact, t_direct = timed(direct_accel_compiled, s.pos, s.mass, EPS, 1.0)
    exact_norm = np.linalg.norm(exact, axis=1)
    rows = []
    for theta in THETAS:
        approx, t = timed(bh_accel, s.pos, s.mass, theta, EPS, 1.0)
        err = np.linalg.norm(approx - exact, axis=1) / exact_norm
        rows.append({"theta": theta, "median_rel_error": float(np.median(err)),
                     "p99_rel_error": float(np.percentile(err, 99)), "max_rel_error": float(err.max()),
                     "seconds": t, "speedup_vs_direct": t_direct / t})
        r = rows[-1]
        print(f"theta={theta:.1f}  median err {r['median_rel_error']:.2e}  99th pct {r['p99_rel_error']:.2e}  "
              f"max {r['max_rel_error']:.2e}  time {t:.3f} s  ({r['speedup_vs_direct']:.1f}x faster than direct)")
    print(f"direct sum (Numba) for N={N}: {t_direct:.3f} s")
    save_json("stage4_theta", {"n": N, "softening": EPS, "model": "Plummer sphere, N-body units", "seed": SEED,
                               "backend": "numba (all threads)", "direct_seconds": t_direct, "rows": rows})

    th = [r["theta"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.semilogy(th, [r["median_rel_error"] for r in rows], "o-", label="median")
    ax1.semilogy(th, [r["p99_rel_error"] for r in rows], "s-", label="99th percentile")
    ax1.set_xlabel(r"opening angle $\theta$")
    ax1.set_ylabel("relative force error vs direct sum")
    ax1.set_title(f"Force error, Plummer sphere, N = {N:,}")
    ax1.legend()
    ax2.plot(th, [r["seconds"] for r in rows], "o-", label="Barnes-Hut")
    ax2.axhline(t_direct, color="k", ls="--", label="direct sum")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"opening angle $\theta$")
    ax2.set_ylabel("seconds per force evaluation")
    ax2.set_title("Cost (Numba, all threads)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "stage4_theta.png", dpi=150)


if __name__ == "__main__":
    main()
