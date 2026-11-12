"""Stage 4c: runtime of one force evaluation against N, for direct sum and Barnes-Hut.

Every method runs on ONE thread so the comparison is like for like; Numba's multi-threaded
Barnes-Hut is measured separately. Numba compile time is excluded (we warm up first).

Run: python -m scripts.stage4_benchmark
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np

from nbody.barnes_hut import bh_accel, direct_accel_compiled
from nbody.forces import direct_accel
from nbody.initial_conditions import plummer_sphere
from scripts._common import RESULTS, save_json

THETA, EPS, SEED = 0.5, 0.01, 7
N_VALUES = [2**k for k in range(6, 19)]  # 64 ... 262,144
STOP_AFTER_SECONDS = 10.0  # once one call takes this long, larger N is not attempted for that method

METHODS = {
    "direct, NumPy (vectorised)": lambda p, m: direct_accel(p, m, EPS, 1.0),
    "direct, pure Python loops": lambda p, m: direct_accel_compiled(p, m, EPS, 1.0, backend="python"),
    "Barnes-Hut, pure Python": lambda p, m: bh_accel(p, m, THETA, EPS, 1.0, backend="python"),
    "direct, Numba": lambda p, m: direct_accel_compiled(p, m, EPS, 1.0, backend="numba"),
    "Barnes-Hut, Numba": lambda p, m: bh_accel(p, m, THETA, EPS, 1.0, backend="numba"),
}


def time_call(fn, pos: np.ndarray, mass: np.ndarray) -> float:
    """Fastest of 3 to 7 calls (1 call if it is slow). The minimum is the least noisy estimate,
    because interruptions from the OS can only ever make a run slower."""
    times = []
    while len(times) < 7:
        start = time.perf_counter()
        fn(pos, mass)
        times.append(time.perf_counter() - start)
        if sum(times) > 2.0 and (len(times) >= 3 or times[0] > 2.0):
            break
    return float(min(times))


def sweep(fn) -> dict[int, float]:
    out = {}
    for n in N_VALUES:
        s = plummer_sphere(n, np.random.default_rng(SEED))
        out[n] = time_call(fn, s.pos, s.mass)
        print(f"  N={n:7d}  {out[n]:.4g} s", flush=True)
        if out[n] > STOP_AFTER_SECONDS:
            break
    return out


def crossover(direct: dict[int, float], tree: dict[int, float]) -> dict | None:
    """First measured N where the tree is faster, plus a log-log interpolation between the bracketing points."""
    common = sorted(set(direct) & set(tree))
    ratio = [np.log(tree[n] / direct[n]) for n in common]
    for k in range(1, len(common)):
        if ratio[k - 1] > 0 >= ratio[k]:
            lo, hi = common[k - 1], common[k]
            frac = ratio[k - 1] / (ratio[k - 1] - ratio[k])
            return {"last_n_direct_faster": lo, "first_n_tree_faster": hi,
                    "interpolated_n": float(np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))))}
    if common and ratio[0] <= 0:
        return {"tree_faster_at_all_measured_n": True}
    return None


def main() -> None:
    s = plummer_sphere(256, np.random.default_rng(0))
    for fn in METHODS.values():  # warm-up: triggers Numba compilation
        fn(s.pos, s.mass)
    numba.set_num_threads(1)
    results = {}
    for name, fn in METHODS.items():
        print(name, "(1 thread)")
        results[name] = sweep(fn)
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    threads = numba.get_num_threads()
    name_mt = f"Barnes-Hut, Numba ({threads} threads)"
    print(name_mt)
    results[name_mt] = sweep(METHODS["Barnes-Hut, Numba"])

    pairs = {
        "pure-Python Barnes-Hut vs NumPy direct": ("direct, NumPy (vectorised)", "Barnes-Hut, pure Python"),
        "pure-Python Barnes-Hut vs pure-Python direct": ("direct, pure Python loops", "Barnes-Hut, pure Python"),
        "Numba Barnes-Hut vs Numba direct": ("direct, Numba", "Barnes-Hut, Numba"),
    }
    crossovers = {label: crossover(results[d], results[t]) for label, (d, t) in pairs.items()}
    common = sorted(set(results["Barnes-Hut, pure Python"]) & set(results["Barnes-Hut, Numba"]))
    numba_speedup = {n: results["Barnes-Hut, pure Python"][n] / results["Barnes-Hut, Numba"][n] for n in common}
    for label, c in crossovers.items():
        print(f"crossover, {label}: {c}")
    print("Numba speed-up of Barnes-Hut (1 thread):", {n: round(v, 1) for n, v in numba_speedup.items()})
    save_json("stage4_benchmark", {
        "theta": THETA, "softening": EPS, "model": "Plummer sphere", "cpu_count": os.cpu_count(),
        "numba_version": numba.__version__, "seconds_per_force_evaluation": results,
        "crossovers": crossovers, "numba_speedup_barnes_hut_1_thread": numba_speedup})

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = ["o-", "^-", "s-", "o--", "s--", "D:"]
    for (name, data), style in zip(results.items(), styles):
        ax.loglog(list(data), list(data.values()), style, label=name, ms=4)
    ax.set_xlabel("number of bodies N")
    ax.set_ylabel("seconds per force evaluation")
    ax.set_title(rf"Force evaluation cost, Plummer sphere, $\theta$ = {THETA}")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "stage4_benchmark.png", dpi=150)


if __name__ == "__main__":
    main()
