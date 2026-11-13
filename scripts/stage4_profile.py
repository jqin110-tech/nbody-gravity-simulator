"""Stage 4a: profile the pure-Python Barnes-Hut force calculation to see where the time goes.

Run: python -m scripts.stage4_profile
"""

import cProfile
import io
import pstats
import time

import numpy as np

from nbody.barnes_hut import bh_accel, build
from nbody.initial_conditions import plummer_sphere
from scripts._common import RESULTS, save_json

N, THETA, EPS = 2000, 0.5, 0.01


def main() -> None:
    system = plummer_sphere(N, np.random.default_rng(42))
    profiler = cProfile.Profile()
    profiler.enable()
    bh_accel(system.pos, system.mass, THETA, EPS, G=1.0, backend="python")
    profiler.disable()
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).strip_dirs().sort_stats("cumulative").print_stats(12)
    report = text.getvalue()
    print(report)
    (RESULTS / "stage4_profile.txt").write_text(report)

    # Split the time between building the tree and walking it (without profiler overhead).
    t0 = time.perf_counter()
    build(system.pos, system.mass, backend="python")
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    bh_accel(system.pos, system.mass, THETA, EPS, G=1.0, backend="python")
    t_total = time.perf_counter() - t0
    print(f"N={N}: build {t_build:.3f} s of {t_total:.3f} s total ({100 * t_build / t_total:.1f}%); "
          f"tree walk {t_total - t_build:.3f} s ({100 * (t_total - t_build) / t_total:.1f}%)")
    save_json("stage4_profile", {"n": N, "theta": THETA, "backend": "python", "build_seconds": t_build,
                                 "total_seconds": t_total, "walk_fraction": (t_total - t_build) / t_total})


if __name__ == "__main__":
    main()
