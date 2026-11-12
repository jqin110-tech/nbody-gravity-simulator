"""Print every number quoted in README.md, read straight from results/*.json.

Run: python -m scripts.summary
If you rerun the stage scripts, rerun this and compare its output with the README.
"""

import json

from scripts._common import RESULTS


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def stage2() -> None:
    kepler, energy, conv = load("stage2_kepler"), load("stage2_energy"), load("stage2_convergence")
    print("## Stage 2\n")
    print(f"Kepler: {len(kepler['cases'])} orbits, worst relative period error {kepler['worst_rel_error']:.2e}\n")
    print("| Integrator | max dE/E | max dE/E, first 10% | max dE/E, last 10% | max dL/L |")
    print("|---|---|---|---|---|")
    for name, s in energy["summary"].items():
        print(f"| {name} | {s['max_dE_rel']:.2e} | {s['max_dE_first_10pct']:.2e} | "
              f"{s['max_dE_last_10pct']:.2e} | {s['max_dL_rel']:.2e} |")
    print("\n| Integrator | Expected order | Fitted order | Error ratios when dt is halved |")
    print("|---|---|---|---|")
    for name, r in conv["results"].items():
        ratios = ", ".join(f"{x:.2f}" for x in r["ratios_when_halving_dt"])
        print(f"| {name} | {r['expected_order']} | {r['fitted_order']:.2f} | {ratios} |")


def stage3() -> None:
    d = load("stage3_solar_system")
    print("\n## Stage 3\n")
    print("| Planet | Simulated (days) | NASA sidereal (days) | Relative error | Orbits averaged |")
    print("|---|---|---|---|---|")
    for p in d["planets"]:
        print(f"| {p['planet']} | {p['simulated_days']:.3f} | {p['nasa_sidereal_days']:.3f} | "
              f"{p['rel_error_vs_sidereal']:+.1e} | {p['turns']} |")
    print(f"\nWorst |relative error|: {d['max_abs_rel_error_vs_sidereal']:.1e}")
    print(f"Largest change in any period when dt is halved: {max(p['rel_change_when_dt_halved'] for p in d['planets']):.1e}")
    for r in d["runs"]:
        print(f"dt = {r['dt_days']} d: {r['n_steps']:,} steps, {r['wall_seconds']:.0f} s, "
              f"max |dE/E| = {r['max_rel_energy_error']:.1e}")
    neptune = d["planets"][-1]
    print(f"Neptune against the tropical period ({neptune['nasa_tropical_days']} d): {neptune['rel_error_vs_tropical']:+.2e}")


def stage4() -> None:
    profile, theta, bench = load("stage4_profile"), load("stage4_theta"), load("stage4_benchmark")
    print("\n## Stage 4\n")
    print(f"Profile (pure Python, N = {profile['n']}): tree walk is {100 * profile['walk_fraction']:.1f}% of "
          f"{profile['total_seconds']:.2f} s\n")
    print(f"Force error, N = {theta['n']:,}; direct sum takes {theta['direct_seconds']:.2f} s\n")
    print("| theta | Median error | 99th percentile | Max | Seconds | Speed-up vs direct |")
    print("|---|---|---|---|---|---|")
    for r in theta["rows"]:
        print(f"| {r['theta']} | {r['median_rel_error']:.1e} | {r['p99_rel_error']:.1e} | {r['max_rel_error']:.1e} | "
              f"{r['seconds']:.3f} | {r['speedup_vs_direct']:.1f}x |")
    times = bench["seconds_per_force_evaluation"]
    all_n = sorted({int(n) for series in times.values() for n in series})
    print(f"\nSeconds per force evaluation (theta = {bench['theta']}, {bench['cpu_count']} CPUs available)\n")
    print("| N | " + " | ".join(times) + " |")
    print("|---|" + "---|" * len(times))
    for n in all_n:
        cells = [f"{series[str(n)]:.3g}" if str(n) in series else "-" for series in times.values()]
        print(f"| {n:,} | " + " | ".join(cells) + " |")
    print()
    for label, c in bench["crossovers"].items():
        print(f"Crossover, {label}: {c}")
    speedups = bench["numba_speedup_barnes_hut_1_thread"].values()
    print(f"Numba speed-up of Barnes-Hut on 1 thread: {min(speedups):.0f}x to {max(speedups):.0f}x")


def stage5() -> None:
    d = load("stage5_galaxy")
    print("\n## Stage 5\n")
    print(f"{d['n_bodies']:,} bodies ({d['n_disc']:,} in discs), {d['n_steps']:,} leapfrog steps, theta = {d['theta']}, "
          f"softening = {d['softening']}, dt = {d['dt']}")
    print(f"Wall time {d['wall_seconds'] / 60:.1f} min, {d['seconds_per_step']:.2f} s per step")
    print(f"Energy change |dE/E| = {d['energy_rel_change']:.1e}; angular momentum change "
          f"{d['angular_momentum_rel_change']:.1e}")
    meta = json.loads((RESULTS.parent / "viewer" / "data" / "meta.json").read_text())
    print(f"Viewer: {meta['n_particles']:,} bodies, {meta['n_frames']} frames; video: {d['n_steps'] // 10 + 1} frames")


if __name__ == "__main__":
    stage2()
    stage3()
    stage4()
    if (RESULTS / "stage5_galaxy.json").exists():
        stage5()
