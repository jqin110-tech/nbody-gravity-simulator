"""Stage 5b: turn the saved galaxy snapshots into a video and into frames for the web viewer.

Run after stage5_galaxy:  python -m scripts.stage5_render [--quick]
Writes results/stage5_galaxy.mp4, results/stage5_galaxy_frames.png (a strip of stills),
and viewer/data/{meta.json, frames.bin, groups.bin}.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts._common import RESULTS

VIEWER_DATA = Path(__file__).resolve().parent.parent / "viewer" / "data"
BOX = 6.0  # half-width of the square region shown, in N-body length units
PIXELS = 600
VIEWER_DISC, VIEWER_HALO = 12_000, 3_000  # how many disc and halo bodies the web viewer gets
VIEWER_FRAME_STEP = 2  # the viewer gets every 2nd snapshot to keep the download small
DISC_COLOURS = np.array([[1.0, 0.62, 0.3], [0.4, 0.65, 1.0]])  # galaxy 1 orange, galaxy 2 blue


def density_image(xy: np.ndarray, galaxy: np.ndarray, is_disc: np.ndarray, peak: float) -> np.ndarray:
    """RGB image of disc particles: a 2D histogram per galaxy, log-scaled, coloured and added.

    `peak` is the pixel count drawn at full brightness; it is fixed for the whole video.
    """
    image = np.zeros((PIXELS, PIXELS, 3))
    edges = np.linspace(-BOX, BOX, PIXELS + 1)
    for g in (0, 1):
        sel = is_disc & (galaxy == g)
        counts, _, _ = np.histogram2d(xy[sel, 1], xy[sel, 0], bins=[edges, edges])
        image += np.log1p(counts)[:, :, None] * DISC_COLOURS[g]
    return np.clip(image / np.log1p(peak), 0.0, 1.0)[::-1]  # flip so +y is up


def brightness_peak(xy: np.ndarray, is_disc: np.ndarray) -> float:
    """99.5th percentile of non-empty pixel counts in the first frame: bright cores without washing out."""
    edges = np.linspace(-BOX, BOX, PIXELS + 1)
    counts, _, _ = np.histogram2d(xy[is_disc, 1], xy[is_disc, 0], bins=[edges, edges])
    return float(np.percentile(counts[counts > 0], 99.5))


def write_video(pos: np.ndarray, galaxy: np.ndarray, is_disc: np.ndarray, out: Path) -> None:
    """Render every snapshot to PNG and join them with ffmpeg at 25 frames per second."""
    peak = brightness_peak(pos[0, :, :2], is_disc)
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(len(pos)):
            plt.imsave(Path(tmp) / f"f{k:05d}.png", density_image(pos[k, :, :2], galaxy, is_disc, peak))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", f"{tmp}/f%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)


def write_stills(pos: np.ndarray, times: np.ndarray, galaxy: np.ndarray, is_disc: np.ndarray, out: Path) -> None:
    """Six evenly spaced snapshots side by side, for the README."""
    picks = np.linspace(0, len(pos) - 1, 6).astype(int)
    peak = brightness_peak(pos[0, :, :2], is_disc)
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.4))
    for ax, k in zip(axes, picks):
        ax.imshow(density_image(pos[k, :, :2], galaxy, is_disc, peak), extent=(-BOX, BOX, -BOX, BOX))
        ax.set_title(f"t = {times[k]:.1f}", color="white")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="black")


def export_viewer(pos: np.ndarray, times: np.ndarray, galaxy: np.ndarray, is_disc: np.ndarray) -> None:
    """A random subset of disc and halo particles, x and y quantised to int16 over [-BOX, BOX]."""
    pos, times = pos[::VIEWER_FRAME_STEP], times[::VIEWER_FRAME_STEP]
    rng = np.random.default_rng(0)
    disc, halo = np.flatnonzero(is_disc), np.flatnonzero(~is_disc)
    pick = np.sort(np.concatenate([rng.choice(disc, size=min(VIEWER_DISC, len(disc)), replace=False),
                                   rng.choice(halo, size=min(VIEWER_HALO, len(halo)), replace=False)]))
    xy = np.clip(pos[:, pick, :2] / BOX, -1.0, 1.0)
    VIEWER_DATA.mkdir(parents=True, exist_ok=True)
    np.round(xy * 32767).astype("<i2").tofile(VIEWER_DATA / "frames.bin")
    groups = (2 * galaxy[pick] + is_disc[pick]).astype(np.uint8)  # 0/2 halo, 1/3 disc
    groups.tofile(VIEWER_DATA / "groups.bin")
    meta = {"n_particles": len(pick), "n_frames": len(pos), "box": BOX, "times": [round(float(t), 3) for t in times],
            "n_bodies_simulated": int(pos.shape[1])}
    (VIEWER_DATA / "meta.json").write_text(json.dumps(meta) + "\n")


def main(name: str, for_viewer: bool) -> None:
    data = np.load(RESULTS / f"{name}.npz")
    pos, times, is_disc = data["pos"], data["times"], data["is_disc"]
    n = pos.shape[1]
    galaxy = (np.arange(n) >= n // 2).astype(int)  # first half of the bodies is galaxy 1
    if shutil.which("ffmpeg"):
        write_video(pos, galaxy, is_disc, RESULTS / f"{name}.mp4")
    else:
        print("ffmpeg not found: skipping the video")
    write_stills(pos, times, galaxy, is_disc, RESULTS / f"{name}_frames.png")
    if for_viewer:  # the quick test run must not overwrite the viewer's data
        export_viewer(pos, times, galaxy, is_disc)
    print(f"rendered {len(pos)} frames of {n} bodies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    quick = parser.parse_args().quick
    main("stage5_galaxy_quick" if quick else "stage5_galaxy", for_viewer=not quick)
