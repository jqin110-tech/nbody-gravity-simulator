"""Load the real Sun and planets from the saved JPL Horizons and NASA files in data/raw/."""

import re
from pathlib import Path

import numpy as np

from nbody.constants import DAYS_PER_YEAR
from nbody.simulation import System

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BODIES = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
HORIZONS_IDS = ["10", "1", "2", "3", "4", "5", "6", "7", "8"]  # Sun, then planetary-system barycentres


def read_horizons_vector(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Parse one Horizons CSV vector table: returns (JD TDB, position in AU, velocity in AU/day)."""
    text = path.read_text()
    body = text.split("$$SOE")[1].split("$$EOE")[0].strip()
    fields = [f.strip() for f in body.splitlines()[0].split(",")]
    jd = float(fields[0])
    x, y, z, vx, vy, vz = (float(f) for f in fields[2:8])
    return jd, np.array([x, y, z]), np.array([vx, vy, vz])


def _page_text(path: Path) -> str:
    """Visible text of a saved HTML page: tags removed, whitespace collapsed."""
    text = re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8"))
    return re.sub(r"\s+", " ", text.replace("&nbsp;", " "))


def read_gm_values(path: Path) -> dict[str, float]:
    """GM in km^3/s^2 for the Sun and each planetary system, from JPL's Astrodynamic Parameters page (DE440)."""
    text = _page_text(path)
    sun = re.search(r"GM sun ([\d.]+) x 10 (\d+) m 3 s -2", text)
    gm = {"Sun": float(sun.group(1)) * 10 ** int(sun.group(2)) / 1e9}  # m^3 -> km^3
    for name in ["Mercury", "Venus", "Earth", "Moon"]:
        gm[name] = float(re.search(rf"\b{name} ([\d.]+)", text).group(1))
    for name in ["Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]:
        gm[name] = float(re.search(rf"{name} system ([\d.]+)", text).group(1))
    gm["Earth"] += gm.pop("Moon")  # Horizons body 3 is the Earth-Moon barycentre
    return gm


def read_reference_periods(raw: Path = RAW) -> dict[str, dict[str, float]]:
    """Sidereal and tropical orbit periods in days from each planet's NASA NSSDC fact sheet.

    Sidereal means one orbit relative to the fixed stars, which is what the simulation measures.
    Tropical means equinox to equinox; it is shorter because Earth's axis precesses, and it is
    the value NSSDC's summary table lists as "Orbital Period" (see data/raw/nssdc_notes.html).
    """
    periods = {}
    for name in BODIES[1:]:
        text = _page_text(raw / f"nssdc_{name.lower()}.html")
        sidereal = re.search(r"Sidereal orbit period \(days\) ([\d,.]+)", text).group(1)
        tropical = re.search(r"Tropical orbit period \*? ?\(days\) ([\d,.]+)", text).group(1)
        periods[name] = {"sidereal": float(sidereal.replace(",", "")), "tropical": float(tropical.replace(",", ""))}
    return periods


def load_solar_system(raw: Path = RAW) -> tuple[System, float]:
    """Sun and planetary-system barycentres at the Horizons epoch, in AU, AU/yr and M_sun. Returns (system, JD)."""
    gm = read_gm_values(raw / "jpl_astro_par.html")
    pos, vel, jds = [], [], []
    for horizons_id in HORIZONS_IDS:
        jd, p, v = read_horizons_vector(raw / f"horizons_{horizons_id}.txt")
        pos.append(p)
        vel.append(v * DAYS_PER_YEAR)  # AU/day -> AU/yr (our "year" is the Gaussian year)
        jds.append(jd)
    assert len(set(jds)) == 1, "all bodies must be at the same epoch"
    mass = np.array([gm[name] / gm["Sun"] for name in BODIES])
    return System(np.array(pos), np.array(vel), mass), jds[0]
