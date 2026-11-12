"""Download the raw source data for Stage 3 and save it unmodified, with the retrieval time.

Uses only the Python standard library, so it runs anywhere with internet access:

    python fetch_horizons.py [output_dir]

Saves:
  horizons_<id>.txt   JPL Horizons state vectors (AU, AU/day) relative to the Solar System
                      barycentre, ecliptic J2000 frame, at EPOCH_JD (TDB). Ids: 10 = Sun,
                      1..8 = barycentres of the planetary systems (planet plus its moons).
  jpl_astro_par.html  JPL Solar System Dynamics "Astrodynamic Parameters" page (GM values).
  nssdc_factsheet.html NASA NSSDC Planetary Fact Sheet (its "Orbital Period" row is the tropical period).
  nssdc_notes.html    NASA NSSDC notes that define each fact-sheet column.
  nssdc_<planet>.html NASA NSSDC fact sheet for each planet (lists the sidereal orbit period).
  manifest.json       URLs, retrieval time (UTC) and epoch.
"""

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

EPOCH_JD = "2460676.5"  # 2025-01-01 00:00 TDB
BODY_IDS = ["10", "1", "2", "3", "4", "5", "6", "7", "8"]
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
PAGES = {
    "jpl_astro_par.html": "https://ssd.jpl.nasa.gov/astro_par.html",
    "nssdc_factsheet.html": "https://nssdc.gsfc.nasa.gov/planetary/factsheet/",
    "nssdc_notes.html": "https://nssdc.gsfc.nasa.gov/planetary/factsheet/planetfact_notes.html",
}
for _planet in ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]:
    PAGES[f"nssdc_{_planet}.html"] = f"https://nssdc.gsfc.nasa.gov/planetary/factsheet/{_planet}fact.html"


def horizons_url(body_id: str) -> str:
    """URL for one body's position and velocity at the epoch, as plain text."""
    params = {
        "format": "text", "COMMAND": f"'{body_id}'", "OBJ_DATA": "'YES'", "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@0'", "TLIST": f"'{EPOCH_JD}'", "TLIST_TYPE": "'JD'",
        "TIME_TYPE": "'TDB'", "OUT_UNITS": "'AU-D'", "VEC_TABLE": "'2'", "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'ICRF'", "CSV_FORMAT": "'YES'",
    }
    return HORIZONS_API + "?" + urllib.parse.urlencode(params)


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nbody-sim-data-fetch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for body_id in BODY_IDS:
        url = horizons_url(body_id)
        (out_dir / f"horizons_{body_id}.txt").write_bytes(download(url))
        sources[f"horizons_{body_id}.txt"] = url
        print("saved Horizons body", body_id)
    for name, url in PAGES.items():
        (out_dir / name).write_bytes(download(url))
        sources[name] = url
        print("saved", name)
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "epoch_jd_tdb": EPOCH_JD, "sources": sources}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("done:", out_dir.resolve())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "raw")
