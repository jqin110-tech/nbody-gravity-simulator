"""Helpers shared by the result scripts: where results go and how they are saved."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def save_json(name: str, data: dict) -> Path:
    """Write `data` to results/<name>.json with a timestamp and machine description."""
    RESULTS.mkdir(exist_ok=True)
    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, NumPy {np.__version__}",
        **data,
    }
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path
