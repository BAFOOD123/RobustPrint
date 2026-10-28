"""Optional loader for the public Kaggle "3D Printer Dataset for Mechanical Engineers".

The dataset's licence is listed as "Unknown" on Kaggle, so it is **not** redistributed
with this repository. :func:`download_kaggle_3dprinter` fetches it into the git-ignored
``data/external/`` folder and checks its SHA-256 digest. It contains PLA and ABS only and
therefore says nothing about recycled composition effects; we use it only to check that
model families recover known parameter trends and as a small secondary benchmark.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/afumetto/3dprinter"
KAGGLE_PAGE = "https://www.kaggle.com/datasets/afumetto/3dprinter"
KAGGLE_SHA256 = "b98f601f714f2744dff18f5d1521288306b908b9da7fff3a42ae3d136a916e2d"
DEFAULT_PATH = Path("data/external/kaggle_3dprinter.csv")

RENAME = {
    "layer_height": "layer_height_mm",
    "wall_thickness": "wall_thickness_mm",
    "infill_density": "infill_pct",
    "infill_pattern": "infill_pattern",
    "nozzle_temperature": "nozzle_temp_C",
    "bed_temperature": "bed_temp_C",
    "print_speed": "print_speed_mm_s",
    "material": "material",
    "fan_speed": "cooling_fan_pct",
    "roughness": "roughness_um",
    "tension_strenght": "tensile_strength_MPa",
    "elongation": "elongation_pct",
}


def sha256_bytes(payload: bytes) -> str:
    """Hex SHA-256 digest of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def download_kaggle_3dprinter(path: Path = DEFAULT_PATH, timeout: float = 60.0) -> Path:
    """Download the dataset CSV to ``path`` and verify its checksum."""
    with urllib.request.urlopen(KAGGLE_URL, timeout=timeout) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        payload = zf.read("data.csv")
    digest = sha256_bytes(payload)
    if digest != KAGGLE_SHA256:
        raise ValueError(f"Checksum mismatch for Kaggle data: {digest} != {KAGGLE_SHA256}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def load_kaggle_3dprinter(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the dataset and rename columns to RobustPrint units and names."""
    frame = pd.read_csv(path)
    missing = set(RENAME) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks expected columns: {sorted(missing)}")
    frame = frame.rename(columns=RENAME)
    frame["material"] = frame["material"].str.strip().str.lower()
    frame["infill_pattern"] = frame["infill_pattern"].str.strip().str.lower()
    return frame
