"""A small starter experimental design a lab can run before any model exists."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from robustprint import space

#: Settings varied in the starter design: the four that govern interlayer bonding and
#: extrusion ([SUN-2008]); the rest are held at fixed, conventional values.
STARTER_FACTORS: dict[str, tuple[float, float, float]] = {
    "nozzle_temp_C": (240.0, 290.0, 5.0),
    "print_speed_mm_s": (15.0, 60.0, 5.0),
    "layer_height_mm": (0.12, 0.30, 0.02),
    "cooling_fan_pct": (0.0, 60.0, 10.0),
}
STARTER_FIXED: dict[str, float] = {
    "bed_temp_C": 100.0,
    "infill_pct": 100.0,
    "raster_angle_deg": 0.0,
    "flow_rate_pct": 100.0,
}


def _maximin_lhs(n: int, d: int, seed: int, tries: int = 200) -> np.ndarray:
    """Latin hypercube with the largest minimum pairwise distance among ``tries`` draws."""
    best, best_score = None, -1.0
    for t in range(tries):
        pts = qmc.LatinHypercube(d, seed=seed + t).random(n)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        score = float(dist[np.triu_indices(n, 1)].min())
        if score > best_score:
            best, best_score = pts, score
    assert best is not None
    return best


def starter_design(n_settings: int = 12, repeat_settings: int = 4, n_batches: int = 3, replicates: int = 3,
                   seed: int = 0) -> pd.DataFrame:
    """Maximin LHS over four bonding-related factors, rounded to printer-friendly steps.

    Batch 1 gets all ``n_settings``; later batches repeat the ``repeat_settings`` settings
    closest to the design centre plus spread, so batch-to-batch variation can be estimated
    separately from setting effects.
    """
    u = _maximin_lhs(n_settings, len(STARTER_FACTORS), seed)
    rows = []
    for k, (name, (lo, hi, step)) in enumerate(STARTER_FACTORS.items()):
        values = lo + u[:, k] * (hi - lo)
        rows.append(np.round(values / step) * step)
    settings = np.column_stack(rows)
    centre_order = np.argsort(np.linalg.norm(u - 0.5, axis=1))
    repeats = centre_order[:repeat_settings]
    records = []
    for b in range(1, n_batches + 1):
        chosen = range(n_settings) if b == 1 else repeats
        for i in chosen:
            rec = {"batch": f"batch_{b}", "setting_id": f"S{i + 1:02d}", "n_replicates": replicates}
            rec.update({name: float(settings[i, k]) for k, name in enumerate(STARTER_FACTORS)})
            rec.update(STARTER_FIXED)
            records.append(rec)
    frame = pd.DataFrame(records)
    ordered = ["batch", "setting_id", *space.PARAM_NAMES, "n_replicates"]
    return frame[ordered]
