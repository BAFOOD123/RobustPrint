"""Shared helpers for experiments: seeding, campaigns, parallel map, result files."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from robustprint.data.loaders import ExperimentArrays, to_arrays
from robustprint.data.synthetic import generate_dataset
from robustprint.physics.simulator import DEFAULT_CONFIG, Feedstock, SimulatorConfig

T = TypeVar("T")
R = TypeVar("R")

#: Label stamped on every synthetic result file and figure.
SYNTHETIC_LABEL = "SYNTHETIC DATA (RobustPrint simulator) - not experimental results"


def seed_for(*keys: int) -> int:
    """Deterministic 31-bit seed derived from integer keys."""
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1)[0] % (2**31 - 1))


def campaign(feedstock: Feedstock, n_batches: int, settings_per_batch: int, seed: int, n_replicates: int = 3,
             cfg: SimulatorConfig = DEFAULT_CONFIG, prefix: str = "B") -> ExperimentArrays:
    """Simulate a training campaign and return model-ready arrays."""
    frame, _ = generate_dataset(feedstock, n_batches, settings_per_batch, np.random.default_rng(seed),
                                n_replicates=n_replicates, cfg=cfg, batch_prefix=prefix)
    return to_arrays(frame)


def _init_worker() -> None:
    """Pin each worker to one thread (BLAS, OpenMP and PyTorch) to avoid oversubscription."""
    import torch
    from threadpoolctl import threadpool_limits

    os.environ["OMP_NUM_THREADS"] = "1"
    threadpool_limits(1)
    torch.set_num_threads(1)


def parallel_map(fn: Callable[[T], R], items: Iterable[T], n_jobs: int) -> list[R]:
    """Map ``fn`` over ``items`` preserving order; serial when ``n_jobs <= 1``."""
    items = list(items)
    if n_jobs <= 1 or len(items) <= 1:
        _init_worker()
        return [fn(item) for item in items]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx, initializer=_init_worker) as pool:
        return list(pool.map(fn, items))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.floating | float):
        f = float(obj)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with the synthetic label, rounding floats to 6 decimals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"label": SYNTHETIC_LABEL, **payload}
    path.write_text(json.dumps(_jsonable(body), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV whose first line is a comment with the synthetic label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {SYNTHETIC_LABEL}\n")
        frame.to_csv(fh, index=False, float_format="%.6g")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV written by :func:`write_csv`."""
    return pd.read_csv(path, comment="#")
