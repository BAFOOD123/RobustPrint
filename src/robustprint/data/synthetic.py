"""Synthetic dataset generation in the experimental CSV schema.

Every generated row carries ``data_origin = "synthetic"`` and a source description that
starts with "SYNTHETIC", so synthetic data can never be mistaken for measurements.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import qmc

from robustprint import space
from robustprint.data.schema import DEFAULT_SCHEMA, Schema
from robustprint.physics.composition import FloatArray
from robustprint.physics.simulator import (
    DEFAULT_CONFIG,
    Batches,
    Feedstock,
    SimulatorConfig,
    realise,
    sample_batches,
)

DESIGNS = ("sobol", "lhs", "random")


def design_points(n: int, rng: np.random.Generator, design: str = "sobol") -> FloatArray:
    """Space-filling or random points in the unit hypercube of print parameters."""
    seed = int(rng.integers(2**31 - 1))
    if design == "sobol":
        m = int(np.ceil(np.log2(max(n, 2))))
        pts = qmc.Sobol(space.N_PARAMS, scramble=True, seed=seed).random_base2(m)[:n]
    elif design == "lhs":
        pts = qmc.LatinHypercube(space.N_PARAMS, seed=seed).random(n)
    elif design == "random":
        pts = rng.random((n, space.N_PARAMS))
    else:
        raise ValueError(f"Unknown design {design!r}; choose one of {DESIGNS}.")
    return np.asarray(pts, dtype=float)


def experiments_frame(
    batches: Batches,
    batch_index: np.ndarray,
    params: FloatArray,
    rng: np.random.Generator,
    n_replicates: int = 3,
    cfg: SimulatorConfig = DEFAULT_CONFIG,
    schema: Schema = DEFAULT_SCHEMA,
    measurement_concentration: float = 300.0,
    start_date: dt.date = dt.date(2026, 1, 5),
    id_prefix: str = "E",
) -> pd.DataFrame:
    """Simulate print jobs and return them as a schema-conformant DataFrame.

    Each row is one print job of ``n_replicates`` tensile specimens from one batch. The job
    succeeds with the simulator's success probability; if it succeeds the reported strength
    is the mean of ``n_replicates`` noisy specimens and ``tensile_strength_sd_MPa`` their
    sample standard deviation.
    """
    n = len(batch_index)
    comp = batches.true_composition[batch_index]
    moisture = batches.moisture_pct[batch_index]
    effect = batches.batch_effect[batch_index]
    reps = [realise(comp, params, moisture, effect, rng, cfg) for _ in range(n_replicates)]
    job_success = reps[0]["success"]
    strengths = np.stack([r["strength_MPa"] for r in reps])
    if n_replicates == 1:
        strength_sd = np.full(n, np.nan)
    else:
        spread = np.std(np.nan_to_num(strengths, nan=0.0), axis=0, ddof=1)
        strength_sd = np.where(job_success, spread, np.nan)

    def _mean(key: str) -> FloatArray:
        stacked = np.stack([np.nan_to_num(r[key], nan=0.0) for r in reps])
        return np.where(job_success, stacked.mean(axis=0), np.nan)

    data: dict[str, object] = {
        "experiment_id": [f"{id_prefix}{i:05d}" for i in range(n)],
        "data_origin": "synthetic",
        "batch_id": [batches.batch_id[j] for j in batch_index],
        "date": [(start_date + dt.timedelta(days=int(j))).isoformat() for j in batch_index],
        "source_description": "SYNTHETIC: simulated ISS-like mixed waste (robustprint simulator)",
    }
    reported = batches.reported_composition[batch_index]
    for k, col in enumerate(schema.fraction_columns):
        data[col] = np.round(reported[:, k], 6)
    data["composition_concentration"] = measurement_concentration
    data["moisture_pct"] = np.round(moisture, 4)
    for k, name in enumerate(space.PARAM_NAMES):
        data[name] = np.round(params[:, k], 4)
    data["tensile_strength_MPa"] = np.round(_mean("strength_MPa"), 4)
    data["tensile_strength_sd_MPa"] = np.round(strength_sd, 4)
    data["youngs_modulus_GPa"] = np.round(_mean("modulus_GPa"), 5)
    data["elongation_pct"] = np.round(_mean("elongation_pct"), 4)
    data["print_success"] = job_success
    data["n_replicates"] = n_replicates
    data["defect_notes"] = np.where(job_success, "", "simulated print failure")
    frame = pd.DataFrame(data)
    return _renormalise_rounded_fractions(frame, schema)


def _renormalise_rounded_fractions(frame: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Absorb rounding error into the largest fraction so each row sums to exactly 1."""
    cols = schema.fraction_columns
    values = frame[cols].to_numpy(dtype=float, copy=True)
    largest = values.argmax(axis=1)
    values[np.arange(len(values)), largest] += 1.0 - values.sum(axis=1)
    frame[cols] = np.round(values, 6)
    return frame


def generate_dataset(
    feedstock: Feedstock,
    n_batches: int,
    settings_per_batch: int,
    rng: np.random.Generator,
    n_replicates: int = 3,
    design: str = "sobol",
    cfg: SimulatorConfig = DEFAULT_CONFIG,
    measurement_concentration: float = 300.0,
    batch_prefix: str = "B",
    id_prefix: str = "E",
) -> tuple[pd.DataFrame, Batches]:
    """Generate a full synthetic campaign: batches x settings x replicates.

    Print settings follow one space-filling design across the whole campaign, assigned to
    batches in order, which mimics a lab that works through a design while new feedstock
    batches arrive.
    """
    batches = sample_batches(feedstock, n_batches, rng, measurement_concentration, cfg, prefix=batch_prefix)
    n = n_batches * settings_per_batch
    params = space.from_unit(design_points(n, rng, design))
    batch_index = np.repeat(np.arange(n_batches), settings_per_batch)
    frame = experiments_frame(
        batches,
        batch_index,
        params,
        rng,
        n_replicates,
        cfg,
        measurement_concentration=measurement_concentration,
        id_prefix=id_prefix,
    )
    return frame, batches
