"""Turn validated experiment tables into model-ready arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma

from robustprint import space
from robustprint.data.schema import DEFAULT_SCHEMA, Schema
from robustprint.data.validation import ValidationReport, validate_csv
from robustprint.physics.composition import FloatArray, closure, replace_zeros
from robustprint.physics.simulator import Feedstock

#: Moisture assumed when a batch has no measurement (median of the default feedstock).
DEFAULT_MOISTURE_PCT = 0.3


@dataclass(frozen=True)
class ExperimentArrays:
    """Arrays extracted from an experiment table.

    Attributes:
        composition: (n, D) reported composition.
        params: (n, 8) print parameters.
        moisture: (n,) moisture, imputed with the median where missing.
        strength: (n,) mean tensile strength (NaN for failed jobs).
        strength_sd: (n,) specimen standard deviation (NaN where unknown).
        modulus: (n,) Young's modulus.
        elongation: (n,) elongation at break.
        success: (n,) print success flags.
        n_replicates: (n,) number of specimens per job.
        groups: (n,) batch identifiers for grouped cross-validation.
    """

    composition: FloatArray
    params: FloatArray
    moisture: FloatArray
    strength: FloatArray
    strength_sd: FloatArray
    modulus: FloatArray
    elongation: FloatArray
    success: np.ndarray
    n_replicates: np.ndarray
    groups: np.ndarray

    def __len__(self) -> int:
        return int(self.params.shape[0])

    def subset(self, mask: np.ndarray) -> ExperimentArrays:
        """Rows selected by a boolean mask or index array."""
        return ExperimentArrays(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    @property
    def strength_rows(self) -> np.ndarray:
        """Boolean mask of rows usable for strength regression."""
        return np.asarray(self.success & np.isfinite(self.strength))


def to_arrays(frame: pd.DataFrame, schema: Schema = DEFAULT_SCHEMA) -> ExperimentArrays:
    """Convert a validated frame to :class:`ExperimentArrays`."""
    comp = replace_zeros(frame[schema.fraction_columns].to_numpy(dtype=float))
    params = frame[space.PARAM_NAMES].to_numpy(dtype=float)
    if "moisture_pct" in frame.columns and frame["moisture_pct"].notna().any():
        moisture = frame["moisture_pct"].to_numpy(dtype=float)
        moisture = np.where(np.isfinite(moisture), moisture, np.nanmedian(moisture))
    else:
        moisture = np.full(len(frame), DEFAULT_MOISTURE_PCT)

    def _col(name: str) -> FloatArray:
        if name not in frame.columns:
            return np.full(len(frame), np.nan)
        return frame[name].to_numpy(dtype=float)

    return ExperimentArrays(
        composition=comp,
        params=params,
        moisture=moisture,
        strength=_col("tensile_strength_MPa"),
        strength_sd=_col("tensile_strength_sd_MPa"),
        modulus=_col("youngs_modulus_GPa"),
        elongation=_col("elongation_pct"),
        success=frame["print_success"].to_numpy(dtype=bool),
        n_replicates=frame["n_replicates"].to_numpy(dtype=int),
        groups=frame["batch_id"].astype(str).to_numpy(),
    )


def load_experiments(path: str, schema: Schema = DEFAULT_SCHEMA) -> tuple[ExperimentArrays, ValidationReport]:
    """Validate a CSV and convert it; raises ``ValueError`` listing all errors if invalid."""
    frame, report = validate_csv(path, schema)
    if not report.ok:
        raise ValueError(f"{path} failed validation:\n{report.summary()}")
    return to_arrays(frame, schema), report


def fit_dirichlet(compositions: FloatArray, max_iter: int = 200, tol: float = 1e-8) -> FloatArray:
    """Maximum-likelihood Dirichlet parameters (Minka's fixed-point iteration).

    Initialised by moment matching. Used to estimate the feedstock distribution from the
    batches observed so far.
    """
    x = replace_zeros(np.atleast_2d(compositions))
    if x.shape[0] < 2:
        raise ValueError("At least two batches are needed to estimate a feedstock distribution.")
    mean = x.mean(axis=0)
    var = x.var(axis=0, ddof=1)
    s = np.median(mean * (1 - mean) / np.maximum(var, 1e-12) - 1.0)
    alpha = np.maximum(mean * max(s, 1.0), 1e-3)
    log_bar = np.log(x).mean(axis=0)
    for _ in range(max_iter):
        target = digamma(alpha.sum()) + log_bar
        new = _inverse_digamma(target)
        if np.max(np.abs(new - alpha)) < tol:
            alpha = new
            break
        alpha = new
    return np.asarray(alpha, dtype=float)


def _inverse_digamma(y: FloatArray, n_iter: int = 8) -> FloatArray:
    """Newton inversion of the digamma function (Minka, 2000)."""
    x = np.where(y >= -2.22, np.exp(y) + 0.5, -1.0 / (y - digamma(1.0)))
    for _ in range(n_iter):
        x = x - (digamma(x) - y) / polygamma(1, x)
    return np.asarray(x, dtype=float)


def estimate_feedstock(arrays: ExperimentArrays, name: str = "estimated") -> Feedstock:
    """Feedstock distribution estimated from one composition per observed batch."""
    _, first = np.unique(arrays.groups, return_index=True)
    comps = arrays.composition[np.sort(first)]
    alpha = fit_dirichlet(comps)
    moist = arrays.moisture[np.sort(first)]
    log_m = np.log(np.maximum(moist, 1e-3))
    return Feedstock(
        name=name,
        mean=tuple(closure(alpha).tolist()),
        concentration=float(alpha.sum()),
        moisture_median_pct=float(np.exp(np.median(log_m))),
        moisture_log_sd=float(np.maximum(np.std(log_m, ddof=1), 1e-3)) if len(log_m) > 1 else 0.5,
    )
