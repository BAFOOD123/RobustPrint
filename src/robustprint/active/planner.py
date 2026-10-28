"""Turn an acquisition strategy into a concrete, printable list of next experiments."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from robustprint import space
from robustprint.active.acquisition import STRATEGIES, GPPosterior, experiment_cost, greedy_select
from robustprint.models.gp import GPRegressor
from robustprint.optimize.robust import safe_bounds
from robustprint.physics.composition import FloatArray
from robustprint.physics.materials import DEFAULT_LIBRARY, MaterialLibrary
from robustprint.physics.simulator import Feedstock
from robustprint.uncertainty.propagation import ModelPredictor, sample_feedstock


def candidate_pool(
    feedstock: Feedstock, n: int, rng: np.random.Generator, library: MaterialLibrary = DEFAULT_LIBRARY
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Candidate (composition, moisture, settings) triples.

    Compositions and moisture are draws from the feedstock distribution (what the lab can
    obtain or blend); settings are a scrambled Sobol design inside the safe bounds.
    """
    fs = sample_feedstock(feedstock, n, rng)
    low, high = safe_bounds(feedstock, library)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The balance properties of Sobol")
        u = qmc.Sobol(space.N_PARAMS, scramble=True, seed=int(rng.integers(2**31 - 1))).random(n)
    return fs.composition, fs.moisture, low + u * (high - low)


def select_experiments(
    predictor: ModelPredictor,
    feedstock: Feedstock,
    n: int,
    strategy: str,
    rng: np.random.Generator,
    recommended: FloatArray | None = None,
    n_candidates: int = 512,
    n_reference: int = 32,
    n_replicates: int = 3,
    library: MaterialLibrary = DEFAULT_LIBRARY,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Choose ``n`` experiments; returns (composition, moisture, settings, acquisition score)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose one of {STRATEGIES}.")
    comp, moist, params = candidate_pool(feedstock, n_candidates, rng, library)
    if strategy in ("random", "sobol"):
        if strategy == "random":
            low, high = safe_bounds(feedstock, library)
            params = low + rng.random(params.shape) * (high - low)
        idx = np.arange(n)
        return comp[idx], moist[idx], params[idx], np.full(n, np.nan)
    model = predictor.strength_model
    if not isinstance(model, GPRegressor):
        raise TypeError("Model-based acquisition requires a GP-family strength model.")
    feats = predictor.features
    X = feats.transform(comp, params, moist)
    p_success = predictor.success_model.predict_proba(X)
    reference = None
    if strategy != "max_variance":
        if recommended is None:
            raise ValueError("IVR strategies need the currently recommended setting.")
        ref = sample_feedstock(feedstock, n_reference, rng)
        reference = feats.transform(ref.composition, np.tile(recommended, (n_reference, 1)), ref.moisture)
    sel = greedy_select(GPPosterior(model), X, p_success, n, strategy, reference, experiment_cost(params), n_replicates)
    idx = np.array(sel.indices)
    return comp[idx], moist[idx], params[idx], np.array(sel.scores)


def experiments_table(
    predictor: ModelPredictor,
    composition: FloatArray,
    moisture: FloatArray,
    params: FloatArray,
    scores: FloatArray,
    strategy: str,
    library: MaterialLibrary = DEFAULT_LIBRARY,
) -> pd.DataFrame:
    """Human-readable next-experiments table with predictions and costs."""
    pred, p = predictor.predict(composition, moisture, params)
    z = norm.ppf(0.95)
    rows: dict[str, object] = {"rank": np.arange(1, len(params) + 1), "strategy": strategy}
    for k, name in enumerate(library.names):
        rows[f"target_frac_{name}"] = np.round(composition[:, k], 3)
    for k, name in enumerate(space.PARAM_NAMES):
        rows[name] = np.round(params[:, k], 3 if name == "layer_height_mm" else 1)
    rows["pred_strength_median_MPa"] = np.round(np.exp(pred.mean), 2)
    rows["pred_strength_90lo_MPa"] = np.round(np.exp(pred.mean - z * pred.std), 2)
    rows["pred_strength_90hi_MPa"] = np.round(np.exp(pred.mean + z * pred.std), 2)
    rows["pred_p_success"] = np.round(p, 3)
    rows["relative_cost"] = np.round(experiment_cost(params), 2)
    rows["acquisition_score"] = scores
    return pd.DataFrame(rows)


def table_to_markdown(table: pd.DataFrame) -> str:
    """Compact Markdown rendering (no external dependency)."""
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
