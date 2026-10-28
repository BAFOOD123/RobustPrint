"""E2: model comparison, learning curves, conformal coverage and success-classifier calibration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from robustprint.data.loaders import ExperimentArrays
from robustprint.eval.common import campaign, parallel_map, seed_for
from robustprint.models.base import Prediction
from robustprint.models.classifier import CLASSIFIERS, SuccessClassifier
from robustprint.models.conformal import ConformalizedQuantileRegression, NormalisedConformal
from robustprint.models.features import FeatureBuilder
from robustprint.models.registry import make_model
from robustprint.physics.simulator import ISS_NOMINAL, ISS_SHIFTED
from robustprint.uncertainty.metrics import (
    classification_report,
    gaussian_interval,
    interval_coverage,
    quantile_calibration,
    regression_report,
    reliability_curve,
)

CONFORMAL_MODELS = ("gp", "hybrid_gp", "deep_ensemble")
COVERAGE_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass(frozen=True)
class ModelTask:
    """One (seed, training size) cell of the comparison."""

    seed_index: int
    n_batches: int
    models: tuple[str, ...]
    folds: int
    test_batches: int
    settings_per_batch: int
    n_replicates: int
    base_seed: int


def _xy(arrays: ExperimentArrays, fb: FeatureBuilder) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = fb.transform(arrays.composition, arrays.params, arrays.moisture)
    ok = arrays.strength_rows
    return X, X[ok], np.log(arrays.strength[ok]), arrays.groups[ok]


def _coverage_rows(y: np.ndarray, pred: Prediction, model: str, split: str, base: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for level in COVERAGE_LEVELS:
        lo, hi = gaussian_interval(pred, level)
        cov, width = interval_coverage(y, lo, hi)
        rows.append({**base, "model": model, "split": split, "level": level, "coverage": cov, "width": width})
    return rows


def run_model_task(task: ModelTask) -> dict[str, list[dict[str, object]]]:
    """Evaluate every model on one simulated campaign."""
    fb = FeatureBuilder()
    s = seed_for(task.base_seed, 2, task.seed_index, task.n_batches)
    train = campaign(ISS_NOMINAL, task.n_batches, task.settings_per_batch, s, task.n_replicates)
    new = campaign(ISS_NOMINAL, task.test_batches, task.settings_per_batch, s + 1, task.n_replicates, prefix="N")
    ood = campaign(ISS_SHIFTED, task.test_batches, task.settings_per_batch, s + 2, task.n_replicates, prefix="S")
    X_all, X, y, g = _xy(train, fb)
    _, Xn, yn, _ = _xy(new, fb)
    _, Xo, yo, _ = _xy(ood, fb)
    base = {"seed": task.seed_index, "n_batches": task.n_batches, "n_train_rows": int(len(y))}
    metrics: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    calibration: list[dict[str, object]] = []
    n_folds = min(task.folds, len(np.unique(g)))
    for name in task.models:
        fold_pred_mean, fold_pred_var_e, fold_pred_var_a, fold_y = [], [], [], []
        for tr, te in GroupKFold(n_folds).split(X, y, g):
            model = make_model(name, fb, s).fit(X[tr], y[tr])
            p = model.predict(X[te])
            fold_pred_mean.append(p.mean)
            fold_pred_var_e.append(p.var_epistemic)
            fold_pred_var_a.append(p.var_aleatoric)
            fold_y.append(y[te])
        cv_pred = Prediction(np.concatenate(fold_pred_mean), np.concatenate(fold_pred_var_e), np.concatenate(fold_pred_var_a))
        cv_y = np.concatenate(fold_y)
        metrics.append({**base, "model": name, "split": "grouped_cv", **regression_report(cv_y, cv_pred)})
        coverage += _coverage_rows(cv_y, cv_pred, name, "grouped_cv", base)
        levels, observed = quantile_calibration(cv_y, cv_pred.mean, cv_pred.std)
        calibration += [{**base, "model": name, "level": lv, "observed": ob} for lv, ob in zip(levels, observed, strict=True)]
        full = make_model(name, fb, s).fit(X, y)
        for split, Xt, yt in (("new_batches", Xn, yn), ("shifted_ood", Xo, yo)):
            pt = full.predict(Xt)
            metrics.append({**base, "model": name, "split": split, **regression_report(yt, pt)})
            coverage += _coverage_rows(yt, pt, name, split, base)
    conformal: list[dict[str, object]] = []
    wrappers: list[object] = [
        NormalisedConformal(partial(make_model, m, fb, s), alpha=0.1, random_state=s)
        for m in task.models if m in CONFORMAL_MODELS
    ]
    wrappers.append(ConformalizedQuantileRegression(alpha=0.1, random_state=s))
    for w in wrappers:
        w.fit(X, y, g)  # type: ignore[attr-defined]
        for split, Xt, yt in (("new_batches", Xn, yn), ("shifted_ood", Xo, yo)):
            lo, hi = w.predict_interval(Xt)  # type: ignore[attr-defined]
            cov, width = interval_coverage(yt, lo, hi)
            conformal.append({**base, "model": w.name, "split": split, "coverage90": cov, "width90_log": width})  # type: ignore[attr-defined]
    classifier: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    groups_all = train.groups
    for kind in CLASSIFIERS:
        probs = np.zeros(len(X_all))
        for tr, te in GroupKFold(min(task.folds, len(np.unique(groups_all)))).split(X_all, train.success, groups_all):
            probs[te] = SuccessClassifier(kind, random_state=s).fit(X_all[tr], train.success[tr]).predict_proba(X_all[te])
        classifier.append({**base, "classifier": kind, "split": "grouped_cv", **classification_report(train.success, probs)})
        mean_p, freq, count = reliability_curve(train.success, probs)
        reliability += [{**base, "classifier": kind, "bin": b, "mean_p": mp, "freq": fr, "count": ct}
                        for b, (mp, fr, ct) in enumerate(zip(mean_p, freq, count, strict=True))]
        fitted = SuccessClassifier(kind, random_state=s).fit(X_all, train.success)
        for split, arr in (("new_batches", new), ("shifted_ood", ood)):
            Xt = fb.transform(arr.composition, arr.params, arr.moisture)
            classifier.append({**base, "classifier": kind, "split": split,
                               **classification_report(arr.success, fitted.predict_proba(Xt))})
    return {"metrics": metrics, "coverage": coverage, "calibration": calibration, "conformal": conformal,
            "classifier": classifier, "reliability": reliability}


def model_comparison(models: list[str], batch_sizes: list[int], seeds: int, folds: int, test_batches: int,
                     settings_per_batch: int, n_replicates: int, base_seed: int, n_jobs: int) -> dict[str, pd.DataFrame]:
    """Run E2 over all (seed, size) cells; returns one DataFrame per result table."""
    tasks = [ModelTask(i, n, tuple(models), folds, test_batches, settings_per_batch, n_replicates, base_seed)
             for n in batch_sizes for i in range(seeds)]
    parts = parallel_map(run_model_task, tasks, n_jobs)
    keys = parts[0].keys()
    return {k: pd.DataFrame([row for part in parts for row in part[k]]) for k in keys}
