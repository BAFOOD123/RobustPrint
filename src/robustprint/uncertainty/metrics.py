"""Accuracy, probabilistic-scoring and calibration metrics.

Regression metrics are computed on the model's target scale (log strength) unless noted;
RMSE and MAE are also reported in MPa after back-transformation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from robustprint.models.base import Prediction
from robustprint.physics.composition import FloatArray

QUANTILE_LEVELS = np.linspace(0.05, 0.95, 19)


def rmse(y: FloatArray, yhat: FloatArray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y: FloatArray, yhat: FloatArray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def gaussian_nll(y: FloatArray, mean: FloatArray, var: FloatArray) -> float:
    """Average negative log-likelihood of a Gaussian predictive."""
    var = np.maximum(var, 1e-12)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var))


def gaussian_crps(y: FloatArray, mean: FloatArray, std: FloatArray) -> float:
    """Closed-form CRPS of a Gaussian predictive (Gneiting & Raftery, 2007); lower is better."""
    std = np.maximum(std, 1e-12)
    z = (y - mean) / std
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def quantile_calibration(y: FloatArray, mean: FloatArray, std: FloatArray,
                         levels: FloatArray = QUANTILE_LEVELS) -> tuple[FloatArray, FloatArray]:
    """Observed frequency of y <= predicted p-quantile, for each level p (Kuleshov et al., 2018)."""
    pit = norm.cdf((y - mean) / np.maximum(std, 1e-12))
    observed = np.array([np.mean(pit <= p) for p in levels])
    return np.asarray(levels, dtype=float), observed


def quantile_calibration_error(y: FloatArray, mean: FloatArray, std: FloatArray) -> float:
    """Mean absolute gap between nominal and observed quantile levels (regression ECE)."""
    levels, observed = quantile_calibration(y, mean, std)
    return float(np.mean(np.abs(observed - levels)))


def interval_coverage(y: FloatArray, lower: FloatArray, upper: FloatArray) -> tuple[float, float]:
    """Empirical coverage and mean width of prediction intervals."""
    inside = (y >= lower) & (y <= upper)
    return float(np.mean(inside)), float(np.mean(upper - lower))


def gaussian_interval(pred: Prediction, level: float = 0.9) -> tuple[FloatArray, FloatArray]:
    """Central Gaussian prediction interval."""
    z = norm.ppf(0.5 + level / 2)
    return pred.mean - z * pred.std, pred.mean + z * pred.std


def regression_report(y_log: FloatArray, pred: Prediction, level: float = 0.9) -> dict[str, float]:
    """All regression metrics for log-strength targets and a Gaussian predictive."""
    lo, hi = gaussian_interval(pred, level)
    coverage, width = interval_coverage(y_log, lo, hi)
    y_mpa = np.exp(y_log)
    mean_mpa = np.exp(pred.mean + 0.5 * pred.var)
    return {
        "rmse_log": rmse(y_log, pred.mean),
        "rmse_MPa": rmse(y_mpa, mean_mpa),
        "mae_MPa": mae(y_mpa, np.exp(pred.mean)),
        "nll_log": gaussian_nll(y_log, pred.mean, pred.var),
        "crps_log": gaussian_crps(y_log, pred.mean, pred.std),
        "qce": quantile_calibration_error(y_log, pred.mean, pred.std),
        "coverage90": coverage,
        "width90_log": width,
    }


def brier_score(y: np.ndarray, p: FloatArray) -> float:
    """Mean squared error of probabilities."""
    return float(np.mean((np.asarray(p) - np.asarray(y, dtype=float)) ** 2))


def log_loss(y: np.ndarray, p: FloatArray) -> float:
    """Binary cross-entropy."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    yy = np.asarray(y, dtype=float)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


def reliability_curve(y: np.ndarray, p: FloatArray, n_bins: int = 10) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Mean predicted probability, observed frequency and count per equal-width bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    yy = np.asarray(y, dtype=float)
    mean_p = np.array([p[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
    freq = np.array([yy[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
    count = np.array([np.sum(idx == b) for b in range(n_bins)], dtype=float)
    return mean_p, freq, count


def expected_calibration_error(y: np.ndarray, p: FloatArray, n_bins: int = 10) -> float:
    """Count-weighted mean |observed frequency - mean probability| over bins."""
    mean_p, freq, count = reliability_curve(y, p, n_bins)
    ok = count > 0
    return float(np.sum(count[ok] * np.abs(freq[ok] - mean_p[ok])) / np.sum(count))


def classification_report(y: np.ndarray, p: FloatArray) -> dict[str, float]:
    """Brier score, log loss, ECE and accuracy at 0.5."""
    return {
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "accuracy": float(np.mean((p >= 0.5) == np.asarray(y, dtype=bool))),
        "base_rate": float(np.mean(y)),
    }
