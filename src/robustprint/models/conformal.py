"""Split conformal prediction wrappers (Lei et al., 2018; Romano et al., 2019).

Calibration rows are split off **by batch**, so calibration and test batches are
exchangeable at the batch level. Rows within a batch share a batch effect, so the
finite-sample guarantee holds for batches rather than individual rows; we report the
empirical row-level coverage.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from robustprint.models.base import ProbabilisticRegressor
from robustprint.models.baselines import GBMQuantile
from robustprint.physics.composition import FloatArray


def conformal_quantile(scores: FloatArray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score, or +inf if that exceeds n."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def group_split(groups: np.ndarray, calib_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (proper, calibration) that keep every group on one side."""
    unique = np.unique(groups)
    n_cal = max(1, round(calib_fraction * len(unique)))
    cal_groups = rng.choice(unique, size=n_cal, replace=False)
    calib = np.isin(groups, cal_groups)
    return ~calib, calib


class NormalisedConformal:
    """Split conformal with scores |y - mu| / sigma around any probabilistic regressor."""

    def __init__(
        self,
        factory: Callable[[], ProbabilisticRegressor],
        alpha: float = 0.1,
        calib_fraction: float = 0.25,
        random_state: int = 0,
    ) -> None:
        self.factory = factory
        self.alpha = alpha
        self.calib_fraction = calib_fraction
        self.random_state = random_state

    @property
    def name(self) -> str:
        """Name of the wrapped model with a conformal suffix."""
        return f"{self.factory().name}+conformal"

    def fit(self, X: FloatArray, y: FloatArray, groups: np.ndarray) -> NormalisedConformal:
        """Fit the base model on the proper split and calibrate on held-out batches."""
        proper, calib = group_split(groups, self.calib_fraction, np.random.default_rng(self.random_state))
        self.model_ = self.factory().fit(X[proper], y[proper])
        pred = self.model_.predict(X[calib])
        scores = np.abs(y[calib] - pred.mean) / pred.std
        self.q_ = conformal_quantile(scores, self.alpha)
        return self

    def predict_interval(self, X: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Conformal (1 - alpha) prediction interval."""
        pred = self.model_.predict(X)
        half = self.q_ * pred.std
        return pred.mean - half, pred.mean + half


class ConformalizedQuantileRegression:
    """CQR on gradient-boosted quantile regression (Romano et al., 2019)."""

    name = "gbm_quantile+cqr"

    def __init__(self, alpha: float = 0.1, calib_fraction: float = 0.25, random_state: int = 0) -> None:
        self.alpha = alpha
        self.calib_fraction = calib_fraction
        self.random_state = random_state

    def fit(self, X: FloatArray, y: FloatArray, groups: np.ndarray) -> ConformalizedQuantileRegression:
        """Fit quantile models at alpha/2 and 1 - alpha/2, then calibrate."""
        proper, calib = group_split(groups, self.calib_fraction, np.random.default_rng(self.random_state))
        self.model_ = GBMQuantile(quantiles=(self.alpha / 2, 1 - self.alpha / 2), random_state=self.random_state)
        self.model_.fit(X[proper], y[proper])
        lo, hi = self.model_.predict_quantiles(X[calib])
        scores = np.maximum(lo - y[calib], y[calib] - hi)
        self.q_ = conformal_quantile(scores, self.alpha)
        return self

    def predict_interval(self, X: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Calibrated quantile interval."""
        lo, hi = self.model_.predict_quantiles(X)
        return lo - self.q_, hi + self.q_
