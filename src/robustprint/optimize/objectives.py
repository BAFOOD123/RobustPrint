"""Robust objectives estimated by Monte Carlo with common random numbers.

For a candidate setting x, the predictor gives, for each of K feedstock draws c_k, a
Gaussian on log strength and a success probability p_k. We condition on success by
weighting draw k with p_k / sum_j p_j (strength is only measured on parts that print),
and expand each Gaussian with S fixed standard-normal draws. All candidates share the same
draws, so the Monte-Carlo objective is a deterministic function of x (sample-average
approximation) and differences between candidates are not swamped by sampling noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustprint.physics.composition import FloatArray
from robustprint.physics.simulator import Feedstock
from robustprint.uncertainty.propagation import FeedstockSamples, Predictor, sample_feedstock

OBJECTIVES = ("nominal", "expected", "quantile", "cvar", "chance")


@dataclass(frozen=True)
class ObjectiveSettings:
    """Risk levels and constraints shared by all objectives.

    Attributes:
        quantile_level: Level of the robust quantile objective (0.05 = 5th percentile).
        cvar_alpha: Tail fraction for CVaR (mean of the worst alpha fraction).
        p_min: Minimum predicted success probability (all objectives).
        chance_threshold_MPa: Strength requirement of the chance-constrained objective.
        chance_prob: Required P(strength >= threshold) for the chance constraint.
        penalty_MPa: Objective penalty per unit of constraint violation (probability).
    """

    quantile_level: float = 0.05
    cvar_alpha: float = 0.1
    p_min: float = 0.9
    chance_threshold_MPa: float = 5.0
    chance_prob: float = 0.9
    penalty_MPa: float = 100.0


def weighted_quantile(values: FloatArray, weights: FloatArray, q: float) -> FloatArray:
    """Row-wise weighted quantile of ``values`` (P, N) with weights (P, N)."""
    order = np.argsort(values, axis=1)
    v = np.take_along_axis(values, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    cw = np.cumsum(w, axis=1) / np.sum(w, axis=1, keepdims=True)
    idx = np.minimum((cw < q).sum(axis=1), values.shape[1] - 1)
    return np.asarray(v[np.arange(values.shape[0]), idx], dtype=float)


def weighted_cvar(values: FloatArray, weights: FloatArray, alpha: float) -> FloatArray:
    """Row-wise weighted lower-tail CVaR: mean of the lowest ``alpha`` probability mass."""
    order = np.argsort(values, axis=1)
    v = np.take_along_axis(values, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    w = w / np.sum(w, axis=1, keepdims=True)
    cw = np.cumsum(w, axis=1)
    prev = cw - w
    take = np.clip(alpha - prev, 0.0, None)
    take = np.minimum(take, w)
    return np.asarray(np.sum(take * v, axis=1) / alpha, dtype=float)


class RobustProblem:
    """Evaluates all robust statistics for batches of candidate settings."""

    def __init__(self, predictor: Predictor, feedstock: Feedstock, settings: ObjectiveSettings | None = None,
                 n_compositions: int = 64, n_noise: int = 16, seed: int = 0) -> None:
        self.predictor = predictor
        self.feedstock = feedstock
        self.settings = settings or ObjectiveSettings()
        rng = np.random.default_rng(seed)
        self.samples: FeedstockSamples = sample_feedstock(feedstock, n_compositions, rng)
        self.normals = rng.standard_normal((n_compositions, n_noise))
        self.nominal_composition = feedstock.mean_array()[None, :]
        self.nominal_moisture = np.array([feedstock.moisture_median_pct])

    def statistics(self, params: FloatArray) -> dict[str, FloatArray]:
        """Robust statistics for each row of ``params`` (P, 8)."""
        params = np.atleast_2d(params)
        P, K = params.shape[0], len(self.samples)
        comp = np.tile(self.samples.composition, (P, 1))
        moist = np.tile(self.samples.moisture, P)
        pred, p = self.predictor.predict(comp, moist, np.repeat(params, K, axis=0))
        mu = pred.mean.reshape(P, K)
        sd = pred.std.reshape(P, K)
        p = p.reshape(P, K)
        log_s = mu[:, :, None] + sd[:, :, None] * self.normals[None, :, :]
        strength = np.exp(log_s).reshape(P, -1)
        S = self.normals.shape[1]
        w = np.repeat(np.maximum(p, 1e-9), S, axis=1)
        s = self.settings
        nominal_pred, nominal_p = self.predictor.predict(
            np.repeat(self.nominal_composition, P, axis=0), np.repeat(self.nominal_moisture, P), params
        )
        return {
            "mean": np.sum(w * strength, axis=1) / np.sum(w, axis=1),
            "quantile": weighted_quantile(strength, w, s.quantile_level),
            "cvar": weighted_cvar(strength, w, s.cvar_alpha),
            "p_exceed": np.sum(w * (strength >= s.chance_threshold_MPa), axis=1) / np.sum(w, axis=1),
            "p_success": p.mean(axis=1),
            "nominal_mean": np.exp(nominal_pred.mean + 0.5 * nominal_pred.var),
            "nominal_p_success": nominal_p,
        }

    def score(self, objective: str, params: FloatArray) -> FloatArray:
        """Penalised objective value to maximise."""
        st = self.statistics(params)
        s = self.settings
        pen = s.penalty_MPa
        if objective == "nominal":
            return st["nominal_mean"] - pen * np.maximum(0.0, s.p_min - st["nominal_p_success"])
        success_pen = pen * np.maximum(0.0, s.p_min - st["p_success"])
        if objective == "expected":
            return st["mean"] - success_pen
        if objective == "quantile":
            return st["quantile"] - success_pen
        if objective == "cvar":
            return st["cvar"] - success_pen
        if objective == "chance":
            return st["mean"] - success_pen - pen * np.maximum(0.0, s.chance_prob - st["p_exceed"])
        raise ValueError(f"Unknown objective {objective!r}; choose one of {OBJECTIVES}.")
