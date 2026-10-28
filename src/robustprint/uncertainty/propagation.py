"""Propagation of feedstock-composition uncertainty through a fitted predictor.

A *predictor* maps (composition, moisture, print settings) to a Gaussian on log strength
(split into epistemic and aleatoric variance) and a success probability. Given a
feedstock distribution for the next, uncharacterised batch, the predictive distribution
of log strength at fixed settings x is the mixture

    p(log y | x) = E_c,m [ N(log y; mu(c, m, x), s2_epi(c, m, x) + s2_ale(c, m, x)) ],

whose variance decomposes by the law of total variance into
input (Var_c[mu]), epistemic (E_c[s2_epi]) and aleatoric (E_c[s2_ale]) parts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from robustprint.data.loaders import ExperimentArrays
from robustprint.models.base import Prediction, ProbabilisticRegressor
from robustprint.models.classifier import SuccessClassifier
from robustprint.models.features import FeatureBuilder
from robustprint.physics.composition import FloatArray, dirichlet_ilr_moments, ilr_inverse
from robustprint.physics.simulator import DEFAULT_CONFIG, Feedstock, SimulatorConfig, true_outcomes


class Predictor(Protocol):
    """Anything that predicts log-strength distributions and success probabilities."""

    def predict(
        self, composition: FloatArray, moisture: FloatArray, params: FloatArray
    ) -> tuple[Prediction, FloatArray]:
        """Row-wise prediction for aligned (n, D), (n,), (n, 8) inputs."""
        ...


@dataclass
class ModelPredictor:
    """A fitted strength model + success classifier.

    Attributes:
        features: Feature builder used at training time.
        strength_model: Fitted regressor on log strength (job means).
        success_model: Fitted success classifier.
        specimen_log_var: Pooled within-job log-variance of single specimens.
        mean_replicates: Average number of replicates per training job.
        single_part: If True, aleatoric variance refers to one printed part rather than the
            mean of ``mean_replicates`` specimens (the training target).
    """

    features: FeatureBuilder
    strength_model: ProbabilisticRegressor
    success_model: SuccessClassifier
    specimen_log_var: float = 0.0
    mean_replicates: float = 1.0
    single_part: bool = True

    def predict(
        self, composition: FloatArray, moisture: FloatArray, params: FloatArray
    ) -> tuple[Prediction, FloatArray]:
        """Predict log strength (with single-part correction) and P(success)."""
        X = self.features.transform(composition, params, moisture)
        pred = self.strength_model.predict(X)
        p = self.success_model.predict_proba(X)
        if self.single_part and self.mean_replicates > 1:
            extra = self.specimen_log_var * (1.0 - 1.0 / self.mean_replicates)
            pred = Prediction(pred.mean, pred.var_epistemic, pred.var_aleatoric + extra)
        return pred, p


@dataclass
class OraclePredictor:
    """The ground-truth simulator in predictor form (no epistemic uncertainty).

    Aleatoric variance on log strength = specimen CV^2 + batch-effect variance, which is
    exactly the spread of a single realised part around the noise-free truth.
    """

    cfg: SimulatorConfig = DEFAULT_CONFIG

    def predict(
        self, composition: FloatArray, moisture: FloatArray, params: FloatArray
    ) -> tuple[Prediction, FloatArray]:
        """Exact log-mean strength and success probability."""
        out = true_outcomes(composition, params, moisture, None, self.cfg)
        ale = out["noise_cv"] ** 2 + self.cfg.batch_effect_sd**2
        return Prediction(np.log(out["strength_MPa"]), np.zeros_like(ale), ale), out["p_success"]


def pooled_specimen_log_variance(arrays: ExperimentArrays) -> float:
    """Pooled within-job variance of log strength from replicate standard deviations.

    Uses var(log y) ~= (sd / mean)^2 (delta method), pooled with n - 1 degrees of freedom.
    Returns 0 when no replicate information exists.
    """
    ok = arrays.strength_rows & np.isfinite(arrays.strength_sd) & (arrays.n_replicates > 1)
    if not ok.any():
        return 0.0
    cv2 = (arrays.strength_sd[ok] / arrays.strength[ok]) ** 2
    dof = arrays.n_replicates[ok] - 1
    return float(np.sum(dof * cv2) / np.sum(dof))


@dataclass(frozen=True)
class FeedstockSamples:
    """Common random numbers: fixed composition and moisture draws for one feedstock."""

    composition: FloatArray
    moisture: FloatArray

    def __len__(self) -> int:
        return int(self.composition.shape[0])


def sample_feedstock(feedstock: Feedstock, n: int, rng: np.random.Generator) -> FeedstockSamples:
    """Draw ``n`` (composition, moisture) pairs from a feedstock distribution."""
    alpha = np.maximum(feedstock.concentration * feedstock.mean_array(), 1e-3)
    comp = np.maximum(rng.dirichlet(alpha, size=n), 1e-6)
    comp = comp / comp.sum(1, keepdims=True)
    moist = feedstock.moisture_median_pct * np.exp(feedstock.moisture_log_sd * rng.standard_normal(n))
    return FeedstockSamples(comp, np.clip(moist, 0.02, 3.0))


@dataclass(frozen=True)
class PropagationResult:
    """Moments of the log-strength predictive mixture at one setting.

    Attributes:
        mean: Mixture mean of log strength.
        var_input: Variance explained by composition/moisture uncertainty.
        var_epistemic: Expected model (epistemic) variance.
        var_aleatoric: Expected noise variance.
        p_success: Expected success probability.
        seconds: Wall-clock time of the computation.
    """

    mean: float
    var_input: float
    var_epistemic: float
    var_aleatoric: float
    p_success: float
    seconds: float

    @property
    def var(self) -> float:
        """Total variance."""
        return self.var_input + self.var_epistemic + self.var_aleatoric

    def shares(self) -> dict[str, float]:
        """Fractions of total variance by source."""
        total = self.var
        return {
            "input": self.var_input / total,
            "epistemic": self.var_epistemic / total,
            "aleatoric": self.var_aleatoric / total,
        }


def _combine(pred: Prediction, p: FloatArray, weights: FloatArray, t0: float) -> PropagationResult:
    mean = float(weights @ pred.mean)
    return PropagationResult(
        mean=mean,
        var_input=float(weights @ (pred.mean - mean) ** 2),
        var_epistemic=float(weights @ pred.var_epistemic),
        var_aleatoric=float(weights @ pred.var_aleatoric),
        p_success=float(weights @ p),
        seconds=time.perf_counter() - t0,
    )


def monte_carlo(predictor: Predictor, params: FloatArray, samples: FeedstockSamples) -> PropagationResult:
    """Monte Carlo propagation at one setting using pre-drawn feedstock samples."""
    t0 = time.perf_counter()
    n = len(samples)
    pred, p = predictor.predict(samples.composition, samples.moisture, np.tile(params, (n, 1)))
    return _combine(pred, p, np.full(n, 1.0 / n), t0)


def unscented(predictor: Predictor, params: FloatArray, feedstock: Feedstock, kappa: float = 0.0) -> PropagationResult:
    """Unscented-transform propagation (Julier & Uhlmann, 2004).

    The Dirichlet composition is approximated by a Gaussian in ilr coordinates with its
    exact ilr mean and covariance (see :func:`dirichlet_ilr_moments`); log moisture is an
    independent Gaussian coordinate. 2n + 1 sigma points are mapped back to the simplex.
    The default kappa = 0 gives equal positive weights (the centre point gets weight 0),
    which avoids negative variance estimates in 5 dimensions.
    """
    t0 = time.perf_counter()
    alpha = np.maximum(feedstock.concentration * feedstock.mean_array(), 1e-3)
    m_ilr, c_ilr = dirichlet_ilr_moments(alpha)
    d = m_ilr.size + 1
    mean = np.append(m_ilr, np.log(feedstock.moisture_median_pct))
    cov = np.zeros((d, d))
    cov[:-1, :-1] = c_ilr
    cov[-1, -1] = feedstock.moisture_log_sd**2
    k = kappa
    root = np.linalg.cholesky((d + k) * cov)
    points = np.vstack([mean, mean + root.T, mean - root.T])
    weights = np.full(2 * d + 1, 1.0 / (2 * (d + k)))
    weights[0] = k / (d + k)
    comp = ilr_inverse(points[:, :-1])
    moist = np.clip(np.exp(points[:, -1]), 0.02, 3.0)
    pred, p = predictor.predict(comp, moist, np.tile(params, (len(points), 1)))
    return _combine(pred, p, weights, t0)


def mixture_samples(pred: Prediction, normals: FloatArray) -> FloatArray:
    """Log-strength samples of the mixture: one row per component, S draws each."""
    return np.asarray(pred.mean[:, None] + pred.std[:, None] * normals, dtype=float)
