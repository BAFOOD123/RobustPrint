"""Baseline regressors: physics rule of mixtures, Bayesian ridge with interactions, GBM quantiles."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from robustprint.models.base import Prediction, gaussian_from_quantiles
from robustprint.models.features import FeatureBuilder
from robustprint.physics import process
from robustprint.physics.composition import FloatArray
from robustprint.physics.materials import DEFAULT_LIBRARY, MaterialLibrary


def rule_of_mixtures_strength(composition: FloatArray, library: MaterialLibrary = DEFAULT_LIBRARY) -> FloatArray:
    """Mass-fraction rule of mixtures of bulk strengths (no blend or process effects)."""
    return np.asarray(composition @ library.vector("strength_MPa"), dtype=float)


def simplified_physics_prior(
    composition: FloatArray, params: FloatArray, library: MaterialLibrary = DEFAULT_LIBRARY
) -> FloatArray:
    """A deliberately incomplete physics prior for the hybrid model (log scale).

    log prior = log ROM(c) + log section(infill) + log TsaiHill(raster; t = q = 0.5).
    It ignores immiscibility, fibre/inclusion mechanics, thermal bonding, extrusion limits,
    degradation and moisture: the kind of first-principles estimate a researcher can write
    down before any experiment. It shares two functional forms (infill, raster) with the
    simulator, which flatters it on synthetic data (see docs/LIMITATIONS.md).
    """
    rom = rule_of_mixtures_strength(composition, library)
    section = process.section_fraction(params[:, 4])
    angle = process.tsai_hill_factor(params[:, 5], np.full(len(params), 0.5), np.full(len(params), 0.5))
    return np.asarray(np.log(rom) + np.log(section) + np.log(angle), dtype=float)


class PhysicsROMBaseline:
    """log y = a + b log ROM(c): composition-only physics baseline with OLS uncertainty."""

    name = "physics_rom"

    def __init__(self, features: FeatureBuilder, library: MaterialLibrary = DEFAULT_LIBRARY) -> None:
        self.features = features
        self.library = library

    def _design(self, X: FloatArray) -> FloatArray:
        comp, _, _ = self.features.decode(X, self.library.n)
        return np.column_stack([np.ones(len(comp)), np.log(rule_of_mixtures_strength(comp, self.library))])

    def fit(self, X: FloatArray, y: FloatArray, n_replicates: np.ndarray | None = None) -> PhysicsROMBaseline:
        """Ordinary least squares on the log rule-of-mixtures strength."""
        A = self._design(X)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = max(len(y) - A.shape[1], 1)
        self.sigma2_ = float(resid @ resid / dof)
        self.coef_ = coef
        self.cov_ = self.sigma2_ * np.linalg.pinv(A.T @ A)
        return self

    def predict(self, X: FloatArray) -> Prediction:
        """Gaussian predictive with parameter (epistemic) and residual (aleatoric) variance."""
        A = self._design(X)
        epi = np.einsum("ni,ij,nj->n", A, self.cov_, A)
        return Prediction(A @ self.coef_, epi, np.full(len(A), self.sigma2_))


class RidgeInteractions:
    """Bayesian ridge regression on standardised features with all pairwise interactions."""

    name = "ridge_interactions"

    def __init__(self, degree: int = 2) -> None:
        self.degree = degree

    def fit(self, X: FloatArray, y: FloatArray, n_replicates: np.ndarray | None = None) -> RidgeInteractions:
        """Fit with evidence-maximised regularisation (scikit-learn ``BayesianRidge``)."""
        self.pipe_: Pipeline = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(self.degree, include_bias=False),
            StandardScaler(),
            BayesianRidge(),
        )
        self.pipe_.fit(X, y)
        return self

    def predict(self, X: FloatArray) -> Prediction:
        """Predictive mean and variance; noise variance is 1 / alpha_ of the Bayesian ridge."""
        mean, std = self.pipe_.predict(X, return_std=True)
        noise = 1.0 / float(self.pipe_[-1].alpha_)
        return Prediction(mean, np.maximum(std**2 - noise, 0.0), np.full(len(mean), noise))


class GBMQuantile:
    """Histogram gradient boosting: one model per quantile plus one for the mean.

    The Gaussian predictive uses the mean model and a standard deviation matched to the
    5 %–95 % quantile interval. No epistemic/aleatoric split is available, so the whole
    variance is reported as aleatoric.
    """

    name = "gbm_quantile"

    def __init__(
        self,
        quantiles: tuple[float, float] = (0.05, 0.95),
        random_state: int = 0,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 15,
    ) -> None:
        self.quantiles = quantiles
        self.random_state = random_state
        self.kwargs = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": 5,
            "random_state": random_state,
        }

    def fit(self, X: FloatArray, y: FloatArray, n_replicates: np.ndarray | None = None) -> GBMQuantile:
        """Fit mean and two quantile models."""
        self.mean_model_ = HistGradientBoostingRegressor(loss="squared_error", **self.kwargs).fit(X, y)
        self.lower_ = HistGradientBoostingRegressor(loss="quantile", quantile=self.quantiles[0], **self.kwargs).fit(
            X, y
        )
        self.upper_ = HistGradientBoostingRegressor(loss="quantile", quantile=self.quantiles[1], **self.kwargs).fit(
            X, y
        )
        return self

    def predict_quantiles(self, X: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Lower and upper quantile predictions (sorted so lower <= upper)."""
        lo, hi = self.lower_.predict(X), self.upper_.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def predict(self, X: FloatArray) -> Prediction:
        """Gaussian approximation from the quantile interval."""
        lo, hi = self.predict_quantiles(X)
        sd = gaussian_from_quantiles(lo, hi, self.quantiles[1] - self.quantiles[0])
        mean = self.mean_model_.predict(X)
        return Prediction(mean, np.zeros(len(mean)), sd**2)
