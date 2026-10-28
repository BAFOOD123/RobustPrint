"""Strength models, conformal wrappers and the success classifier."""

from __future__ import annotations

import numpy as np
import pytest

from robustprint.data.loaders import to_arrays
from robustprint.data.validation import validate_frame
from robustprint.models.base import Prediction, ProbabilisticRegressor, gaussian_from_quantiles
from robustprint.models.classifier import SuccessClassifier
from robustprint.models.conformal import (
    ConformalizedQuantileRegression,
    NormalisedConformal,
    conformal_quantile,
    group_split,
)
from robustprint.models.features import FeatureBuilder
from robustprint.models.registry import MODEL_NAMES, make_model
from robustprint.uncertainty.metrics import interval_coverage

FB = FeatureBuilder()


@pytest.fixture(scope="module")
def xy(small_dataset):  # type: ignore[no-untyped-def]
    arrays = to_arrays(validate_frame(small_dataset)[0])
    X = FB.transform(arrays.composition, arrays.params, arrays.moisture)
    ok = arrays.strength_rows
    return X, arrays, X[ok], np.log(arrays.strength[ok]), arrays.groups[ok]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts_valid_gaussians(name: str, xy) -> None:  # type: ignore[no-untyped-def]
    _, _, X, y, _ = xy
    model = make_model(name, FB, seed=0, epochs=60, n_members=2, gp_restarts=0)
    assert isinstance(model, ProbabilisticRegressor)
    pred = model.fit(X, y).predict(X[:7])
    assert isinstance(pred, Prediction) and len(pred) == 7
    assert np.isfinite(pred.mean).all()
    assert (pred.var_epistemic >= 0).all() and (pred.var_aleatoric > 0).all()
    assert np.abs(pred.mean - y[:7]).mean() < 2.0


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValueError):
        make_model("svm", FB)


def test_neural_models_are_deterministic(xy) -> None:  # type: ignore[no-untyped-def]
    _, _, X, y, _ = xy
    a = make_model("deep_ensemble", FB, seed=3, epochs=40, n_members=2).fit(X, y).predict(X[:5])
    b = make_model("deep_ensemble", FB, seed=3, epochs=40, n_members=2).fit(X, y).predict(X[:5])
    np.testing.assert_array_equal(a.mean, b.mean)


def test_gp_noise_floor_respected(xy) -> None:  # type: ignore[no-untyped-def]
    _, _, X, y, _ = xy
    gp = make_model("gp", FB, gp_restarts=0).fit(X, y)
    assert gp.noise_variance >= 0.03**2 * 0.999  # type: ignore[attr-defined]
    assert gp.length_scales.shape == (X.shape[1],)  # type: ignore[attr-defined]


def test_feature_builder_roundtrip_and_raw_encoding(xy) -> None:  # type: ignore[no-untyped-def]
    _, arrays, *_ = xy
    for enc in ("ilr", "raw"):
        fb = FeatureBuilder(encoding=enc)
        X = fb.transform(arrays.composition, arrays.params, arrays.moisture)
        assert X.shape[1] == fb.n_features(5)
        comp, params, moist = fb.decode(X, 5)
        np.testing.assert_allclose(comp, arrays.composition, atol=1e-6)
        np.testing.assert_allclose(params, arrays.params, atol=1e-8)
        np.testing.assert_allclose(moist, arrays.moisture, rtol=1e-6)
    with pytest.raises(ValueError):
        FeatureBuilder(encoding="alr")


def test_conformal_quantile_finite_sample_rule() -> None:
    scores = np.arange(1, 11, dtype=float)
    assert conformal_quantile(scores, 0.1) == 10.0
    assert conformal_quantile(scores, 0.5) == 6.0
    assert conformal_quantile(scores[:5], 0.05) == float("inf")


def test_group_split_keeps_groups_together() -> None:
    groups = np.repeat(np.arange(20), 3)
    proper, calib = group_split(groups, 0.25, np.random.default_rng(0))
    assert not set(groups[proper]) & set(groups[calib])
    assert len(set(groups[calib])) == 5


def test_conformal_wrappers_cover_on_exchangeable_data() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = X[:, 0] + 0.3 * rng.normal(size=600)
    g = np.repeat(np.arange(300), 2)
    Xt = rng.normal(size=(2000, 3))
    yt = Xt[:, 0] + 0.3 * rng.normal(size=2000)
    for w in (
        NormalisedConformal(lambda: make_model("ridge_interactions", FB), alpha=0.1),
        ConformalizedQuantileRegression(alpha=0.1),
    ):
        w.fit(X, y, g)
        cov, width = interval_coverage(yt, *w.predict_interval(Xt))
        assert 0.85 <= cov <= 0.95, (w.name, cov)
        assert width > 0


@pytest.mark.parametrize("kind", ["logistic", "gbm_isotonic"])
def test_success_classifier_probabilities(kind: str, xy) -> None:  # type: ignore[no-untyped-def]
    X_all, arrays, *_ = xy
    p = SuccessClassifier(kind).fit(X_all, arrays.success).predict_proba(X_all)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.mean((p > 0.5) == arrays.success) > 0.6


def test_success_classifier_constant_fallback() -> None:
    X = np.zeros((5, 2))
    c = SuccessClassifier().fit(X, np.ones(5, dtype=bool))
    np.testing.assert_allclose(c.predict_proba(X), 6 / 7)
    with pytest.raises(ValueError):
        SuccessClassifier("svm")


def test_gaussian_from_quantiles() -> None:
    sd = gaussian_from_quantiles(np.array([-1.6448536]), np.array([1.6448536]), 0.9)
    np.testing.assert_allclose(sd, [1.0], rtol=1e-6)
