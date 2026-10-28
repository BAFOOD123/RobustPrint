"""Metrics and statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from robustprint.eval.stats import bootstrap_ci, holm, paired_comparison, rank_biserial, with_holm
from robustprint.models.base import Prediction
from robustprint.uncertainty import metrics as m_


def test_crps_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    y, mu, sd = 0.7, 0.2, 1.3
    draws = rng.normal(mu, sd, 400_000)
    draws2 = rng.normal(mu, sd, 400_000)
    mc = np.mean(np.abs(draws - y)) - 0.5 * np.mean(np.abs(draws - draws2))
    np.testing.assert_allclose(m_.gaussian_crps(np.array([y]), np.array([mu]), np.array([sd])), mc, rtol=0.01)


def test_calibrated_gaussian_has_small_calibration_error_and_nominal_coverage() -> None:
    rng = np.random.default_rng(1)
    mu = rng.normal(size=20_000)
    y = mu + rng.normal(size=20_000)
    pred = Prediction(mu, np.zeros_like(mu), np.ones_like(mu))
    rep = m_.regression_report(y, pred)
    assert rep["qce"] < 0.01
    assert abs(rep["coverage90"] - 0.9) < 0.01
    np.testing.assert_allclose(rep["nll_log"], 0.5 * np.log(2 * np.pi) + 0.5, atol=0.02)
    over = m_.regression_report(y, Prediction(mu, np.zeros_like(mu), np.full_like(mu, 0.25)))
    assert over["coverage90"] < 0.75 and over["qce"] > 0.1


def test_basic_errors() -> None:
    assert m_.rmse(np.array([0, 0]), np.array([3, 4])) == np.sqrt(12.5)
    assert m_.mae(np.array([0, 0]), np.array([3, -4])) == 3.5
    _lo, hi = m_.gaussian_interval(Prediction(np.zeros(1), np.zeros(1), np.ones(1)), 0.9)
    np.testing.assert_allclose(hi, norm.ppf(0.95))


def test_classification_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    rep = m_.classification_report(y, p)
    np.testing.assert_allclose(rep["brier"], np.mean([0.01, 0.04, 0.04, 0.01]))
    assert rep["accuracy"] == 1.0
    rng = np.random.default_rng(0)
    p = rng.random(50_000)
    yy = rng.random(50_000) < p
    assert m_.expected_calibration_error(yy, p) < 0.02
    mean_p, freq, count = m_.reliability_curve(yy, p)
    assert count.sum() == 50_000 and np.nanmax(np.abs(mean_p - freq)) < 0.03


def test_holm_matches_definition() -> None:
    np.testing.assert_allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    assert holm(np.array([0.5, 0.9])).max() <= 1.0


def test_rank_biserial_and_paired_comparison() -> None:
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    assert rank_biserial(np.array([-1.0, -2.0])) == -1.0
    assert rank_biserial(np.zeros(3)) == 0.0
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1, 40)
    b = a - 0.8 + 0.1 * rng.normal(size=40)
    c = paired_comparison(a, b, "m", "A", "B")
    assert c.p_value < 1e-6 and c.diff_ci_low > 0.5 and c.rank_biserial > 0.9
    adjusted = with_holm([c, paired_comparison(a, a, "m", "A", "A")])
    assert adjusted[1].p_value == 1.0 and adjusted[0].p_holm >= adjusted[0].p_value


def test_bootstrap_ci_contains_mean() -> None:
    est, lo, hi = bootstrap_ci(np.arange(100.0))
    assert lo < est < hi and est == 49.5
    assert np.isnan(bootstrap_ci(np.array([np.nan]))[0])
    med, _, _ = bootstrap_ci(np.array([1.0, 2.0, 10.0]), statistic="median")
    assert med == 2.0
