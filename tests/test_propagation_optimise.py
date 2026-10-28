"""Uncertainty propagation, robust objectives, the optimiser and deployment."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robustprint import space
from robustprint.eval.deployment import deploy, draw_deployment
from robustprint.models.base import Prediction
from robustprint.optimize.objectives import (
    OBJECTIVES,
    ObjectiveSettings,
    RobustProblem,
    weighted_cvar,
    weighted_quantile,
)
from robustprint.optimize.robust import maximise, recommend, safe_bounds
from robustprint.physics.simulator import ISS_NOMINAL, ISS_SHIFTED
from robustprint.uncertainty.propagation import (
    OraclePredictor,
    mixture_samples,
    monte_carlo,
    sample_feedstock,
    unscented,
)

GOOD = np.array([265.0, 100.0, 20.0, 0.15, 100.0, 0.0, 100.0, 0.0])


class LinearPredictor:
    """log y = a . ilr-free linear function of composition; known variance split."""

    def predict(self, composition, moisture, params):  # type: ignore[no-untyped-def]
        mean = 2.0 + 3.0 * composition[:, 1] - 2.0 * composition[:, 3] - 0.001 * (params[:, 0] - 265.0) ** 2
        n = len(mean)
        return Prediction(mean, np.full(n, 0.01), np.full(n, 0.02)), np.full(n, 0.95)


def test_monte_carlo_decomposition_matches_analytic() -> None:
    fs = ISS_NOMINAL
    res = monte_carlo(LinearPredictor(), GOOD, sample_feedstock(fs, 200_000, np.random.default_rng(0)))
    a = fs.concentration * fs.mean_array()
    a0 = a.sum()
    cov = (np.diag(a / a0) - np.outer(a / a0, a / a0)) / (a0 + 1)
    w = np.array([0, 3.0, 0, -2.0, 0])
    np.testing.assert_allclose(res.var_input, w @ cov @ w, rtol=0.03)
    np.testing.assert_allclose(res.var_epistemic, 0.01)
    np.testing.assert_allclose(res.var_aleatoric, 0.02)
    assert abs(sum(res.shares().values()) - 1) < 1e-12
    assert res.p_success == pytest.approx(0.95)


def test_unscented_exact_for_linear_in_ilr_response() -> None:
    """For a response linear in ilr coordinates the UT mean/variance equal the exact moments."""
    from robustprint.physics.composition import dirichlet_ilr_moments, ilr

    w = np.array([0.5, -1.0, 0.3, 0.8])

    class IlrLinear:
        def predict(self, composition, moisture, params):  # type: ignore[no-untyped-def]
            mean = ilr(composition) @ w
            n = len(mean)
            return Prediction(mean, np.zeros(n), np.full(n, 0.01)), np.ones(n)

    fs = replace(ISS_NOMINAL, moisture_log_sd=0.6)
    ut = unscented(IlrLinear(), GOOD, fs)
    m, c = dirichlet_ilr_moments(fs.concentration * fs.mean_array())
    np.testing.assert_allclose(ut.mean, m @ w, rtol=1e-9)
    np.testing.assert_allclose(ut.var_input, w @ c @ w, rtol=1e-9)


def test_unscented_on_oracle_is_finite_and_same_order_as_mc() -> None:
    """On the nonlinear simulator the UT is only roughly right (reported in E4)."""
    oracle = OraclePredictor()
    ref = monte_carlo(oracle, GOOD, sample_feedstock(ISS_NOMINAL, 20_000, np.random.default_rng(1)))
    ut = unscented(oracle, GOOD, ISS_NOMINAL)
    assert abs(ut.mean - ref.mean) < 0.1
    assert 0.33 < ut.var / ref.var < 3.0


def test_mixture_samples_shape() -> None:
    pred = Prediction(np.zeros(3), np.ones(3), np.zeros(3))
    assert mixture_samples(pred, np.zeros((3, 4))).shape == (3, 4)


def test_weighted_quantile_and_cvar_against_unweighted() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(2, 20_000))
    w = np.ones_like(v)
    np.testing.assert_allclose(weighted_quantile(v, w, 0.05), np.quantile(v, 0.05, axis=1), atol=0.01)
    tail = np.sort(v, axis=1)[:, :2000].mean(axis=1)
    np.testing.assert_allclose(weighted_cvar(v, w, 0.1), tail, atol=0.01)
    w2 = np.where(v > 0, 1.0, 1e-9)
    assert (weighted_quantile(v, w2, 0.05) > 0).all()


def test_objective_ordering_on_oracle() -> None:
    prob = RobustProblem(OraclePredictor(), ISS_NOMINAL, n_compositions=128, n_noise=8, seed=0)
    st = prob.statistics(GOOD[None, :])
    assert st["cvar"][0] <= st["quantile"][0] + 1e-9 <= st["mean"][0] + 1e-9
    assert 0 <= st["p_success"][0] <= 1
    for objective in OBJECTIVES:
        assert np.isfinite(prob.score(objective, GOOD[None, :])).all()
    with pytest.raises(ValueError):
        prob.score("median", GOOD[None, :])


def test_success_penalty_applies() -> None:
    strict = RobustProblem(OraclePredictor(), ISS_NOMINAL, ObjectiveSettings(p_min=0.999), 32, 4, seed=0)
    loose = RobustProblem(OraclePredictor(), ISS_NOMINAL, ObjectiveSettings(p_min=0.0), 32, 4, seed=0)
    assert strict.score("expected", GOOD[None, :])[0] < loose.score("expected", GOOD[None, :])[0]


def test_maximise_finds_quadratic_optimum() -> None:
    target = np.array([0.3, 0.7])
    res = maximise(lambda x: -np.sum((x - target) ** 2, axis=1), np.zeros(2), np.ones(2), budget=400, seed=0)
    np.testing.assert_allclose(res.params, target, atol=1e-2)
    assert res.n_evaluations <= 400 + 16


def test_safe_bounds_use_dominant_polymer() -> None:
    low, high = safe_bounds(ISS_NOMINAL)
    assert low[0] == 210.0 and high[0] == 290.0
    pet_rich = replace(ISS_SHIFTED, mean=(0.1, 0.6, 0.1, 0.1, 0.1))
    low2, high2 = safe_bounds(pet_rich)
    assert low2[0] == 255.0 and high2[0] == 285.0


def test_oracle_recommendation_beats_defaults_on_deployment() -> None:
    prob = RobustProblem(OraclePredictor(), ISS_NOMINAL, n_compositions=64, n_noise=8, seed=0)
    rec = recommend(prob, "expected", budget=300, seed=0)
    dep = draw_deployment(ISS_NOMINAL, 800, 5)
    good = deploy(rec.params, dep)
    base = deploy(space.defaults(), dep)
    assert good["expected_utility"] > base["expected_utility"]
    assert 0 <= good["failure_rate"] <= 1
    assert good["q05_strength"] <= good["mean_strength"]


def test_deployment_is_paired_and_deterministic() -> None:
    dep = draw_deployment(ISS_NOMINAL, 300, 1)
    assert deploy(GOOD, dep) == deploy(GOOD, dep)
