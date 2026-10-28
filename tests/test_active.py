"""Active learning: GP posterior algebra, greedy selection and the planner."""

from __future__ import annotations

import numpy as np
import pytest

from robustprint.active.acquisition import GPPosterior, experiment_cost, greedy_select
from robustprint.active.planner import candidate_pool, experiments_table, select_experiments, table_to_markdown
from robustprint.data.loaders import to_arrays
from robustprint.data.validation import validate_frame
from robustprint.physics.simulator import ISS_NOMINAL
from robustprint.pipeline import train_predictor


@pytest.fixture(scope="module")
def predictor(small_dataset):  # type: ignore[no-untyped-def]
    return train_predictor(to_arrays(validate_frame(small_dataset)[0]), "hybrid_gp", seed=0, gp_restarts=0)


def test_posterior_variance_matches_sklearn(predictor) -> None:  # type: ignore[no-untyped-def]
    comp, moist, params = candidate_pool(ISS_NOMINAL, 16, np.random.default_rng(0))
    X = predictor.features.transform(comp, params, moist)
    post = GPPosterior(predictor.strength_model)
    pred = predictor.strength_model.predict(X)
    np.testing.assert_allclose(post.var(X) * predictor.strength_model.y_std_**2, pred.var_epistemic, rtol=1e-5)
    cov = post.cov(X, X)
    np.testing.assert_allclose(np.diag(cov), post.var(X), rtol=1e-5, atol=1e-9)
    assert np.allclose(cov, cov.T, atol=1e-10)


def test_greedy_fantasy_update_is_exact(predictor) -> None:  # type: ignore[no-untyped-def]
    comp, moist, params = candidate_pool(ISS_NOMINAL, 32, np.random.default_rng(1))
    X = predictor.features.transform(comp, params, moist)
    post = GPPosterior(predictor.strength_model)
    sel = greedy_select(post, X, np.ones(len(X)), 3, "max_variance", n_replicates=1)
    assert len(set(sel.indices)) == 3
    assert sel.scores[0] >= sel.scores[1] - 1e-12
    with pytest.raises(ValueError):
        greedy_select(post, X, np.ones(len(X)), 2, "targeted_ivr")
    with pytest.raises(ValueError):
        greedy_select(post, X, np.ones(len(X)), 2, "random")


def test_cost_increases_with_slow_printing() -> None:
    slow = np.array([[265, 100, 10, 0.1, 100, 0, 100, 0.0]])
    fast = np.array([[265, 100, 80, 0.3, 100, 0, 100, 0.0]])
    assert experiment_cost(slow)[0] > experiment_cost(fast)[0] > 1.0


@pytest.mark.parametrize("strategy", ["random", "sobol", "max_variance", "targeted_ivr", "cost_aware_ivr"])
def test_select_experiments_all_strategies(predictor, strategy: str) -> None:  # type: ignore[no-untyped-def]
    rec = np.array([265, 100, 20, 0.15, 100, 0, 100, 0.0])
    comp, moist, params, scores = select_experiments(
        predictor, ISS_NOMINAL, 3, strategy, np.random.default_rng(0), recommended=rec, n_candidates=64
    )
    assert comp.shape == (3, 5) and params.shape == (3, 8)
    np.testing.assert_allclose(comp.sum(1), 1.0)
    table = experiments_table(predictor, comp, moist, params, scores, strategy)
    assert list(table["rank"]) == [1, 2, 3]
    assert (table["pred_strength_90lo_MPa"] <= table["pred_strength_90hi_MPa"]).all()
    md = table_to_markdown(table)
    assert md.count("\n") == 4


def test_select_experiments_errors(predictor) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        select_experiments(predictor, ISS_NOMINAL, 2, "greedy", np.random.default_rng(0))
    with pytest.raises(ValueError):
        select_experiments(predictor, ISS_NOMINAL, 2, "targeted_ivr", np.random.default_rng(0), n_candidates=16)
