"""E4 propagation accuracy, E5 sensitivity, E6 Pareto fronts and E7 public-data checks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold

from robustprint import space
from robustprint.data.public import load_kaggle_3dprinter
from robustprint.eval.common import seed_for
from robustprint.eval.deployment import deploy, draw_deployment
from robustprint.models.features import FeatureBuilder
from robustprint.models.registry import make_model
from robustprint.optimize.objectives import ObjectiveSettings, RobustProblem
from robustprint.optimize.robust import maximise, recommend, safe_bounds
from robustprint.physics.composition import FloatArray
from robustprint.physics.process import energy_proxy, print_time_proxy
from robustprint.physics.simulator import DEFAULT_CONFIG, ISS_NOMINAL, Feedstock, sample_batches, true_outcomes
from robustprint.uncertainty.metrics import regression_report
from robustprint.uncertainty.propagation import (
    ModelPredictor,
    OraclePredictor,
    Predictor,
    monte_carlo,
    sample_feedstock,
    unscented,
)
from robustprint.uncertainty.sensitivity import Sampler, first_order_binned, pivot_coordinate, sobol_indices


def propagation_comparison(
    predictor: Predictor,
    feedstock: Feedstock,
    n_points: int,
    base_seed: int,
    sizes: tuple[int, ...] = (16, 64, 256, 1024),
    repeats: int = 20,
    reference_size: int = 20_000,
    label: str = "",
) -> pd.DataFrame:
    """Accuracy and speed of MC (several sizes) and the unscented transform vs a large MC."""
    low, high = safe_bounds(feedstock)
    rng = np.random.default_rng(seed_for(base_seed, 4))
    points = low + rng.random((n_points, space.N_PARAMS)) * (high - low)
    reference_samples = sample_feedstock(feedstock, reference_size, np.random.default_rng(seed_for(base_seed, 5)))
    rows = []
    for i, x in enumerate(points):
        ref = monte_carlo(predictor, x, reference_samples)
        ref_sd = np.sqrt(ref.var)
        ut = unscented(predictor, x, feedstock)
        rows.append(
            {
                "model": label,
                "point": i,
                "method": "unscented",
                "n_evals": 11,
                "repeat": 0,
                "err_mean": ut.mean - ref.mean,
                "err_sd": np.sqrt(ut.var) - ref_sd,
                "seconds": ut.seconds,
                "ref_sd": ref_sd,
            }
        )
        for k in sizes:
            for r in range(repeats):
                samples = sample_feedstock(feedstock, k, np.random.default_rng(seed_for(base_seed, 6, i, k, r)))
                mc = monte_carlo(predictor, x, samples)
                rows.append(
                    {
                        "model": label,
                        "point": i,
                        "method": f"mc_{k}",
                        "n_evals": k,
                        "repeat": r,
                        "err_mean": mc.mean - ref.mean,
                        "err_sd": np.sqrt(mc.var) - ref_sd,
                        "seconds": mc.seconds,
                        "ref_sd": ref_sd,
                    }
                )
    return pd.DataFrame(rows)


def decomposition_at(
    predictor: Predictor, settings: dict[str, FloatArray], feedstock: Feedstock, seed: int, label: str
) -> pd.DataFrame:
    """Variance decomposition (input / epistemic / aleatoric) at named settings."""
    samples = sample_feedstock(feedstock, 4096, np.random.default_rng(seed))
    rows = []
    for name, x in settings.items():
        res = monte_carlo(predictor, x, samples)
        rows.append(
            {
                "model": label,
                "setting": name,
                "mean_log": res.mean,
                "var_input": res.var_input,
                "var_epistemic": res.var_epistemic,
                "var_aleatoric": res.var_aleatoric,
                "p_success": res.p_success,
                **{f"share_{k}": v for k, v in res.shares().items()},
            }
        )
    return pd.DataFrame(rows)


def _uniform_sampler(low: float, high: float) -> Sampler:
    def draw(n: int, rng: np.random.Generator) -> FloatArray:
        return np.asarray(low + rng.random((n, 1)) * (high - low), dtype=float)

    return draw


def _group_samplers(feedstock: Feedstock, low: FloatArray, high: FloatArray) -> dict[str, Sampler]:
    samplers: dict[str, Sampler] = {}
    for k, name in enumerate(space.PARAM_NAMES):
        samplers[name] = _uniform_sampler(float(low[k]), float(high[k]))
    samplers["composition"] = lambda n, rng: sample_feedstock(feedstock, n, rng).composition
    samplers["moisture"] = lambda n, rng: sample_feedstock(feedstock, n, rng).moisture[:, None]
    samplers["batch_effect"] = lambda n, rng: DEFAULT_CONFIG.batch_effect_sd * rng.standard_normal((n, 1))
    return samplers


def sensitivity_analysis(
    feedstock: Feedstock, n: int, base_seed: int, model: ModelPredictor | None, recommended: FloatArray
) -> dict[str, pd.DataFrame]:
    """Global Sobol indices (truth and fitted model) and per-component indices at a setting."""
    low, high = safe_bounds(feedstock)
    samplers = _group_samplers(feedstock, low, high)

    def stack(groups: dict[str, FloatArray]) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        x = np.hstack([groups[name] for name in space.PARAM_NAMES])
        effect = groups["batch_effect"][:, 0] if "batch_effect" in groups else np.zeros(len(x))
        return groups["composition"], x, groups["moisture"][:, 0], effect

    def truth(key: str) -> Callable[[dict[str, FloatArray]], FloatArray]:
        def f(groups: dict[str, FloatArray]) -> FloatArray:
            c, x, m, b = stack(groups)
            out = true_outcomes(c, x, m, b)
            return np.log(out["strength_MPa"]) if key == "log_strength" else out["p_success"]

        return f

    rows = []
    for key in ("log_strength", "p_success"):
        res = sobol_indices(truth(key), samplers, n, np.random.default_rng(seed_for(base_seed, 7)))
        rows += [{"source": "ground_truth", "output": key, "input": g, **v} for g, v in res.items()]
    if model is not None:

        def fitted(groups: dict[str, FloatArray]) -> FloatArray:
            c, x, m, _ = stack(groups)
            return model.predict(c, m, x)[0].mean

        model_samplers = {g: s for g, s in samplers.items() if g != "batch_effect"}
        res = sobol_indices(fitted, model_samplers, n, np.random.default_rng(seed_for(base_seed, 7)))
        rows += [{"source": "fitted_model", "output": "log_strength", "input": g, **v} for g, v in res.items()]
    global_df = pd.DataFrame(rows)

    rng = np.random.default_rng(seed_for(base_seed, 8))
    b = sample_batches(feedstock, 20_000, rng)
    X = np.tile(recommended, (len(b), 1))
    out = true_outcomes(b.true_composition, X, b.moisture_pct, b.batch_effect)
    y = np.log(out["strength_MPa"])
    comp_rows = []
    for j, name in enumerate(DEFAULT_CONFIG.library.names):
        z = pivot_coordinate(b.true_composition, j)
        comp_rows.append(
            {
                "component": name,
                "first_order_log_strength": first_order_binned(z, y),
                "first_order_p_success": first_order_binned(z, out["p_success"]),
                "spearman_log_strength": float(spearmanr(z, y).statistic),
            }
        )
    comp_rows.append(
        {
            "component": "moisture",
            "first_order_log_strength": first_order_binned(b.moisture_pct, y),
            "first_order_p_success": first_order_binned(b.moisture_pct, out["p_success"]),
            "spearman_log_strength": float(spearmanr(b.moisture_pct, y).statistic),
        }
    )
    return {"global": global_df, "components": pd.DataFrame(comp_rows)}


def pareto_front(
    predictor: ModelPredictor,
    feedstock: Feedstock,
    time_budgets: list[float],
    base_seed: int,
    n_compositions: int,
    budget: int,
    deploy_batches: int,
) -> pd.DataFrame:
    """epsilon-constraint front: maximise CVaR subject to print time <= budget."""
    settings = ObjectiveSettings()
    problem = RobustProblem(predictor, feedstock, settings, n_compositions, 16, seed=seed_for(base_seed, 9))
    low, high = safe_bounds(feedstock)
    deployment = draw_deployment(feedstock, deploy_batches, seed_for(base_seed, 10))
    rows = []
    for t_max in [*time_budgets, float("inf")]:
        for objective in ("cvar", "expected"):

            def score(x: FloatArray, t_max: float = t_max, objective: str = objective) -> FloatArray:
                t = print_time_proxy(x[:, 2], x[:, 3], x[:, 4])
                return problem.score(objective, x) - settings.penalty_MPa * np.maximum(0.0, t - t_max)

            res = maximise(score, low, high, budget=budget, seed=seed_for(base_seed, 11))
            x = res.params
            st = problem.statistics(x[None, :])
            rows.append(
                {
                    "objective": objective,
                    "time_budget": t_max,
                    "print_time": float(print_time_proxy(x[None, 2], x[None, 3], x[None, 4])[0]),
                    "energy": float(energy_proxy(x[None, 0], x[None, 1], x[None, 2], x[None, 3], x[None, 4])[0]),
                    "pred_cvar": float(st["cvar"][0]),
                    "pred_mean": float(st["mean"][0]),
                    "pred_p_success": float(st["p_success"][0]),
                    **deploy(x, deployment),
                    **{name: float(v) for name, v in zip(space.PARAM_NAMES, x, strict=True)},
                }
            )
    frame = pd.DataFrame(rows)
    frame["non_dominated"] = _non_dominated(
        frame[["pred_cvar", "print_time", "energy"]].to_numpy(), maximise_mask=np.array([True, False, False])
    )
    return frame


def _non_dominated(values: FloatArray, maximise_mask: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows."""
    v = np.where(maximise_mask, -values, values)
    keep = np.ones(len(v), dtype=bool)
    for i in range(len(v)):
        dominated = np.all(v <= v[i], axis=1) & np.any(v < v[i], axis=1)
        keep[i] = not dominated.any()
    return keep


def oracle_reference_settings(feedstock: Feedstock, base_seed: int, budget: int = 1500) -> dict[str, FloatArray]:
    """Oracle nominal and CVaR settings (used to place decomposition and sensitivity analyses)."""
    problem = RobustProblem(OraclePredictor(), feedstock, ObjectiveSettings(), 256, 16, seed=seed_for(base_seed, 99))
    return {
        obj: recommend(problem, obj, budget=budget, seed=seed_for(base_seed, 98)).params for obj in ("nominal", "cvar")
    }


KAGGLE_NUMERIC = [
    "layer_height_mm",
    "wall_thickness_mm",
    "infill_pct",
    "nozzle_temp_C",
    "bed_temp_C",
    "print_speed_mm_s",
    "cooling_fan_pct",
]
SIMULATOR_OVERLAP = {
    "layer_height_mm": 3,
    "infill_pct": 4,
    "nozzle_temp_C": 0,
    "bed_temp_C": 1,
    "print_speed_mm_s": 2,
    "cooling_fan_pct": 7,
}


def public_benchmark(
    path: Path,
    base_seed: int,
    models: tuple[str, ...] = ("ridge_interactions", "gbm_quantile", "gp", "deep_ensemble"),
    repeats: int = 10,
) -> dict[str, pd.DataFrame] | None:
    """Trend check and repeated-CV benchmark on the Kaggle dataset, if it has been downloaded."""
    if not path.exists():
        return None
    df = load_kaggle_3dprinter(path)
    X = np.column_stack(
        [
            df[KAGGLE_NUMERIC].to_numpy(dtype=float),
            (df["material"] == "pla").to_numpy(dtype=float),
            (df["infill_pattern"] == "honeycomb").to_numpy(dtype=float),
        ]
    )
    y = np.log(df["tensile_strength_MPa"].to_numpy(dtype=float))
    fb = FeatureBuilder()
    rows = []
    for name in models:
        for fold, (tr, te) in enumerate(
            RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed_for(base_seed, 12)).split(X)
        ):
            model = make_model(name, fb, seed_for(base_seed, 13, fold), epochs=300)
            rows.append(
                {"model": name, "fold": fold, **regression_report(y[te], model.fit(X[tr], y[tr]).predict(X[te]))}
            )
    bench = pd.DataFrame(rows)

    trend_rows = []
    gp = make_model("gp", fb, base_seed).fit(X, y)
    mu = ISS_NOMINAL.mean_array()
    for k, name in enumerate(KAGGLE_NUMERIC):
        rho = float(spearmanr(df[name], df["tensile_strength_MPa"]).statistic)
        lo, hi = np.quantile(X[:, k], [0.1, 0.9])
        X_lo, X_hi = X.copy(), X.copy()
        X_lo[:, k], X_hi[:, k] = lo, hi
        pd_effect = float(np.mean(gp.predict(X_hi).mean - gp.predict(X_lo).mean))
        sim_dir = float("nan")
        if name in SIMULATOR_OVERLAP:
            j = SIMULATOR_OVERLAP[name]
            plo, phi = space.PARAMETERS[j].low, space.PARAMETERS[j].high
            xs = np.tile(space.defaults(), (2, 1))
            xs[:, 0] = 265.0
            xs[0, j], xs[1, j] = plo + 0.25 * (phi - plo), plo + 0.75 * (phi - plo)
            s = true_outcomes(np.tile(mu, (2, 1)), xs, np.full(2, 0.3))["strength_MPa"]
            sim_dir = float(np.sign(s[1] - s[0]))
        trend_rows.append(
            {
                "parameter": name,
                "spearman_data": rho,
                "gp_partial_effect_log": pd_effect,
                "simulator_direction": sim_dir,
            }
        )
    return {
        "benchmark": bench,
        "trends": pd.DataFrame(trend_rows),
        "summary": pd.DataFrame(
            [
                {
                    "n_rows": len(df),
                    "n_pla": int((df["material"] == "pla").sum()),
                    "strength_min": float(df["tensile_strength_MPa"].min()),
                    "strength_max": float(df["tensile_strength_MPa"].max()),
                }
            ]
        ),
    }
