"""E3: simulated sequential experimentation with different experiment-selection strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from robustprint import space
from robustprint.active.acquisition import experiment_cost
from robustprint.active.planner import candidate_pool, select_experiments
from robustprint.data.loaders import ExperimentArrays
from robustprint.eval.common import campaign, parallel_map, seed_for
from robustprint.optimize.objectives import ObjectiveSettings, RobustProblem
from robustprint.optimize.robust import recommend
from robustprint.physics.composition import FloatArray
from robustprint.physics.simulator import DEFAULT_CONFIG, ISS_NOMINAL, Batches, realise
from robustprint.pipeline import train_predictor
from robustprint.uncertainty.propagation import OraclePredictor

#: Dirichlet concentration of the error between a target blend and the blend actually made.
BLEND_CONCENTRATION = 200.0
#: Objective whose recommendation is tracked (the pre-registered robust objective).
TRACKED_OBJECTIVE = "cvar"


@dataclass(frozen=True)
class ALTask:
    """One (seed, strategy) sequential run."""

    seed_index: int
    strategy: str
    base_seed: int
    initial_jobs: int
    rounds: int
    batch_size: int
    candidates: int
    n_replicates: int
    budget: int
    n_compositions: int


def _new_jobs(
    composition: FloatArray,
    moisture: FloatArray,
    params: FloatArray,
    rng: np.random.Generator,
    n_replicates: int,
    start: int,
) -> ExperimentArrays:
    """Prepare blends near the target compositions and print them (simulated)."""
    n = len(params)
    true = np.array([rng.dirichlet(np.maximum(BLEND_CONCENTRATION * c, 1e-3)) for c in composition])
    true = np.maximum(true, 1e-6)
    true /= true.sum(1, keepdims=True)
    reported = np.array([rng.dirichlet(np.maximum(300.0 * c, 1e-3)) for c in true])
    reported = np.maximum(reported, 1e-6)
    reported /= reported.sum(1, keepdims=True)
    effect = DEFAULT_CONFIG.batch_effect_sd * rng.standard_normal(n)
    batches = Batches([f"AL{start + i:04d}" for i in range(n)], true, reported, moisture, effect)
    reps = [realise(batches.true_composition, params, moisture, effect, rng) for _ in range(n_replicates)]
    success = reps[0]["success"]
    strengths = np.stack([np.nan_to_num(r["strength_MPa"]) for r in reps])
    mean = np.where(success, strengths.mean(0), np.nan)
    sd = np.where(success, strengths.std(0, ddof=1), np.nan) if n_replicates > 1 else np.full(n, np.nan)
    nan = np.full(n, np.nan)
    return ExperimentArrays(
        reported, params, moisture, mean, sd, nan, nan, success, np.full(n, n_replicates), np.array(batches.batch_id)
    )


def _concat(a: ExperimentArrays, b: ExperimentArrays) -> ExperimentArrays:
    return ExperimentArrays(*(np.concatenate([getattr(a, f), getattr(b, f)]) for f in a.__dataclass_fields__))


def true_value(params: FloatArray, oracle: RobustProblem) -> dict[str, float]:
    """Ground-truth robust statistics of a setting (large fixed Monte-Carlo sample)."""
    st = oracle.statistics(np.atleast_2d(params))
    utility = st["mean"] * st["p_success"]
    return {
        "true_cvar": float(st["cvar"][0]),
        "true_p_success": float(st["p_success"][0]),
        "true_expected_utility": float(utility[0]),
        "true_score": float(oracle.score(TRACKED_OBJECTIVE, np.atleast_2d(params))[0]),
    }


def run_al_task(task: ALTask) -> list[dict[str, object]]:
    """Run one sequential campaign and record the true quality of each round's recommendation."""
    s = seed_for(task.base_seed, 3, task.seed_index)
    rng = np.random.default_rng(s + 1)
    n_init_batches = max(task.initial_jobs // 4, 1)
    data = campaign(ISS_NOMINAL, n_init_batches, 4, s, task.n_replicates, prefix="I")
    oracle = RobustProblem(
        OraclePredictor(), ISS_NOMINAL, ObjectiveSettings(), 512, 16, seed=seed_for(task.base_seed, 97)
    )
    rows: list[dict[str, object]] = []
    cumulative_cost = float(np.sum(experiment_cost(data.params)))
    total = task.rounds * task.batch_size
    sobol_comp, sobol_moist, sobol_params = candidate_pool(ISS_NOMINAL, total, np.random.default_rng(s + 4))
    for round_index in range(task.rounds + 1):
        predictor = train_predictor(data, "hybrid_gp", seed=s, gp_restarts=1)
        problem = RobustProblem(predictor, ISS_NOMINAL, ObjectiveSettings(), task.n_compositions, 8, seed=s + 2)
        rec = recommend(problem, TRACKED_OBJECTIVE, budget=task.budget, seed=s + 3).params
        rows.append(
            {
                "seed": task.seed_index,
                "strategy": task.strategy,
                "round": round_index,
                "n_jobs": len(data),
                "n_strength_rows": int(data.strength_rows.sum()),
                "cumulative_cost": cumulative_cost,
                **true_value(rec, oracle),
                **{f"x_{i}": float(v) for i, v in enumerate(rec)},
            }
        )
        if round_index == task.rounds:
            break
        if task.strategy == "sobol":
            sl = slice(round_index * task.batch_size, (round_index + 1) * task.batch_size)
            comp, moist, params = sobol_comp[sl], sobol_moist[sl], sobol_params[sl]
        else:
            comp, moist, params, _ = select_experiments(
                predictor,
                ISS_NOMINAL,
                task.batch_size,
                task.strategy,
                rng,
                recommended=rec,
                n_candidates=task.candidates,
                n_replicates=task.n_replicates,
            )
        cumulative_cost += float(np.sum(experiment_cost(params)))
        new = _new_jobs(comp, moist, params, rng, task.n_replicates, len(data))
        data = _concat(data, new)
    return rows


def oracle_reference(budget: int, base_seed: int) -> dict[str, float]:
    """Best true score found by optimising the oracle directly (regret reference)."""
    oracle = RobustProblem(OraclePredictor(), ISS_NOMINAL, ObjectiveSettings(), 512, 16, seed=seed_for(base_seed, 97))
    best = recommend(oracle, TRACKED_OBJECTIVE, budget=budget, seed=seed_for(base_seed, 96)).params
    return {**true_value(best, oracle), **{name: float(v) for name, v in zip(space.PARAM_NAMES, best, strict=True)}}


def active_learning(
    strategies: list[str],
    seeds: int,
    base_seed: int,
    initial_jobs: int,
    rounds: int,
    batch_size: int,
    candidates: int,
    n_replicates: int,
    budget: int,
    n_compositions: int,
    oracle_budget: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run E3; regret is measured against the oracle optimum of the tracked objective."""
    reference = oracle_reference(oracle_budget, base_seed)
    tasks = [
        ALTask(i, st, base_seed, initial_jobs, rounds, batch_size, candidates, n_replicates, budget, n_compositions)
        for st in strategies
        for i in range(seeds)
    ]
    frame = pd.DataFrame([row for part in parallel_map(run_al_task, tasks, n_jobs) for row in part])
    best = max(reference["true_score"], float(frame["true_score"].max()))
    frame["regret"] = best - frame["true_score"]
    frame["regret_expected_utility"] = reference["true_expected_utility"] - frame["true_expected_utility"]
    return frame, {**reference, "best_score_reference": best}
