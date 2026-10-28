"""E1: robust versus nominal optimisation, deployed on fresh batches; ablations and stress tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from robustprint.data.loaders import ExperimentArrays, estimate_feedstock
from robustprint.eval.common import campaign, parallel_map, seed_for
from robustprint.eval.deployment import deploy, draw_deployment
from robustprint.eval.stats import bootstrap_ci, paired_comparison, with_holm
from robustprint.models.base import Prediction
from robustprint.optimize.objectives import OBJECTIVES, ObjectiveSettings, RobustProblem
from robustprint.optimize.robust import recommend
from robustprint.physics.composition import FloatArray
from robustprint.physics.simulator import (
    DEFAULT_CONFIG,
    FEEDSTOCKS,
    Feedstock,
    SimulatorConfig,
)
from robustprint.pipeline import train_predictor
from robustprint.uncertainty.propagation import ModelPredictor, OraclePredictor

DEPLOY_METRICS = ("expected_utility", "failure_rate", "q05_strength", "cvar10_strength", "mean_strength",
                  "p_meet_threshold")
#: Metrics where smaller is better.
LOWER_IS_BETTER = {"failure_rate"}


@dataclass(frozen=True)
class AleatoricOnly:
    """Predictor wrapper that drops epistemic variance (ablation)."""

    inner: ModelPredictor

    def predict(self, composition: FloatArray, moisture: FloatArray, params: FloatArray) -> tuple[Prediction, FloatArray]:
        """Same as the wrapped predictor with zero epistemic variance."""
        pred, p = self.inner.predict(composition, moisture, params)
        return Prediction(pred.mean, np.zeros_like(pred.var_epistemic), pred.var_aleatoric), p


@dataclass(frozen=True)
class Variant:
    """One configuration of the optimisation pipeline.

    Attributes:
        name: Identifier used in result tables.
        model: Strength model name.
        train_batches: Training batches (x settings_per_batch jobs).
        objectives: Objectives to compare.
        include_epistemic: Keep epistemic variance in the robust objectives.
        believed_concentration_factor: Multiplier on the concentration the optimiser is told
            (< 1: told more spread than reality; > 1: told less; 1e5: no propagation).
        true_concentration_factor: Multiplier on the real feedstock concentration
            (training and deployment).
        believed_feedstock: ``"true"``, ``"estimated"`` (fitted to training batches) or a
            feedstock name.
        deploy_feedstock: Feedstock name that the fresh deployment batches come from.
        noise_factor: Multiplier on specimen noise coefficients (label noise).
        outlier_fraction: Fraction of training strengths corrupted by a factor U(0.3, 0.6).
        n_members: Deep-ensemble size.
    """

    name: str
    model: str = "hybrid_gp"
    train_batches: int = 60
    objectives: tuple[str, ...] = OBJECTIVES
    include_epistemic: bool = True
    believed_concentration_factor: float = 1.0
    true_concentration_factor: float = 1.0
    believed_feedstock: str = "true"
    deploy_feedstock: str = "iss_nominal"
    noise_factor: float = 1.0
    outlier_fraction: float = 0.0
    n_members: int = 5

    def simulator(self) -> SimulatorConfig:
        """Simulator configuration implied by the noise factor."""
        f = self.noise_factor
        c = DEFAULT_CONFIG
        return replace(c, noise_cv_base=c.noise_cv_base * f, noise_cv_contaminant=c.noise_cv_contaminant * f,
                       noise_cv_fiber=c.noise_cv_fiber * f, noise_cv_porosity=c.noise_cv_porosity * f)

    def true_feedstock(self) -> Feedstock:
        """Feedstock that training batches are drawn from."""
        f = FEEDSTOCKS["iss_nominal"]
        return replace(f, concentration=f.concentration * self.true_concentration_factor)

    def deployment_feedstock(self) -> Feedstock:
        """Feedstock that deployment batches are drawn from."""
        f = FEEDSTOCKS[self.deploy_feedstock]
        return replace(f, concentration=f.concentration * self.true_concentration_factor)


@dataclass(frozen=True)
class OptTask:
    """One seed of one variant."""

    variant: Variant
    seed_index: int
    base_seed: int
    settings_per_batch: int
    n_replicates: int
    deploy_batches: int
    n_compositions: int
    n_noise: int
    budget: int
    oracle_params: dict[str, list[float]]


def _corrupt(arrays: ExperimentArrays, fraction: float, rng: np.random.Generator) -> ExperimentArrays:
    if fraction <= 0:
        return arrays
    rows = np.flatnonzero(arrays.strength_rows)
    n_bad = int(round(fraction * rows.size))
    bad = rng.choice(rows, size=n_bad, replace=False)
    strength = arrays.strength.copy()
    strength[bad] *= rng.uniform(0.3, 0.6, size=n_bad)
    return replace(arrays, strength=strength)


def believed_feedstock(variant: Variant, arrays: ExperimentArrays) -> Feedstock:
    """Feedstock distribution the optimiser integrates over."""
    if variant.believed_feedstock == "true":
        base = variant.deployment_feedstock()
    elif variant.believed_feedstock == "estimated":
        base = estimate_feedstock(arrays)
    else:
        base = FEEDSTOCKS[variant.believed_feedstock]
    factor = variant.believed_concentration_factor
    if factor >= 1e4:
        return replace(base, concentration=base.concentration * factor, moisture_log_sd=1e-6)
    return replace(base, concentration=base.concentration * factor)


def run_opt_task(task: OptTask) -> list[dict[str, object]]:
    """Train, recommend with every objective and deploy on fresh batches (one seed)."""
    v = task.variant
    cfg = v.simulator()
    s = seed_for(task.base_seed, 1, task.seed_index)
    arrays = campaign(v.true_feedstock(), v.train_batches, task.settings_per_batch, s, task.n_replicates, cfg)
    arrays = _corrupt(arrays, v.outlier_fraction, np.random.default_rng(s + 7))
    predictor = train_predictor(arrays, v.model, seed=s, n_members=v.n_members)
    pred_used = predictor if v.include_epistemic else AleatoricOnly(predictor)
    belief = believed_feedstock(v, arrays)
    settings = ObjectiveSettings()
    problem = RobustProblem(pred_used, belief, settings, task.n_compositions, task.n_noise, seed=s + 3)
    deployment = draw_deployment(v.deployment_feedstock(), task.deploy_batches, s + 11, cfg)
    base = {"variant": v.name, "seed": task.seed_index, "n_strength_rows": int(arrays.strength_rows.sum()),
            "n_jobs": len(arrays)}
    rows: list[dict[str, object]] = []
    for objective in v.objectives:
        res = recommend(problem, objective, budget=task.budget, seed=s + 5)
        stats = problem.statistics(res.params[None, :])
        rows.append({**base, "method": objective, **deploy(res.params, deployment, settings.chance_threshold_MPa, cfg),
                     **{f"pred_{k}": float(val[0]) for k, val in stats.items()},
                     **{f"x_{i}": float(val) for i, val in enumerate(res.params)}})
    for objective, params in task.oracle_params.items():
        x = np.asarray(params)
        rows.append({**base, "method": f"oracle_{objective}", **deploy(x, deployment, settings.chance_threshold_MPa, cfg),
                     **{f"x_{i}": float(val) for i, val in enumerate(x)}})
    return rows


def oracle_settings(variant: Variant, objectives: tuple[str, ...], n_compositions: int, budget: int,
                    base_seed: int) -> dict[str, list[float]]:
    """Optimise each objective on the ground truth for the deployment feedstock."""
    problem = RobustProblem(OraclePredictor(variant.simulator()), variant.deployment_feedstock(), ObjectiveSettings(),
                            n_compositions, 16, seed=seed_for(base_seed, 99))
    return {obj: recommend(problem, obj, budget=budget, seed=seed_for(base_seed, 98)).params.tolist()
            for obj in objectives}


def run_variant(variant: Variant, seeds: int, base_seed: int, settings_per_batch: int, n_replicates: int,
                deploy_batches: int, n_compositions: int, n_noise: int, budget: int, oracle_compositions: int,
                oracle_budget: int, n_jobs: int, oracle_objectives: tuple[str, ...] = ("expected", "cvar")
                ) -> pd.DataFrame:
    """All seeds of one variant, including oracle settings deployed on the same batches."""
    oracle = oracle_settings(variant, oracle_objectives, oracle_compositions, oracle_budget, base_seed)
    tasks = [OptTask(variant, i, base_seed, settings_per_batch, n_replicates, deploy_batches, n_compositions,
                     n_noise, budget, oracle) for i in range(seeds)]
    frame = pd.DataFrame([row for part in parallel_map(run_opt_task, tasks, n_jobs) for row in part])
    ref = frame[frame["method"] == "oracle_expected"].set_index("seed")["expected_utility"]
    frame["regret_expected_utility"] = frame["seed"].map(ref) - frame["expected_utility"]
    return frame


def summarise(frame: pd.DataFrame, reference: str = "nominal") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap CIs per method and Holm-corrected paired tests against ``reference``.

    The Holm family is every (method, metric) comparison within one variant.
    """
    summary_rows = []
    tests = []
    for variant, vf in frame.groupby("variant", sort=False):
        methods = [m for m in vf["method"].unique()]
        for method in methods:
            mf = vf[vf["method"] == method].sort_values("seed")
            for metric in (*DEPLOY_METRICS, "regret_expected_utility"):
                est, lo, hi = bootstrap_ci(mf[metric].to_numpy())
                summary_rows.append({"variant": variant, "method": method, "metric": metric, "mean": est,
                                     "ci_low": lo, "ci_high": hi, "n_seeds": len(mf)})
        if reference not in methods:
            continue
        ref = vf[vf["method"] == reference].sort_values("seed")
        family = []
        for method in methods:
            if method == reference or method.startswith("oracle"):
                continue
            mf = vf[vf["method"] == method].sort_values("seed")
            common = np.intersect1d(mf["seed"], ref["seed"])
            a = mf.set_index("seed").loc[common]
            b = ref.set_index("seed").loc[common]
            for metric in DEPLOY_METRICS:
                family.append(paired_comparison(a[metric].to_numpy(), b[metric].to_numpy(), metric, method, reference))
        tests += [{"variant": variant, **c.as_dict()} for c in with_holm(family)]
    return pd.DataFrame(summary_rows), pd.DataFrame(tests)
