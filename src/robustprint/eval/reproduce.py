"""Run every experiment of a configuration and write results/<config-name>/."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robustprint.config import ExperimentConfig
from robustprint.data.public import DEFAULT_PATH as KAGGLE_PATH
from robustprint.eval import analyses
from robustprint.eval.active_learning import active_learning
from robustprint.eval.common import campaign, read_csv, seed_for, write_csv, write_json
from robustprint.eval.model_comparison import model_comparison
from robustprint.eval.optimisation import Variant, compare_to_main, run_variants, summarise
from robustprint.optimize.objectives import ObjectiveSettings, RobustProblem
from robustprint.optimize.robust import recommend
from robustprint.physics.simulator import ISS_NOMINAL
from robustprint.pipeline import train_predictor
from robustprint.uncertainty.propagation import OraclePredictor

ABLATION_OBJECTIVES = ("nominal", "expected", "cvar")


def ablation_variants(cfg: ExperimentConfig) -> list[Variant]:
    """Variants of E1b, built from the configuration (METHODS.md section 8.1)."""
    a = cfg.ablations
    o = ABLATION_OBJECTIVES
    tb = cfg.optimisation.train_batches
    variants = [Variant(f"model_{m}", model=m, objectives=o, train_batches=tb) for m in a.models]
    variants += [Variant(f"train_{n}", train_batches=n, objectives=o) for n in a.train_batches]
    variants.append(
        Variant("no_composition_propagation", believed_concentration_factor=1e5, objectives=o, train_batches=tb)
    )
    variants.append(Variant("no_epistemic", include_epistemic=False, objectives=o, train_batches=tb))
    variants += [
        Variant(f"told_concentration_x{f:g}", believed_concentration_factor=f, objectives=o, train_batches=tb)
        for f in a.concentration_factors
    ]
    variants.append(Variant("estimated_feedstock", believed_feedstock="estimated", objectives=o, train_batches=tb))
    variants.append(Variant("true_spread_x2", true_concentration_factor=0.5, objectives=o, train_batches=tb))
    variants += [Variant(f"label_noise_x{f:g}", noise_factor=f, objectives=o, train_batches=tb) for f in a.noise_levels]
    variants.append(Variant("outliers_5pct", outlier_fraction=0.05, objectives=o, train_batches=tb))
    variants.append(
        Variant(
            "ood_deploy_shifted",
            deploy_feedstock="iss_shifted",
            believed_feedstock="iss_nominal",
            objectives=o,
            train_batches=tb,
        )
    )
    variants += [
        Variant(f"ensemble_{k}", model="deep_ensemble", n_members=k, objectives=o, train_batches=tb)
        for k in a.ensemble_members
    ]
    return variants


def _git_state() -> dict[str, object]:
    """Commit hash and whether tracked source files had uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "src", "configs"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"commit": head, "source_dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "source_dirty": None}


def _versions() -> dict[str, str]:
    pkgs = ("numpy", "scipy", "pandas", "scikit-learn", "torch", "cma", "matplotlib")
    return {"python": platform.python_version(), **{p: version(p) for p in pkgs}}


STAGES = ("e2", "e1", "e1b", "e3", "e4", "e5", "e6", "e7")


def run_all(cfg: ExperimentConfig, out: Path, log: bool = True, stages: tuple[str, ...] = STAGES) -> dict[str, object]:
    """Run the selected experiment stages and write raw outputs, ``summary.json`` and ``manifest.json``.

    Every stage is seeded independently from ``cfg.seed``, so running stages separately gives
    the same files as running them together. The manifest records the git state of each stage.
    """
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stage(s) {sorted(unknown)}; choose from {STAGES}.")
    out.mkdir(parents=True, exist_ok=True)
    git = _git_state()
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    timings: dict[str, float] = dict(previous.get("timings_s", {}))
    stage_git: dict[str, object] = dict(previous.get("stage_git", {}))
    seed = cfg.seed

    def stamp(name: str, t0: float) -> None:
        timings[name] = round(time.perf_counter() - t0, 1)
        stage_git[name] = git
        if log:
            print(f"[robustprint] {name} done in {timings[name]:.1f}s", flush=True)

    o = cfg.optimisation
    common: dict[str, Any] = dict(
        base_seed=seed,
        settings_per_batch=cfg.settings_per_batch,
        n_replicates=cfg.n_replicates,
        deploy_batches=o.deploy_batches,
        n_compositions=o.n_compositions,
        n_noise=o.n_noise,
        budget=o.budget,
        oracle_compositions=o.oracle_compositions,
        oracle_budget=o.oracle_budget,
        n_jobs=cfg.n_jobs,
    )

    if "e2" in stages:
        t0 = time.perf_counter()
        m = cfg.models
        e2 = model_comparison(
            m.models,
            m.batch_sizes,
            m.seeds,
            m.folds,
            m.test_batches,
            cfg.settings_per_batch,
            cfg.n_replicates,
            seed,
            cfg.n_jobs,
        )
        for key, frame in e2.items():
            write_csv(out / "e2_models" / f"{key}.csv", frame)
        stamp("E2 model comparison", t0)

    if "e1" in stages:
        t0 = time.perf_counter()
        main = run_variants(
            [Variant("main", model=o.model, train_batches=o.train_batches)],
            seeds=o.seeds,
            oracle_objectives=("nominal", "expected", "quantile", "cvar", "chance"),
            **common,
        )
        write_csv(out / "e1_optimisation" / "per_seed.csv", main)
        summary, tests = summarise(main)
        write_csv(out / "e1_optimisation" / "summary.csv", summary)
        write_csv(out / "e1_optimisation" / "tests.csv", tests)
        _, oracle_tests = summarise(main, reference="oracle_nominal", oracle_methods=True)
        write_csv(out / "e1_optimisation" / "tests_oracle_exploratory.csv", oracle_tests)
        stamp("E1 robust vs nominal", t0)

    if "e1b" in stages:
        t0 = time.perf_counter()
        ablations = run_variants(ablation_variants(cfg), seeds=cfg.ablations.seeds, **common)
        write_csv(out / "e1b_ablations" / "per_seed.csv", ablations)
        ab_summary, ab_tests = summarise(ablations)
        write_csv(out / "e1b_ablations" / "summary.csv", ab_summary)
        write_csv(out / "e1b_ablations" / "tests.csv", ab_tests)
        main_frame = read_csv(out / "e1_optimisation" / "per_seed.csv")
        write_csv(out / "e1b_ablations" / "vs_main_paired.csv", compare_to_main(main_frame, ablations))
        stamp("E1b ablations", t0)

    if "e3" in stages:
        t0 = time.perf_counter()
        a = cfg.active
        al, al_ref = active_learning(
            a.strategies,
            a.seeds,
            seed,
            a.initial_jobs,
            a.rounds,
            a.batch_size,
            a.candidates,
            cfg.n_replicates,
            budget=max(o.budget // 2, 200),
            n_compositions=32,
            oracle_budget=o.oracle_budget,
            n_jobs=cfg.n_jobs,
        )
        write_csv(out / "e3_active_learning" / "per_round.csv", al)
        write_json(out / "e3_active_learning" / "reference.json", al_ref)
        stamp("E3 active learning", t0)

    if {"e4", "e5", "e6"} & set(stages):
        arrays = campaign(ISS_NOMINAL, o.train_batches, cfg.settings_per_batch, seed_for(seed, 20), cfg.n_replicates)
        hybrid = train_predictor(arrays, "hybrid_gp", seed=seed_for(seed, 21))

    if "e4" in stages:
        t0 = time.perf_counter()
        ensemble = train_predictor(arrays, "deep_ensemble", seed=seed_for(seed, 21))
        prop = pd.concat(
            [
                analyses.propagation_comparison(
                    p, ISS_NOMINAL, cfg.propagation_points, seed, label=name, repeats=cfg.propagation_repeats
                )
                for name, p in (("hybrid_gp", hybrid), ("deep_ensemble", ensemble), ("ground_truth", OraclePredictor()))
            ],
            ignore_index=True,
        )
        write_csv(out / "e4_propagation" / "errors.csv", prop)
        problem = RobustProblem(
            hybrid, ISS_NOMINAL, ObjectiveSettings(), o.n_compositions, o.n_noise, seed=seed_for(seed, 22)
        )
        recs = {
            obj: recommend(problem, obj, budget=o.budget, seed=seed_for(seed, 23)).params for obj in ("nominal", "cvar")
        }
        decomp = pd.concat(
            [
                analyses.decomposition_at(p, recs, ISS_NOMINAL, seed_for(seed, 24), name)
                for name, p in (("hybrid_gp", hybrid), ("deep_ensemble", ensemble), ("ground_truth", OraclePredictor()))
            ],
            ignore_index=True,
        )
        write_csv(out / "e4_propagation" / "decomposition.csv", decomp)
        stamp("E4 propagation", t0)

    if "e5" in stages:
        t0 = time.perf_counter()
        oracle = analyses.oracle_reference_settings(ISS_NOMINAL, seed, o.oracle_budget)
        sens = analyses.sensitivity_analysis(ISS_NOMINAL, cfg.sobol_samples, seed, hybrid, oracle["cvar"])
        for key, frame in sens.items():
            write_csv(out / "e5_sensitivity" / f"{key}.csv", frame)
        stamp("E5 sensitivity", t0)

    if "e6" in stages:
        t0 = time.perf_counter()
        pareto = analyses.pareto_front(
            hybrid, ISS_NOMINAL, cfg.pareto_budgets, seed, o.n_compositions, o.budget, o.deploy_batches
        )
        write_csv(out / "e6_pareto" / "front.csv", pareto)
        stamp("E6 Pareto", t0)

    if "e7" in stages:
        t0 = time.perf_counter()
        public = analyses.public_benchmark(KAGGLE_PATH, seed)
        if public is not None:
            for key, frame in public.items():
                frame.to_csv(out / f"e7_public_{key}.csv", index=False, float_format="%.6g")
        stamp("E7 public data", t0)

    key_numbers = summary_numbers(out)
    write_json(out / "summary.json", key_numbers)
    write_json(
        out / "manifest.json",
        {
            "config": cfg.model_dump(),
            "versions": _versions(),
            "timings_s": timings,
            "stage_git": stage_git,
            "public_data_available": (out / "e7_public_benchmark.csv").exists(),
            "objective_settings": asdict(ObjectiveSettings()),
        },
    )
    return key_numbers


def _group_means(frame: pd.DataFrame, keys: list[str], cols: list[str], digits: int = 4) -> dict[str, object]:
    """Mean of ``cols`` per group, keyed by 'key1|key2'."""
    table = frame.groupby(keys)[cols].mean().round(digits)
    return {
        "|".join(map(str, idx if isinstance(idx, tuple) else (idx,))): row.to_dict() for idx, row in table.iterrows()
    }


def summary_numbers(out: Path) -> dict[str, object]:
    """Collect the headline numbers quoted in the README from the raw result files.

    Sections whose input files do not exist (stages not run) are left out.
    """
    res: dict[str, object] = {}

    def have(*parts: str) -> bool:
        return all((out / p).exists() for p in parts)

    if have("e1_optimisation/summary.csv", "e1_optimisation/tests.csv"):
        s = read_csv(out / "e1_optimisation" / "summary.csv")
        res["e1_by_method"] = {f"{r.method}|{r.metric}": [r.mean, r.ci_low, r.ci_high] for r in s.itertuples()}
        res["e1_tests"] = read_csv(out / "e1_optimisation" / "tests.csv").to_dict(orient="records")
    if have("e1_optimisation/tests_oracle_exploratory.csv"):
        res["e1_oracle_tests_exploratory"] = read_csv(out / "e1_optimisation" / "tests_oracle_exploratory.csv").to_dict(
            orient="records"
        )
    if have("e2_models/metrics.csv", "e2_models/conformal.csv", "e2_models/classifier.csv"):
        e2 = read_csv(out / "e2_models" / "metrics.csv")
        res["e2_means"] = _group_means(
            e2,
            ["n_batches", "split", "model"],
            ["n_train_rows", "rmse_MPa", "nll_log", "crps_log", "qce", "coverage90"],
        )
        res["e2_conformal"] = _group_means(
            read_csv(out / "e2_models" / "conformal.csv"),
            ["n_batches", "split", "model"],
            ["coverage90", "width90_log"],
        )
        res["e2_classifier"] = _group_means(
            read_csv(out / "e2_models" / "classifier.csv"),
            ["n_batches", "split", "classifier"],
            ["brier", "ece", "base_rate"],
        )
    if have("e1b_ablations/per_seed.csv"):
        res["e1b_by_variant"] = _group_means(
            read_csv(out / "e1b_ablations" / "per_seed.csv"),
            ["variant", "method"],
            ["expected_utility", "failure_rate", "q05_strength", "regret_expected_utility"],
        )
    if have("e3_active_learning/per_round.csv"):
        al = read_csv(out / "e3_active_learning" / "per_round.csv")
        last = al[al["round"] == al["round"].max()]
        res["e3_final"] = _group_means(
            last, ["strategy"], ["regret", "regret_expected_utility", "cumulative_cost", "n_jobs", "n_strength_rows"]
        )
    if have("e4_propagation/errors.csv", "e4_propagation/decomposition.csv"):
        prop = read_csv(out / "e4_propagation" / "errors.csv")
        prop["abs_err_sd"] = np.abs(prop["err_sd"])
        prop["abs_err_mean"] = np.abs(prop["err_mean"])
        prop["rel_err_sd"] = np.abs(prop["err_sd"]) / prop["ref_sd"]
        res["e4_propagation"] = _group_means(
            prop, ["model", "method"], ["abs_err_mean", "abs_err_sd", "rel_err_sd", "seconds"], 6
        )
        res["e4_decomposition"] = (
            read_csv(out / "e4_propagation" / "decomposition.csv").round(4).to_dict(orient="records")
        )
    if have("e5_sensitivity/global.csv"):
        res["e5_global"] = read_csv(out / "e5_sensitivity" / "global.csv").round(4).to_dict(orient="records")
        res["e5_components"] = read_csv(out / "e5_sensitivity" / "components.csv").round(4).to_dict(orient="records")
    if have("e6_pareto/front.csv"):
        res["e6_pareto"] = read_csv(out / "e6_pareto" / "front.csv").round(4).to_dict(orient="records")
    return res
