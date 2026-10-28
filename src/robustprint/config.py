"""Experiment configuration (validated with pydantic, loaded from YAML)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelComparisonConfig(BaseModel):
    """E2: model comparison and learning curves."""

    models: list[str]
    batch_sizes: list[int] = Field(description="Numbers of training batches (x settings_per_batch jobs).")
    seeds: int = 3
    folds: int = 5
    test_batches: int = 30


class OptimisationConfig(BaseModel):
    """E1: robust versus nominal optimisation on fresh batches."""

    model: str = "hybrid_gp"
    seeds: int = 30
    train_batches: int = 60
    deploy_batches: int = 2000
    n_compositions: int = 64
    n_noise: int = 16
    budget: int = 800
    oracle_compositions: int = 256
    oracle_budget: int = 1500


class AblationConfig(BaseModel):
    """E1b: ablations and stress tests of the optimisation pipeline."""

    seeds: int = 15
    models: list[str] = ["deep_ensemble"]
    train_batches: list[int] = [30, 120]
    concentration_factors: list[float] = [0.25, 4.0]
    noise_levels: list[float] = [2.0]
    ensemble_members: list[int] = [3, 10]


class ActiveLearningConfig(BaseModel):
    """E3: sequential experimentation."""

    seeds: int = 20
    initial_jobs: int = 16
    rounds: int = 8
    batch_size: int = 4
    candidates: int = 512
    strategies: list[str] = ["random", "sobol", "max_variance", "targeted_ivr", "cost_aware_ivr"]


class ExperimentConfig(BaseModel):
    """Top-level configuration for ``robustprint reproduce``."""

    name: str
    seed: int = 20261028
    settings_per_batch: int = 4
    n_replicates: int = 3
    n_jobs: int = 4
    models: ModelComparisonConfig
    optimisation: OptimisationConfig
    ablations: AblationConfig
    active: ActiveLearningConfig
    sobol_samples: int = 2048
    propagation_points: int = 20
    propagation_repeats: int = 20
    pareto_budgets: list[float] = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate a YAML experiment configuration."""
    with open(path, encoding="utf-8") as fh:
        return ExperimentConfig.model_validate(yaml.safe_load(fh))
