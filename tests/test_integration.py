"""End-to-end: the CLI and a tiny reproduction run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from robustprint.cli import app
from robustprint.config import load_config
from robustprint.eval.common import read_csv
from robustprint.eval.reproduce import ablation_variants, run_all
from robustprint.viz.plots import all_figures

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def test_cli_workflow(tmp_path: Path) -> None:
    data = tmp_path / "campaign.csv"
    result = runner.invoke(app, ["simulate", "--out", str(data), "--batches", "12", "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" in result.output
    assert runner.invoke(app, ["validate", str(data)]).exit_code == 0
    model = tmp_path / "model.pkl"
    result = runner.invoke(app, ["train", str(data), "--out", str(model), "--model", "gp"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["predict", str(model), "--settings", "nozzle_temp_C=265,print_speed_mm_s=20",
                                 "--samples", "256"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["interval90_MPa"][0] < payload["median_strength_MPa"] < payload["interval90_MPa"][1]
    rec = tmp_path / "rec.json"
    result = runner.invoke(app, ["optimize", str(model), "--objective", "cvar", "--budget", "120",
                                 "--from-data", str(data), "--out", str(rec)])
    assert result.exit_code == 0, result.output
    assert json.loads(rec.read_text())["objective"] == "cvar"
    stem = tmp_path / "next"
    result = runner.invoke(app, ["recommend-experiments", str(model), "--n", "3", "--out", str(stem),
                                 "--mean", "0.45,0.25,0.15,0.08,0.07", "--concentration", "35"])
    assert result.exit_code == 0, result.output
    assert stem.with_suffix(".csv").exists() and "Next experiments" in stem.with_suffix(".md").read_text()


def test_cli_rejects_bad_input(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("experiment_id\nE1\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1 and "Missing required column" in result.output
    assert runner.invoke(app, ["simulate", "--feedstock", "moon", "--out", str(tmp_path / "x.csv")]).exit_code != 0


def test_example_csv_validates() -> None:
    result = runner.invoke(app, ["validate", str(ROOT / "data" / "examples" / "example_experiments.csv")])
    assert result.exit_code == 0, result.output


def test_configs_load_and_variants() -> None:
    full = load_config(ROOT / "configs" / "full.yaml")
    assert full.optimisation.seeds >= 30
    names = [v.name for v in ablation_variants(full)]
    assert len(names) == len(set(names))
    assert {"no_composition_propagation", "no_epistemic", "outliers_5pct", "ood_deploy_shifted"} <= set(names)


@pytest.mark.slow
def test_tiny_reproduction_end_to_end(tmp_path: Path) -> None:
    cfg_dict = yaml.safe_load((ROOT / "configs" / "fast.yaml").read_text())
    cfg_dict.update(name="tiny", n_jobs=1, sobol_samples=64, propagation_points=1, propagation_repeats=1,
                    pareto_budgets=[2.0])
    cfg_dict["models"].update(models=["physics_rom", "gp"], batch_sizes=[20], folds=2, test_batches=6)
    cfg_dict["optimisation"].update(seeds=2, train_batches=20, deploy_batches=100, budget=80,
                                    oracle_compositions=32, oracle_budget=80, n_compositions=16)
    cfg_dict["ablations"].update(seeds=2, concentration_factors=[])
    cfg_dict["active"].update(seeds=1, rounds=1, candidates=32)
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg_dict))
    out = tmp_path / "results"
    summary = run_all(load_config(path), out, log=False)
    assert "e1_by_method" in summary and "e4_decomposition" in summary
    per_seed = read_csv(out / "e1_optimisation" / "per_seed.csv")
    assert set(per_seed["method"]) >= {"nominal", "cvar", "oracle_expected"}
    assert (out / "manifest.json").exists()
    assert (out / "e1_optimisation" / "per_seed.csv").read_text().startswith("# SYNTHETIC DATA")
    figures = all_figures(out)
    assert len(figures) == 20 and all(p.exists() for p in figures)
