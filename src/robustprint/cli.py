"""Command-line interface: ``robustprint --help``."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from robustprint import space
from robustprint.data.loaders import estimate_feedstock, load_experiments
from robustprint.data.synthetic import DESIGNS, generate_dataset
from robustprint.data.validation import validate_csv
from robustprint.physics.composition import closure
from robustprint.physics.materials import DEFAULT_LIBRARY
from robustprint.physics.simulator import FEEDSTOCKS, Feedstock

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Uncertainty-aware print optimisation for recycled, variable-composition feedstock.",
)

PICKLE_WARNING = "Only load model files you created yourself: they are Python pickles."


def _parse_vector(text: str, n: int, what: str) -> np.ndarray:
    values = np.array([float(v) for v in text.split(",")])
    if values.size != n:
        raise typer.BadParameter(f"{what} needs {n} comma-separated numbers, got {values.size}.")
    return values


def _parse_settings(text: str | None) -> np.ndarray:
    x = space.defaults()
    if not text:
        return x
    for item in text.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in space.PARAM_NAMES:
            raise typer.BadParameter(f"Unknown setting {key!r}; valid: {', '.join(space.PARAM_NAMES)}")
        x[space.PARAM_NAMES.index(key)] = float(value)
    return x


def _feedstock(name: str | None, mean: str | None, concentration: float | None, data: Path | None) -> Feedstock:
    if data is not None:
        arrays, _ = load_experiments(str(data))
        fs = estimate_feedstock(arrays, name=f"estimated from {data.name}")
    elif mean is not None:
        mu = closure(_parse_vector(mean, DEFAULT_LIBRARY.n, "--mean"))
        fs = Feedstock("custom", tuple(mu.tolist()), concentration or 35.0)
    else:
        key = name or "iss_nominal"
        if key not in FEEDSTOCKS:
            raise typer.BadParameter(f"Unknown feedstock {key!r}; choose one of {', '.join(FEEDSTOCKS)}.")
        fs = FEEDSTOCKS[key]
    if concentration is not None:
        fs = replace(fs, concentration=concentration)
    return fs


def _load_model(path: Path):  # type: ignore[no-untyped-def]
    with open(path, "rb") as fh:
        return pickle.load(fh)


FeedstockOpt = Annotated[str | None, typer.Option(help=f"Named feedstock: {', '.join(FEEDSTOCKS)}.")]
MeanOpt = Annotated[str | None, typer.Option(help="Mean composition, comma-separated in library order.")]
ConcOpt = Annotated[float | None, typer.Option(help="Dirichlet concentration (higher = less batch spread).")]
DataOpt = Annotated[Path | None, typer.Option("--from-data", help="Estimate the feedstock from batches in a CSV.")]


@app.command()
def validate(path: Path) -> None:
    """Validate an experiment CSV and print every problem found."""
    _, report = validate_csv(str(path))
    typer.echo(report.summary())
    raise typer.Exit(code=0 if report.ok else 1)


@app.command()
def simulate(
    out: Annotated[Path, typer.Option(help="Output CSV.")] = Path("data/synthetic/campaign.csv"),
    feedstock: FeedstockOpt = "iss_nominal",
    batches: int = 60,
    settings_per_batch: int = 4,
    replicates: int = 3,
    design: Annotated[str, typer.Option(help=f"One of {', '.join(DESIGNS)}.")] = "sobol",
    seed: int = 0,
) -> None:
    """Generate a SYNTHETIC campaign in the experiment schema."""
    fs = _feedstock(feedstock, None, None, None)
    frame, _ = generate_dataset(
        fs, batches, settings_per_batch, np.random.default_rng(seed), n_replicates=replicates, design=design
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    typer.echo(f"Wrote {len(frame)} SYNTHETIC jobs ({int(frame['print_success'].sum())} successful) to {out}")


@app.command()
def train(
    data: Path,
    out: Annotated[Path, typer.Option(help="Where to save the fitted model (pickle).")] = Path("models/model.pkl"),
    model: Annotated[str, typer.Option(help="Strength model name.")] = "hybrid_gp",
    seed: int = 0,
) -> None:
    """Fit a strength model and a print-success classifier to a validated CSV."""
    from robustprint.pipeline import train_predictor

    arrays, report = load_experiments(str(data))
    for w in report.warnings:
        typer.echo(f"WARNING: {w}")
    predictor = train_predictor(arrays, model, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(predictor, fh)
    typer.echo(f"Trained {model} on {int(arrays.strength_rows.sum())} strength rows / {len(arrays)} jobs -> {out}")
    typer.echo(PICKLE_WARNING)


@app.command()
def predict(
    model_path: Path,
    settings: Annotated[str | None, typer.Option(help="e.g. 'nozzle_temp_C=265,print_speed_mm_s=20'.")] = None,
    feedstock: FeedstockOpt = None,
    mean: MeanOpt = None,
    concentration: ConcOpt = None,
    from_data: DataOpt = None,
    samples: int = 4096,
    seed: int = 0,
) -> None:
    """Predictive distribution of strength for the next (uncharacterised) batch."""
    from robustprint.uncertainty.propagation import monte_carlo, sample_feedstock

    predictor = _load_model(model_path)
    fs = _feedstock(feedstock, mean, concentration, from_data)
    x = _parse_settings(settings)
    res = monte_carlo(predictor, x, sample_feedstock(fs, samples, np.random.default_rng(seed)))
    sd = float(np.sqrt(res.var))
    payload = {
        "settings": space.as_dict(x),
        "feedstock": {"mean": list(fs.mean), "concentration": fs.concentration},
        "median_strength_MPa": float(np.exp(res.mean)),
        "interval90_MPa": [float(np.exp(res.mean - 1.645 * sd)), float(np.exp(res.mean + 1.645 * sd))],
        "p_success": res.p_success,
        "variance_shares": res.shares(),
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def optimize(
    model_path: Path,
    objective: Annotated[str, typer.Option(help="nominal, expected, quantile, cvar or chance.")] = "cvar",
    feedstock: FeedstockOpt = None,
    mean: MeanOpt = None,
    concentration: ConcOpt = None,
    from_data: DataOpt = None,
    budget: int = 800,
    seed: int = 0,
    out: Annotated[Path | None, typer.Option(help="Optional JSON output.")] = None,
) -> None:
    """Recommend print settings that maximise a (robust) objective."""
    from robustprint.optimize.objectives import RobustProblem
    from robustprint.optimize.robust import recommend

    predictor = _load_model(model_path)
    fs = _feedstock(feedstock, mean, concentration, from_data)
    problem = RobustProblem(predictor, fs, seed=seed)
    res = recommend(problem, objective, budget=budget, seed=seed)
    stats = {k: float(v[0]) for k, v in problem.statistics(res.params[None, :]).items()}
    payload = {
        "objective": objective,
        "settings": space.as_dict(res.params),
        "predicted": stats,
        "note": "Predictions come from the fitted model; verify with a small confirmation print run.",
    }
    text = json.dumps(payload, indent=2)
    typer.echo(text)
    if out is not None:
        out.write_text(text + "\n", encoding="utf-8")


@app.command("recommend-experiments")
def recommend_experiments(
    model_path: Path,
    n: Annotated[int, typer.Option(help="Number of print jobs to propose.")] = 4,
    strategy: Annotated[
        str, typer.Option(help="targeted_ivr, cost_aware_ivr, max_variance, sobol or random.")
    ] = "targeted_ivr",
    feedstock: FeedstockOpt = None,
    mean: MeanOpt = None,
    concentration: ConcOpt = None,
    from_data: DataOpt = None,
    seed: int = 0,
    out: Annotated[Path, typer.Option(help="Output stem; writes .csv and .md.")] = Path("next_experiments"),
) -> None:
    """Propose the next most informative print jobs (CSV + Markdown table)."""
    from robustprint.active.planner import experiments_table, select_experiments, table_to_markdown
    from robustprint.optimize.objectives import RobustProblem
    from robustprint.optimize.robust import recommend

    predictor = _load_model(model_path)
    fs = _feedstock(feedstock, mean, concentration, from_data)
    rec = recommend(RobustProblem(predictor, fs, seed=seed), "cvar", seed=seed).params
    rng = np.random.default_rng(seed)
    comp, moist, params, scores = select_experiments(predictor, fs, n, strategy, rng, recommended=rec)
    table = experiments_table(predictor, comp, moist, params, scores, strategy)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    header = (
        f"# Next experiments ({strategy})\n\nGenerated {time.strftime('%Y-%m-%d')} by `robustprint "
        "recommend-experiments`. Blend each target composition as closely as you can, record the *measured* "
        "composition, print the settings with 3 tensile specimens, and append the results to your CSV.\n\n"
    )
    out.with_suffix(".md").write_text(header + table_to_markdown(table) + "\n", encoding="utf-8")
    typer.echo(table.to_string(index=False))
    typer.echo(f"\nWrote {out.with_suffix('.csv')} and {out.with_suffix('.md')}")


@app.command()
def reproduce(
    fast: Annotated[bool, typer.Option("--fast/--full", help="Fast smoke run (< 3 min) or full run.")] = True,
    config: Annotated[Path | None, typer.Option(help="Explicit YAML config (overrides --fast/--full).")] = None,
    results: Annotated[Path, typer.Option(help="Results root folder.")] = Path("results"),
    figures: bool = True,
) -> None:
    """Run all experiments of a configuration and write results/<name>/."""
    from robustprint.config import load_config
    from robustprint.eval.reproduce import run_all

    path = config or Path("configs") / ("fast.yaml" if fast else "full.yaml")
    cfg = load_config(path)
    out = results / cfg.name
    t0 = time.perf_counter()
    run_all(cfg, out)
    if figures:
        from robustprint.viz.plots import all_figures

        all_figures(out)
    typer.echo(f"Finished {cfg.name} in {time.perf_counter() - t0:.0f}s -> {out}")


@app.command("figures")
def figures_cmd(results_dir: Path) -> None:
    """Re-render all figures from saved results."""
    from robustprint.viz.plots import all_figures

    for p in all_figures(results_dir):
        typer.echo(str(p))


@app.command("public-data")
def public_data() -> None:
    """Download the optional Kaggle FDM dataset (licence: unknown; not redistributed)."""
    from robustprint.data.public import KAGGLE_PAGE, download_kaggle_3dprinter

    path = download_kaggle_3dprinter()
    typer.echo(f"Downloaded {path} (source: {KAGGLE_PAGE}; licence listed as 'Unknown' - do not redistribute).")


if __name__ == "__main__":
    app()
