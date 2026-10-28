"""Figures generated from saved result files (PNG + SVG, colour-blind-safe palette).

Every figure built from simulator output carries a visible "SYNTHETIC DATA" stamp.
Colours follow a validated categorical palette in a fixed order; text uses neutral ink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robustprint.eval.common import read_csv

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
STAMP = "SYNTHETIC DATA — simulator output, not experimental results"

METHOD_ORDER = ["nominal", "expected", "quantile", "cvar", "chance"]
METHOD_LABEL = {
    "nominal": "Nominal",
    "expected": "Expected",
    "quantile": "5% quantile",
    "cvar": "CVaR(10%)",
    "chance": "Chance-constr.",
    "oracle_expected": "Oracle (expected)",
    "oracle_cvar": "Oracle (CVaR)",
}
METHOD_COLOR = {m: PALETTE[i] for i, m in enumerate(METHOD_ORDER)}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "svg.hashsalt": "robustprint",
        }
    )


def _save(fig: plt.Figure, out: Path, name: str, synthetic: bool = True) -> list[Path]:
    if synthetic:
        fig.text(0.995, 0.005, STAMP, ha="right", va="bottom", fontsize=7.5, color=INK_2, style="italic")
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.svg"]
    fig.savefig(paths[0], dpi=160, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(paths[1], bbox_inches="tight", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    return paths


def _ci(values: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boots = values[idx].mean(axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def hero(res: Path, out: Path) -> list[Path]:
    """Summary of E1: nominal vs composition-aware objectives, learned model vs perfect model."""
    _style()
    df = read_csv(res / "e1_optimisation" / "per_seed.csv")
    rng = np.random.default_rng(0)
    methods = ["nominal", "expected", "cvar"]
    groups = [("learned model\n(240 synthetic jobs)", ""), ("perfect model\n(ground-truth oracle)", "oracle_")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    panels = [
        ("failure_rate", "Failed prints on unseen batches", "failure rate (%)", 100.0),
        ("q05_strength", "Weakest 5 % of printed parts", "5th-percentile strength (MPa)", 1.0),
        ("expected_utility", "Expected utility", "mean strength, failed part = 0 (MPa)", 1.0),
    ]
    width = 0.26
    for ax, (metric, title, ylabel, scale) in zip(axes, panels, strict=True):
        for gi, (_, prefix) in enumerate(groups):
            for mi, m in enumerate(methods):
                v = df.loc[df["method"] == prefix + m, metric].to_numpy() * scale
                if v.size == 0:
                    continue
                lo, hi = _ci(v, rng)
                x = gi + (mi - 1) * width
                ax.bar(
                    x,
                    v.mean(),
                    width=width,
                    color=METHOD_COLOR[m],
                    edgecolor=SURFACE,
                    linewidth=2,
                    label=METHOD_LABEL[m] if gi == 0 else None,
                )
                ax.errorbar(x, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], color=INK, capsize=2.5, lw=1.1)
                ax.text(x, hi, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=3,
        fontsize=8.5,
        title="objective",
        title_fontsize=8.5,
        bbox_to_anchor=(0.995, 0.995),
    )
    n_seeds = df["seed"].nunique()
    fig.suptitle(
        f"Robust vs nominal settings on unseen feedstock batches ({n_seeds} seeds, mean ± 95 % CI)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return _save(fig, out, "hero")


def realised_distributions(res: Path, out: Path) -> list[Path]:
    """Per-seed realised metrics by objective (dots = seeds), paired lines to nominal."""
    _style()
    df = read_csv(res / "e1_optimisation" / "per_seed.csv")
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, metric, label in zip(
        axes,
        ["mean_strength", "q05_strength", "failure_rate"],
        ["mean strength of printed parts (MPa)", "5th percentile strength (MPa)", "failure rate"],
        strict=True,
    ):
        wide = df.pivot_table(index="seed", columns="method", values=metric)
        for seed_row in wide.itertuples(index=False):
            ax.plot(range(len(methods)), [getattr(seed_row, m) for m in methods], color=GRID, lw=0.8, zorder=1)
        for i, m in enumerate(methods):
            v = wide[m].to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(v))
            ax.scatter(i + jitter, v, s=14, color=METHOD_COLOR[m], zorder=2, edgecolor=SURFACE, linewidth=0.5)
            ax.hlines(np.median(v), i - 0.25, i + 0.25, color=INK, lw=2, zorder=3)
        ax.set_xticks(range(len(methods)), [METHOD_LABEL[m] for m in methods], rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Realised outcomes per seed (grey lines join the same seed; black bar = median)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(fig, out, "realised_per_seed")


def calibration(res: Path, out: Path) -> list[Path]:
    """Quantile calibration curves (grouped CV) and interval coverage vs nominal level."""
    _style()
    cal = read_csv(res / "e2_models" / "calibration.csv")
    cov = read_csv(res / "e2_models" / "coverage.csv")
    size = int(cal["n_batches"].max())
    models = list(cal["model"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sub = cal[cal["n_batches"] == size].groupby(["model", "level"])["observed"].mean().reset_index()
    for i, m in enumerate(models):
        d = sub[sub["model"] == m]
        axes[0].plot(d["level"], d["observed"], color=PALETTE[i % 8], label=m, lw=1.6)
    axes[0].plot([0, 1], [0, 1], color=INK_2, ls="--", lw=1)
    axes[0].set(
        xlabel="nominal quantile level",
        ylabel="observed frequency",
        title=f"Quantile calibration (grouped CV, {size} batches)",
    )
    for ax, split, title in (
        (axes[1], "new_batches", "Coverage on new batches"),
        (axes[2], "shifted_ood", "Coverage on shifted (OOD) feedstock"),
    ):
        d = cov[(cov["n_batches"] == size) & (cov["split"] == split)].groupby(["model", "level"])["coverage"].mean()
        d = d.reset_index()
        for i, m in enumerate(models):
            dm = d[d["model"] == m]
            ax.plot(dm["level"], dm["coverage"], color=PALETTE[i % 8], marker="o", ms=4, lw=1.4, label=m)
        ax.plot([0.5, 0.95], [0.5, 0.95], color=INK_2, ls="--", lw=1)
        ax.set(xlabel="nominal interval level", ylabel="empirical coverage", title=title)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "calibration")


def learning_curves(res: Path, out: Path) -> list[Path]:
    """CRPS and RMSE vs training size (grouped CV), mean ± range over seeds."""
    _style()
    m = read_csv(res / "e2_models" / "metrics.csv")
    m = m[m["split"] == "grouped_cv"]
    models = list(m["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in (
        (axes[0], "crps_log", "CRPS on log strength (lower is better)"),
        (axes[1], "rmse_MPa", "RMSE (MPa)"),
    ):
        for i, name in enumerate(models):
            g = m[m["model"] == name].groupby("n_batches")[metric]
            mean, lo, hi = g.mean(), g.min(), g.max()
            ax.plot(mean.index * 4, mean.values, color=PALETTE[i % 8], marker="o", ms=4, label=name)
            ax.fill_between(mean.index * 4, lo.values, hi.values, color=PALETTE[i % 8], alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set(xlabel="training print jobs (log scale)", ylabel=label)
    axes[0].legend(fontsize=7.5)
    axes[0].set_title("Learning curves (grouped CV by batch; band = min-max over seeds)", loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "learning_curves")


def reliability(res: Path, out: Path) -> list[Path]:
    """Reliability diagram of the success classifiers (grouped CV, pooled)."""
    _style()
    r = read_csv(res / "e2_models" / "reliability.csv")
    size = int(r["n_batches"].max())
    r = r[r["n_batches"] == size]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for i, (kind, d) in enumerate(r.groupby("classifier")):
        p_bins, f_bins = [], []
        for _, b in d.groupby("bin"):
            w = b["count"].to_numpy(dtype=float)
            if w.sum() > 0:
                p_bins.append(np.average(b["mean_p"].fillna(0), weights=w))
                f_bins.append(np.average(b["freq"].fillna(0), weights=w))
        g = {"p": p_bins, "f": f_bins}
        ax.plot(g["p"], g["f"], marker="o", ms=5, color=PALETTE[i], label=str(kind))
    ax.plot([0, 1], [0, 1], color=INK_2, ls="--", lw=1)
    ax.set(xlabel="predicted P(success)", ylabel="observed success rate", title="Print-success calibration")
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, "reliability")


def active_learning_curves(res: Path, out: Path) -> list[Path]:
    """Regret of the CVaR recommendation after each round, mean ± 95 % CI over seeds."""
    _style()
    al = read_csv(res / "e3_active_learning" / "per_round.csv")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    panels = (
        ("regret", "pre-registered: regret of penalised CVaR score", "regret (MPa, incl. success penalty)"),
        ("regret_expected_utility", "regret of expected utility vs oracle", "regret (MPa)"),
    )
    for ax, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for i, (strategy, d) in enumerate(al.groupby("strategy", sort=False)):
            x = np.array(sorted(d["n_jobs"].unique()))
            mean = d.groupby("n_jobs")[metric].mean().reindex(x).to_numpy()
            cis = np.array([_ci(d.loc[d["n_jobs"] == n, metric].to_numpy(), rng) for n in x])
            ax.plot(x, mean, color=PALETTE[i], marker="o", ms=4, label=strategy)
            ax.fill_between(x, cis[:, 0], cis[:, 1], color=PALETTE[i], alpha=0.10, lw=0)
        ax.set(xlabel="print jobs run", ylabel=ylabel, title=title)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Sequential experimentation: quality of the robust recommendation (lower is better)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return _save(fig, out, "active_learning")


def pareto(res: Path, out: Path) -> list[Path]:
    """Strength vs print time (energy as marker size), predicted and realised."""
    _style()
    p = read_csv(res / "e6_pareto" / "front.csv")
    p = p[np.isfinite(p["time_budget"]) | (p["time_budget"].astype(str) == "inf")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (obj, d) in enumerate(p.groupby("objective", sort=False)):
        d = d.sort_values("print_time")
        axes[0].plot(
            d["print_time"], d["pred_cvar"], color=PALETTE[i], marker="o", ms=5, label=f"{obj}: predicted CVaR"
        )
        axes[1].plot(
            d["print_time"],
            d["cvar10_strength"],
            color=PALETTE[i],
            marker="o",
            ms=5,
            label=f"{obj}: realised CVaR(10%)",
        )
        axes[1].scatter(d["print_time"], d["expected_utility"], color=PALETTE[i], marker="s", s=22, alpha=0.6)
        for _, row in d.iterrows():
            axes[0].annotate(
                f"E={row['energy']:.1f}",
                (row["print_time"], row["pred_cvar"]),
                fontsize=6.5,
                color=INK_2,
                xytext=(3, 3),
                textcoords="offset points",
            )
    axes[0].set(
        xlabel="relative print time",
        ylabel="predicted CVaR(10%) strength (MPa)",
        title="Trade-off: robust strength vs print time",
    )
    axes[1].set(
        xlabel="relative print time",
        ylabel="realised on fresh batches (MPa)",
        title="Realised: CVaR(10%) (circles), expected utility (squares)",
    )
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "pareto")


def decomposition(res: Path, out: Path) -> list[Path]:
    """Stacked shares of predictive variance (input / epistemic / aleatoric)."""
    _style()
    d = read_csv(res / "e4_propagation" / "decomposition.csv")
    labels = [f"{r.model}\n{r.setting}" for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    left = np.zeros(len(d))
    for i, (col, name) in enumerate(
        (
            ("share_input", "feedstock composition + moisture"),
            ("share_epistemic", "epistemic (model)"),
            ("share_aleatoric", "aleatoric (noise)"),
        )
    ):
        ax.barh(labels, d[col], left=left, color=PALETTE[i], edgecolor=SURFACE, linewidth=2, label=name)
        for y, (lft, v) in enumerate(zip(left, d[col], strict=True)):
            if v > 0.08:
                ax.text(lft + v / 2, y, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left += d[col].to_numpy()
    ax.set(xlabel="share of predictive variance of log strength", xlim=(0, 1), title="Where the uncertainty comes from")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "uncertainty_decomposition")


def propagation(res: Path, out: Path) -> list[Path]:
    """Error of the predictive standard deviation vs cost for MC sizes and the unscented transform."""
    _style()
    e = read_csv(res / "e4_propagation" / "errors.csv")
    e["abs_err_sd"] = e["err_sd"].abs()
    fig, ax = plt.subplots(figsize=(5.8, 4))
    for i, (model, d) in enumerate(e.groupby("model")):
        g = d.groupby("method").agg(n=("n_evals", "first"), err=("abs_err_sd", "mean")).sort_values("n")
        mc = g[g.index.str.startswith("mc_")]
        ax.plot(mc["n"], mc["err"], color=PALETTE[i], marker="o", ms=5, label=f"{model}: Monte Carlo")
        ut = g.loc["unscented"]
        ax.scatter(
            [ut["n"]],
            [ut["err"]],
            color=PALETTE[i],
            marker="D",
            s=50,
            edgecolor=INK,
            zorder=3,
            label=f"{model}: unscented (11 pts)",
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="model evaluations per setting",
        ylabel="mean |error| of predictive SD (log strength)",
        title="Propagation accuracy vs cost",
    )
    ax.legend(fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, "propagation")


def sensitivity(res: Path, out: Path) -> list[Path]:
    """Sobol total and first-order indices of the ground truth (log strength and P(success))."""
    _style()
    s = read_csv(res / "e5_sensitivity" / "global.csv")
    s = s[s["source"] == "ground_truth"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, output, title in ((axes[0], "log_strength", "log strength"), (axes[1], "p_success", "P(success)")):
        d = s[s["output"] == output].sort_values("total")
        y = np.arange(len(d))
        ax.barh(y + 0.2, d["total"], height=0.38, color=PALETTE[0], label="total index", edgecolor=SURFACE)
        ax.barh(
            y - 0.2,
            d["first"].clip(lower=0),
            height=0.38,
            color=PALETTE[1],
            label="first-order index",
            edgecolor=SURFACE,
        )
        ax.set_yticks(y, d["input"])
        ax.set(xlabel="Sobol index", title=f"Global sensitivity of {title}")
        ax.grid(axis="y", visible=False)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "sensitivity")


def all_figures(res: Path) -> list[Path]:
    """Render every figure whose inputs exist into ``res/figures``."""
    out = res / "figures"
    makers = [
        hero,
        realised_distributions,
        calibration,
        learning_curves,
        reliability,
        active_learning_curves,
        pareto,
        decomposition,
        propagation,
        sensitivity,
    ]
    paths: list[Path] = []
    for make in makers:
        paths += make(res, out)
    return paths
