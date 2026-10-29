"""Export the headline study results (results/full) to web/src/data/study.json for the Virtual Lab.

Usage: python scripts/export_web_data.py [results/full]
Everything exported is SYNTHETIC (simulator output) except the optional E7 rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from robustprint import __version__
from robustprint.eval.common import SYNTHETIC_LABEL, read_csv
from robustprint.eval.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "src" / "data" / "study.json"
METRICS = ["failure_rate", "q05_strength", "expected_utility", "mean_strength", "regret_expected_utility"]


def r(v: float, d: int = 4) -> float | None:
    """Round for JSON; NaN/inf become null."""
    return None if not np.isfinite(v) else round(float(v), d)


def main(res: Path) -> None:
    """Write study.json."""
    manifest = json.loads((res / "manifest.json").read_text())
    s = read_csv(res / "e1_optimisation" / "summary.csv")
    t = read_csv(res / "e1_optimisation" / "tests.csv")
    per = read_csv(res / "e1_optimisation" / "per_seed.csv")
    e1 = {
        "seeds": int(per["seed"].nunique()),
        "trainJobs": int(per["n_jobs"].iloc[0]),
        "meanSuccessfulJobs": r(per["n_strength_rows"].mean(), 1),
        "methods": {
            m: {
                met: [r(row.mean), r(row.ci_low), r(row.ci_high)]
                for row in g.itertuples()
                for met in [row.metric]
                if met in METRICS
            }
            for m, g in s.groupby("method")
        },
        "tests": [
            {
                "method": x.method,
                "metric": x.metric,
                "diff": r(x.mean_diff, 6),
                "lo": r(x.diff_ci_low, 6),
                "hi": r(x.diff_ci_high, 6),
                "pHolm": r(x.p_holm, 5),
            }
            for x in t.itertuples()
        ],
    }
    ab = read_csv(res / "e1b_ablations" / "per_seed.csv")
    abt = read_csv(res / "e1b_ablations" / "tests.csv")
    e1b = [
        {
            "variant": v,
            "method": m,
            **{k: r(g[k].mean(), 3) for k in ["expected_utility", "failure_rate", "q05_strength"]},
        }
        for (v, m), g in ab.groupby(["variant", "method"])
        if not str(m).startswith("oracle")
    ]
    sig = [
        {
            "variant": x.variant,
            "method": x.method,
            "metric": x.metric,
            "diff": r(x.mean_diff, 6),
            "pHolm": r(x.p_holm, 5),
        }
        for x in abt.itertuples()
        if x.p_holm < 0.05
    ]
    m = read_csv(res / "e2_models" / "metrics.csv")
    cv = m[m["split"] == "grouped_cv"]
    sizes = sorted(int(v) for v in cv["n_batches"].unique())
    crps = {name: [r(g[g["n_batches"] == n]["crps_log"].mean()) for n in sizes] for name, g in cv.groupby("model")}
    largest = (
        cv[cv["n_batches"] == sizes[-1]].groupby("model")[["rmse_MPa", "nll_log", "crps_log", "coverage90"]].mean()
    )
    conf = read_csv(res / "e2_models" / "conformal.csv")
    conf_l = conf[conf["n_batches"] == sizes[-1]].groupby(["model", "split"])["coverage90"].mean()
    al = read_csv(res / "e3_active_learning" / "per_round.csv")
    e3: dict[str, list[dict[str, float | None]]] = {}
    for strat, g in al.groupby("strategy"):
        rows = []
        for n, gg in g.groupby("n_jobs"):
            est, lo, hi = bootstrap_ci(gg["regret_expected_utility"].to_numpy())
            rows.append({"nJobs": int(n), "mean": r(est), "lo": r(lo), "hi": r(hi)})
        e3[str(strat)] = rows
    last = al[al["n_jobs"] == al["n_jobs"].max()].pivot_table(
        index="seed", columns="strategy", values="regret_expected_utility"
    )
    e3_contrasts = []
    for strat in last.columns:
        if strat == "random":
            continue
        est, lo, hi = bootstrap_ci((last[strat] - last["random"]).to_numpy())
        e3_contrasts.append({"strategy": str(strat), "diff": r(est), "lo": r(lo), "hi": r(hi)})
    dec = read_csv(res / "e4_propagation" / "decomposition.csv")
    sens = read_csv(res / "e5_sensitivity" / "global.csv")
    payload = {
        "label": SYNTHETIC_LABEL,
        "version": __version__,
        "commit": manifest.get("stage_git", {}).get("E1 robust vs nominal", {}).get("commit", "unknown"),
        "e1": e1,
        "e1b": e1b,
        "e1bSignificant": sig,
        "e2": {
            "sizes": sizes,
            "crps": crps,
            "largest": [
                {
                    "model": k,
                    "rmse": r(v.rmse_MPa, 3),
                    "nll": r(v.nll_log, 3),
                    "crps": r(v.crps_log),
                    "coverage": r(v.coverage90, 3),
                }
                for k, v in largest.iterrows()
            ],
            "conformal": [{"model": k[0], "split": k[1], "coverage": r(v, 3)} for k, v in conf_l.items()],
        },
        "e3": e3,
        "e3Seeds": int(al["seed"].nunique()),
        "e3Contrasts": e3_contrasts,
        "e4": [
            {k: (r(v) if isinstance(v, float) else v) for k, v in row.items()} for row in dec.to_dict(orient="records")
        ],
        "e5": [
            {"output": x.output, "input": x.input, "first": r(x.first), "total": r(x.total)}
            for x in sens.itertuples()
            if x.source == "ground_truth"
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "full")
