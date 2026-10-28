"""Print every headline number quoted in README.md, at full precision, with its source file.

Usage: python scripts/readme_numbers.py [results/full]
All values are SYNTHETIC (simulator output) except the E7 rows (public Kaggle data).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from robustprint.eval.common import read_csv


def main(res: Path) -> None:
    """Print the numbers grouped by experiment."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    s = read_csv(res / "e1_optimisation/summary.csv")
    print("## E1 summary (e1_optimisation/summary.csv)")
    keep = ["failure_rate", "q05_strength", "expected_utility", "regret_expected_utility"]
    print(s[s.metric.isin(keep)].pivot_table(index="method", columns="metric", values=["mean", "ci_low", "ci_high"]))
    t = read_csv(res / "e1_optimisation/tests.csv")
    print("\n## E1 tests (e1_optimisation/tests.csv)")
    print(t[["method", "metric", "mean_diff", "diff_ci_low", "diff_ci_high", "p_value", "p_holm"]].to_string())
    print("min p_holm", t.p_holm.min(), "min p", t.p_value.min())
    per = read_csv(res / "e1_optimisation/per_seed.csv")
    print("mean successful jobs", per.n_strength_rows.mean(), "jobs", per.n_jobs.iloc[0])
    o = read_csv(res / "e1_optimisation/tests_oracle_exploratory.csv")
    print("\n## E1 oracle exploratory")
    print(o[o.method == "oracle_expected"][["metric", "mean_a", "mean_b", "mean_diff", "p_holm"]].to_string())

    d = read_csv(res / "e4_propagation/decomposition.csv")
    print("\n## E4 decomposition (e4_propagation/decomposition.csv)")
    print(d.to_string())

    ab = read_csv(res / "e1b_ablations/per_seed.csv")
    abt = read_csv(res / "e1b_ablations/tests.csv")
    print("\n## E1b (e1b_ablations/)")
    print(ab.groupby(["variant", "method"])[["expected_utility", "failure_rate", "q05_strength"]].mean().to_string())
    print(abt[abt.p_holm < 0.05][["variant", "method", "metric", "mean_diff", "p_holm"]].to_string())

    m = read_csv(res / "e2_models/metrics.csv")
    print("\n## E2 metrics (e2_models/metrics.csv)")
    print(
        m.groupby(["n_batches", "split", "model"])[["n_train_rows", "rmse_MPa", "nll_log", "crps_log", "coverage90"]]
        .mean()
        .to_string()
    )
    c = read_csv(res / "e2_models/conformal.csv")
    print(c.groupby(["n_batches", "split", "model"])[["coverage90", "width90_log"]].mean().to_string())
    k = read_csv(res / "e2_models/classifier.csv")
    print(k.groupby(["n_batches", "split", "classifier"])[["brier", "ece", "base_rate"]].mean().to_string())

    al = read_csv(res / "e3_active_learning/per_round.csv")
    last = al[al["round"] == al["round"].max()]
    print("\n## E3 final round (e3_active_learning/per_round.csv)")
    print(last.groupby("strategy")[["regret", "regret_expected_utility", "n_jobs", "cumulative_cost"]].mean())
    ref = last[last.strategy == "random"].set_index("seed")["regret"]
    for st, g in last.groupby("strategy"):
        diff = g.set_index("seed")["regret"] - ref
        boots = np.random.default_rng(0).choice(diff.to_numpy(), size=(5000, len(diff))).mean(1)
        print(f"  {st} - random: {diff.mean():.3f} [{np.quantile(boots, 0.025):.3f}, {np.quantile(boots, 0.975):.3f}]")

    e = read_csv(res / "e4_propagation/errors.csv")
    e["rel_err_sd"] = e.err_sd.abs() / e.ref_sd
    print("\n## E4 propagation (e4_propagation/errors.csv)")
    print(e.groupby(["model", "method"])[["rel_err_sd", "seconds"]].mean().to_string())

    g = read_csv(res / "e5_sensitivity/global.csv")
    print("\n## E5 (e5_sensitivity/)")
    print(g.to_string())
    print(read_csv(res / "e5_sensitivity/components.csv").to_string())

    p = read_csv(res / "e6_pareto/front.csv")
    print("\n## E6 (e6_pareto/front.csv)")
    print(p[["objective", "time_budget", "print_time", "cvar10_strength", "expected_utility"]].to_string())

    bench = res / "e7_public_benchmark.csv"
    if bench.exists():
        print("\n## E7 PUBLIC DATA (not synthetic)")
        print(pd.read_csv(bench).groupby("model")[["rmse_MPa", "nll_log", "crps_log", "coverage90"]].mean())
        print(pd.read_csv(res / "e7_public_trends.csv").to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/full"))
