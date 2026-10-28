"""Statistical tools: bootstrap confidence intervals, paired tests, effect sizes, Holm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from robustprint.physics.composition import FloatArray


def bootstrap_ci(values: FloatArray, n_boot: int = 5000, level: float = 0.95, seed: int = 0,
                 statistic: str = "mean") -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean or median. Returns (estimate, low, high)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    fn = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = fn(x[idx], axis=1)
    a = (1 - level) / 2
    return float(fn(x)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def rank_biserial(differences: FloatArray) -> float:
    """Matched-pairs rank-biserial correlation of paired differences, in [-1, 1].

    (sum of ranks of positive differences - sum of ranks of negative) / total rank sum,
    with zero differences dropped.
    """
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


@dataclass(frozen=True)
class PairedComparison:
    """Result of a paired comparison of method A against reference B."""

    metric: str
    method: str
    reference: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    diff_ci_low: float
    diff_ci_high: float
    median_diff: float
    rank_biserial: float
    p_value: float
    p_holm: float = float("nan")

    def as_dict(self) -> dict[str, float | int | str]:
        """Plain dictionary for JSON/CSV output."""
        return dict(self.__dict__)


def paired_comparison(a: FloatArray, b: FloatArray, metric: str, method: str, reference: str,
                      seed: int = 0) -> PairedComparison:
    """Two-sided Wilcoxon signed-rank test with bootstrap CI of the mean difference a - b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    est, lo, hi = bootstrap_ci(d, seed=seed)
    p = float(wilcoxon(a, b, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
    return PairedComparison(metric, method, reference, int(d.size), float(a.mean()), float(b.mean()),
                            est, lo, hi, float(np.median(d)), rank_biserial(d), p)


def holm(p_values: FloatArray) -> FloatArray:
    """Holm (1979) step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def with_holm(comparisons: list[PairedComparison]) -> list[PairedComparison]:
    """Attach Holm-adjusted p-values across a family of comparisons."""
    adj = holm(np.array([c.p_value for c in comparisons]))
    return [PairedComparison(**{**c.__dict__, "p_holm": float(q)}) for c, q in zip(comparisons, adj, strict=True)]
