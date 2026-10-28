"""Variance-based global sensitivity analysis with grouped inputs.

Inputs are sampled from independent *groups* (each print setting alone, the composition
vector as one group, moisture, batch effect). Groups must be mutually independent; the
parts of a composition are not (they sum to one), which is why the composition is one
group. First-order indices use the Saltelli et al. (2010) estimator and total indices the
Jansen estimator, both from Sobol' (2001) A/B/AB_i designs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from robustprint.physics.composition import FloatArray

Sampler = Callable[[int, np.random.Generator], FloatArray]


def sobol_indices(
    f: Callable[[dict[str, FloatArray]], FloatArray], samplers: dict[str, Sampler], n: int, rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    """First-order and total Sobol indices for each input group.

    Args:
        f: Model taking a dict of group samples (each (n, k_g)) and returning (n,) outputs.
        samplers: Group name -> function drawing (n, k_g) samples.
        n: Base sample size; cost is n * (2 + number of groups) evaluations.
        rng: Random generator.
    """
    A = {g: s(n, rng) for g, s in samplers.items()}
    B = {g: s(n, rng) for g, s in samplers.items()}
    fA, fB = f(A), f(B)
    var = float(np.var(np.concatenate([fA, fB]), ddof=1))
    out: dict[str, dict[str, float]] = {}
    for g in samplers:
        ABg = {**A, g: B[g]}
        fAB = f(ABg)
        first = float(np.mean(fB * (fAB - fA)) / var)
        total = float(0.5 * np.mean((fA - fAB) ** 2) / var)
        out[g] = {"first": first, "total": total}
    return out


def first_order_binned(x: FloatArray, y: FloatArray, n_bins: int = 20) -> float:
    """Given-data first-order index Var(E[y | x]) / Var(y) by quantile binning.

    Valid for dependent inputs (it is a correlation ratio), but not additive across inputs.
    """
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    means = np.array([y[idx == b].mean() for b in range(n_bins) if np.any(idx == b)])
    counts = np.array([np.sum(idx == b) for b in range(n_bins) if np.any(idx == b)])
    between = float(np.sum(counts * (means - y.mean()) ** 2) / len(y))
    return between / float(np.var(y))


def pivot_coordinate(composition: FloatArray, j: int) -> FloatArray:
    """First pivot log-ratio coordinate of part j: sqrt((D-1)/D) ln(c_j / g(c_-j))."""
    c = np.asarray(composition, dtype=float)
    d = c.shape[1]
    others = np.delete(c, j, axis=1)
    g = np.exp(np.mean(np.log(others), axis=1))
    return np.asarray(np.sqrt((d - 1) / d) * np.log(c[:, j] / g), dtype=float)
