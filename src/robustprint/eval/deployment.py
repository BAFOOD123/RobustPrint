"""Ground-truth deployment of recommended settings on fresh, unseen batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustprint.physics.composition import FloatArray
from robustprint.physics.simulator import DEFAULT_CONFIG, Batches, Feedstock, SimulatorConfig, realise, sample_batches


@dataclass(frozen=True)
class DeploymentBatches:
    """Fixed fresh batches and random numbers, shared by all methods (paired comparison)."""

    batches: Batches
    seed: int


def draw_deployment(feedstock: Feedstock, n_batches: int, seed: int, cfg: SimulatorConfig = DEFAULT_CONFIG) -> DeploymentBatches:
    """Draw fresh batches from the feedstock (never seen in training)."""
    rng = np.random.default_rng(seed)
    return DeploymentBatches(sample_batches(feedstock, n_batches, rng, cfg=cfg, prefix="D"), seed)


def deploy(params: FloatArray, deployment: DeploymentBatches, threshold_MPa: float = 5.0,
           cfg: SimulatorConfig = DEFAULT_CONFIG) -> dict[str, float]:
    """Print one part per fresh batch with fixed settings and summarise realised outcomes.

    Each part fails with the true failure probability; successful parts have a noisy
    measured strength. The same random numbers are used for every ``params`` so that
    differences between methods are paired.

    Returns:
        ``mean_strength`` (successful parts), ``q05_strength`` and ``cvar10_strength``
        (successful parts), ``failure_rate``, ``expected_utility`` (failed part counts as
        0 MPa), and ``p_meet_threshold`` (fraction of *all* parts that print and
        reach the threshold).
    """
    b = deployment.batches
    n = len(b)
    X = np.tile(np.asarray(params, dtype=float), (n, 1))
    rng = np.random.default_rng(deployment.seed + 1)
    out = realise(b.true_composition, X, b.moisture_pct, b.batch_effect, rng, cfg)
    ok = out["success"]
    s = out["strength_MPa"][ok]
    utility = np.where(ok, np.nan_to_num(out["strength_MPa"]), 0.0)
    if s.size:
        tail = np.sort(s)[: max(1, int(np.ceil(0.1 * s.size)))]
        q05, cvar, mean = float(np.quantile(s, 0.05)), float(tail.mean()), float(s.mean())
    else:
        q05 = cvar = mean = 0.0
    return {
        "mean_strength": mean,
        "q05_strength": q05,
        "cvar10_strength": cvar,
        "failure_rate": float(1.0 - ok.mean()),
        "expected_utility": float(utility.mean()),
        "p_meet_threshold": float(np.mean(utility >= threshold_MPa)),
    }
