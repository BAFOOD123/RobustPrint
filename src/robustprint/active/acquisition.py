"""Next-experiment selection for GP strength models.

All acquisition functions here use the GP posterior *covariance*, which does not depend on
the not-yet-observed outcomes, so greedy batch selection with exact "fantasy" updates is
possible without simulating outcomes (Rasmussen & Williams, 2006, ch. 2).

Strategies
----------
- ``max_variance``: uncertainty sampling. For a GP with Gaussian noise the expected
  information gain about the latent function at a single point, 0.5 log(1 + s2_f / s2_n),
  is monotone in the latent variance s2_f, so this is also the single-point EIG/BALD
  choice for regression (cf. Houlsby et al., 2011).
- ``targeted_ivr``: integrated variance reduction over a *reference set*: the K feedstock
  draws at the currently recommended robust setting. It spends experiments where the
  optimiser's decision is most uncertain.
- ``cost_aware_ivr``: targeted IVR per unit cost.

Every score is multiplied by the predicted success probability, because a failed print
yields no strength measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.gaussian_process.kernels import WhiteKernel

from robustprint.models.gp import GPRegressor
from robustprint.physics.composition import FloatArray
from robustprint.physics.process import print_time_proxy

STRATEGIES = ("random", "sobol", "max_variance", "targeted_ivr", "cost_aware_ivr")


def experiment_cost(params: FloatArray, setup_cost: float = 1.0) -> FloatArray:
    """Relative cost of one print job: fixed setup (blend, load, test) + print time.

    The setup term (1.0 = one reference print time) is an ASSUMPTION (ADR-017).
    """
    p = np.atleast_2d(params)
    return np.asarray(setup_cost + print_time_proxy(p[:, 2], p[:, 3], p[:, 4]), dtype=float)


class GPPosterior:
    """Latent posterior covariance of a fitted :class:`GPRegressor` (any subclass)."""

    def __init__(self, model: GPRegressor) -> None:
        self.model = model
        gp = model.gp_
        self.kernel = gp.kernel_
        self.X_train = gp.X_train_
        self.L = gp.L_
        k2 = getattr(self.kernel, "k2", None)
        self.noise = float(k2.noise_level) if isinstance(k2, WhiteKernel) else 0.0

    def _scaled(self, X: FloatArray) -> FloatArray:
        return np.asarray(self.model.x_scaler_.transform(X), dtype=float)

    def cov(self, A: FloatArray, B: FloatArray) -> FloatArray:
        """Posterior latent covariance between feature rows A and B (standardised y units)."""
        a, b = self._scaled(A), self._scaled(B)
        ka = self.kernel(a, self.X_train)
        kb = self.kernel(b, self.X_train)
        va = solve_triangular(self.L, ka.T, lower=True)
        vb = solve_triangular(self.L, kb.T, lower=True)
        return np.asarray(self.kernel(a, b) - va.T @ vb, dtype=float)

    def var(self, A: FloatArray) -> FloatArray:
        """Posterior latent variance (diagonal), excluding white noise."""
        a = self._scaled(A)
        ka = self.kernel(a, self.X_train)
        va = solve_triangular(self.L, ka.T, lower=True)
        return np.asarray(np.maximum(self.kernel.diag(a) - self.noise - np.sum(va**2, axis=0), 1e-12), dtype=float)

    @property
    def noise_var(self) -> float:
        """Observation noise variance in standardised units (homoscedastic GP)."""
        return max(self.noise, 1e-6)


@dataclass(frozen=True)
class Selection:
    """Indices of selected candidates and their acquisition values at selection time."""

    indices: list[int]
    scores: list[float]


def greedy_select(
    posterior: GPPosterior,
    candidates: FloatArray,
    p_success: FloatArray,
    batch_size: int,
    strategy: str,
    reference: FloatArray | None = None,
    cost: FloatArray | None = None,
    n_replicates: int = 3,
) -> Selection:
    """Greedy batch selection with exact posterior-covariance updates.

    Args:
        posterior: GP posterior.
        candidates: (N, d) candidate feature rows.
        p_success: (N,) predicted success probability of each candidate.
        batch_size: Number of experiments to select.
        strategy: ``max_variance``, ``targeted_ivr`` or ``cost_aware_ivr``.
        reference: (R, d) reference feature rows for IVR strategies.
        cost: (N,) relative cost for the cost-aware strategy.
        n_replicates: Specimens per job (the observation noise is divided by this).
    """
    if strategy not in ("max_variance", "targeted_ivr", "cost_aware_ivr"):
        raise ValueError(f"greedy_select does not handle strategy {strategy!r}")
    noise = posterior.noise_var / n_replicates
    var_c = posterior.var(candidates)
    use_ref = strategy != "max_variance"
    if use_ref:
        if reference is None:
            raise ValueError("IVR strategies need a reference set.")
        cross = posterior.cov(reference, candidates)
    cc = posterior.cov(candidates, candidates)
    chosen: list[int] = []
    scores: list[float] = []
    available = np.ones(len(candidates), dtype=bool)
    for _ in range(min(batch_size, len(candidates))):
        gain = np.sum(cross**2, axis=0) / (var_c + noise) if use_ref else var_c.copy()
        value = gain * p_success
        if strategy == "cost_aware_ivr":
            if cost is None:
                raise ValueError("cost_aware_ivr needs costs.")
            value = value / cost
        value = np.where(available, value, -np.inf)
        j = int(np.argmax(value))
        chosen.append(j)
        scores.append(float(value[j]))
        available[j] = False
        # Fantasy update: condition on a noisy observation at candidate j.
        denom = var_c[j] + noise
        kj = cc[:, j].copy()
        if use_ref:
            cross = cross - np.outer(cross[:, j], kj) / denom
        cc = cc - np.outer(kj, kj) / denom
        var_c = np.maximum(np.diag(cc).copy(), 1e-12)
    return Selection(chosen, scores)
