"""Common interface for probabilistic regressors.

All strength models in RobustPrint predict the **natural logarithm of tensile strength**
(ADR-013). Strength is positive and its specimen scatter is roughly proportional to its
level, so a Gaussian on log-strength is a better default than a Gaussian on MPa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from robustprint.physics.composition import FloatArray


@dataclass(frozen=True)
class Prediction:
    """Gaussian predictive distribution on the model's target scale.

    Attributes:
        mean: Predictive mean.
        var_epistemic: Variance due to limited data (model uncertainty).
        var_aleatoric: Irreducible noise variance (specimen-to-specimen scatter).
    """

    mean: FloatArray
    var_epistemic: FloatArray
    var_aleatoric: FloatArray

    @property
    def var(self) -> FloatArray:
        """Total predictive variance."""
        return self.var_epistemic + self.var_aleatoric

    @property
    def std(self) -> FloatArray:
        """Total predictive standard deviation."""
        return np.sqrt(self.var)

    def __len__(self) -> int:
        return int(self.mean.shape[0])

    def subset(self, idx: np.ndarray) -> Prediction:
        """Rows selected by a mask or index array."""
        return Prediction(self.mean[idx], self.var_epistemic[idx], self.var_aleatoric[idx])


@runtime_checkable
class ProbabilisticRegressor(Protocol):
    """Minimal protocol shared by all regressors."""

    name: str

    def fit(self, X: FloatArray, y: FloatArray, n_replicates: np.ndarray | None = None) -> ProbabilisticRegressor:
        """Fit to features ``X`` (n, d) and targets ``y`` (n,)."""
        ...

    def predict(self, X: FloatArray) -> Prediction:
        """Predictive distribution at ``X``."""
        ...


def gaussian_from_quantiles(lower: FloatArray, upper: FloatArray, level: float) -> FloatArray:
    """Standard deviation of a Gaussian whose central ``level`` interval is [lower, upper]."""
    from scipy.stats import norm

    z = norm.ppf(0.5 + level / 2.0)
    return np.asarray(np.maximum(upper - lower, 1e-9) / (2.0 * z), dtype=float)
