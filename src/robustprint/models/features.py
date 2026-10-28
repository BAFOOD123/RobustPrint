"""Feature construction: composition (ilr or raw), print parameters and moisture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustprint import space
from robustprint.physics.composition import FloatArray, closure, ilr, ilr_inverse

COMPOSITION_ENCODINGS = ("ilr", "raw")


@dataclass(frozen=True)
class FeatureBuilder:
    """Maps (composition, params, moisture) to a model input matrix.

    Attributes:
        encoding: ``"ilr"`` (default, ADR-005) or ``"raw"`` fractions (ablation only).
        use_moisture: Whether log-moisture is a feature.
    """

    encoding: str = "ilr"
    use_moisture: bool = True

    def __post_init__(self) -> None:
        if self.encoding not in COMPOSITION_ENCODINGS:
            raise ValueError(f"encoding must be one of {COMPOSITION_ENCODINGS}")

    def transform(self, composition: FloatArray, params: FloatArray, moisture: FloatArray) -> FloatArray:
        """Build features. Parameters are scaled to the unit cube of the search space."""
        comp = closure(np.atleast_2d(composition))
        parts = [ilr(comp) if self.encoding == "ilr" else comp[:, :-1]]
        parts.append(space.to_unit(np.atleast_2d(params)))
        if self.use_moisture:
            m = np.broadcast_to(np.asarray(moisture, dtype=float), (comp.shape[0],))
            parts.append(np.log(np.maximum(m, 1e-3))[:, None])
        return np.hstack(parts)

    def n_features(self, n_components: int) -> int:
        """Number of output columns."""
        return n_components - 1 + space.N_PARAMS + int(self.use_moisture)

    def decode(self, X: FloatArray, n_components: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Invert :meth:`transform`: features -> (composition, params, moisture)."""
        X = np.atleast_2d(X)
        k = n_components - 1
        if self.encoding == "ilr":
            comp = ilr_inverse(X[:, :k])
        else:
            comp = closure(np.hstack([X[:, :k], np.clip(1.0 - X[:, :k].sum(1, keepdims=True), 1e-6, None)]))
        params = space.from_unit(X[:, k : k + space.N_PARAMS])
        moisture = np.exp(X[:, -1]) if self.use_moisture else np.full(X.shape[0], 0.3)
        return comp, params, moisture
