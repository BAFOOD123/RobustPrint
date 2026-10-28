"""High-level steps shared by the CLI, the dashboard and the experiments."""

from __future__ import annotations

import numpy as np

from robustprint.data.loaders import ExperimentArrays
from robustprint.models.classifier import SuccessClassifier
from robustprint.models.features import FeatureBuilder
from robustprint.models.registry import make_model
from robustprint.uncertainty.propagation import ModelPredictor, pooled_specimen_log_variance

MIN_STRENGTH_ROWS = 5


def train_predictor(arrays: ExperimentArrays, model: str = "hybrid_gp", seed: int = 0,
                    features: FeatureBuilder | None = None, n_members: int = 5, epochs: int = 400,
                    gp_restarts: int = 2, classifier: str = "gbm_isotonic") -> ModelPredictor:
    """Fit a strength model (on successful jobs) and a success classifier (on all jobs)."""
    fb = features or FeatureBuilder()
    X = fb.transform(arrays.composition, arrays.params, arrays.moisture)
    ok = arrays.strength_rows
    if ok.sum() < MIN_STRENGTH_ROWS:
        raise ValueError(f"Need at least {MIN_STRENGTH_ROWS} successful jobs with a strength; got {int(ok.sum())}.")
    strength_model = make_model(model, fb, seed, n_members=n_members, epochs=epochs, gp_restarts=gp_restarts)
    strength_model.fit(X[ok], np.log(arrays.strength[ok]), arrays.n_replicates[ok])
    success_model = SuccessClassifier(classifier, random_state=seed).fit(X, arrays.success)
    return ModelPredictor(
        features=fb,
        strength_model=strength_model,
        success_model=success_model,
        specimen_log_var=pooled_specimen_log_variance(arrays),
        mean_replicates=float(np.mean(arrays.n_replicates[ok])),
    )
