"""Construct strength models by name."""

from __future__ import annotations

from functools import partial

from robustprint.models.base import ProbabilisticRegressor
from robustprint.models.baselines import GBMQuantile, PhysicsROMBaseline, RidgeInteractions, simplified_physics_prior
from robustprint.models.features import FeatureBuilder
from robustprint.models.gp import GPRegressor, HeteroscedasticGP, HybridGP
from robustprint.models.neural import DeepEnsemble, MCDropout
from robustprint.physics.composition import FloatArray
from robustprint.physics.materials import DEFAULT_LIBRARY, MaterialLibrary

MODEL_NAMES = (
    "physics_rom",
    "ridge_interactions",
    "gbm_quantile",
    "gp",
    "het_gp",
    "hybrid_gp",
    "deep_ensemble",
    "mc_dropout",
)


def physics_prior_on_features(
    X: FloatArray, features: FeatureBuilder, library: MaterialLibrary = DEFAULT_LIBRARY
) -> FloatArray:
    """Evaluate :func:`simplified_physics_prior` on encoded features."""
    comp, params, _ = features.decode(X, library.n)
    return simplified_physics_prior(comp, params, library)


def make_model(
    name: str,
    features: FeatureBuilder,
    seed: int = 0,
    n_members: int = 5,
    epochs: int = 400,
    gp_restarts: int = 2,
    library: MaterialLibrary = DEFAULT_LIBRARY,
) -> ProbabilisticRegressor:
    """Instantiate a strength model.

    Args:
        name: One of :data:`MODEL_NAMES`.
        features: Feature builder used to encode the inputs (needed by physics-aware models).
        seed: Random seed for stochastic models.
        n_members: Deep-ensemble size.
        epochs: Training epochs for neural models.
        gp_restarts: Optimiser restarts for GP hyperparameters.
        library: Material library for physics-aware models.
    """
    if name == "physics_rom":
        return PhysicsROMBaseline(features, library)
    if name == "ridge_interactions":
        return RidgeInteractions()
    if name == "gbm_quantile":
        return GBMQuantile(random_state=seed)
    if name == "gp":
        return GPRegressor(n_restarts=gp_restarts, random_state=seed)
    if name == "het_gp":
        return HeteroscedasticGP(n_restarts=max(gp_restarts - 1, 0), random_state=seed)
    if name == "hybrid_gp":
        return HybridGP(
            partial(physics_prior_on_features, features=features, library=library),
            n_restarts=gp_restarts,
            random_state=seed,
        )
    if name == "deep_ensemble":
        return DeepEnsemble(n_members=n_members, epochs=epochs, random_state=seed)
    if name == "mc_dropout":
        return MCDropout(epochs=epochs, random_state=seed)
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}.")
