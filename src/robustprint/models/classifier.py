"""Print-success classifiers with calibrated probabilities."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from robustprint.physics.composition import FloatArray

CLASSIFIERS = ("logistic", "gbm_isotonic")


class SuccessClassifier:
    """P(print success | features).

    ``logistic``: L2 logistic regression on standardised features with pairwise
    interactions (well calibrated by construction when the model is adequate).
    ``gbm_isotonic``: gradient boosting with isotonic recalibration via 3-fold CV.
    If the training labels are all one class, a constant (Laplace-smoothed) rate is used.
    """

    def __init__(self, kind: str = "gbm_isotonic", C: float = 0.03, random_state: int = 0) -> None:
        if kind not in CLASSIFIERS:
            raise ValueError(f"kind must be one of {CLASSIFIERS}")
        self.kind = kind
        self.C = C
        self.random_state = random_state

    @property
    def name(self) -> str:
        """Classifier identifier."""
        return f"success_{self.kind}"

    def fit(self, X: FloatArray, success: np.ndarray) -> SuccessClassifier:
        """Fit on all print jobs (successful and failed)."""
        y = np.asarray(success, dtype=int)
        self.constant_: float | None = None
        if y.min() == y.max():
            self.constant_ = float((y.sum() + 1) / (len(y) + 2))
            return self
        if self.kind == "logistic":
            self.model_: Pipeline | CalibratedClassifierCV = make_pipeline(
                StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
                LogisticRegression(C=self.C, max_iter=2000),
            )
        else:
            base = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                                  random_state=self.random_state)
            self.model_ = CalibratedClassifierCV(base, method="isotonic", cv=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(X, y)
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Probability of success for each row."""
        if self.constant_ is not None:
            return np.full(len(X), self.constant_)
        return np.asarray(self.model_.predict_proba(X)[:, 1], dtype=float)
