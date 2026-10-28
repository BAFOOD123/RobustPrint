"""Gaussian-process regressors: ARD GP, heteroscedastic GP and physics-residual hybrid GP."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from robustprint.models.base import Prediction
from robustprint.physics.composition import FloatArray

#: Smallest allowed noise standard deviation on log strength: a 3 % coefficient of variation,
#: a conservative lower bound on tensile-test repeatability (ASSUMPTION, ADR-014).
MIN_LOG_NOISE_SD = 0.03


def ard_kernel(n_dims: int, noise: bool = True, noise_floor: float = 1e-5) -> Kernel:
    """Constant x Matern-5/2 with one length scale per input, plus optional white noise.

    ``noise_floor`` is the lower bound of the white-noise variance (standardised units).
    """
    kernel: Kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.full(n_dims, 1.0), length_scale_bounds=(5e-2, 1e3), nu=2.5
    )
    if noise:
        kernel = kernel + WhiteKernel(max(1e-1, 2 * noise_floor), (noise_floor, 1.0))
    return kernel


class GPRegressor:
    """Homoscedastic ARD GP on standardised inputs and outputs (Rasmussen & Williams, 2006)."""

    name = "gp"

    def __init__(self, n_restarts: int = 2, random_state: int = 0) -> None:
        self.n_restarts = n_restarts
        self.random_state = random_state

    def _fit_gp(
        self, Xs: FloatArray, ys: FloatArray, alpha: FloatArray | float, noise: bool, noise_floor: float = 1e-5
    ) -> GaussianProcessRegressor:
        gp = GaussianProcessRegressor(
            kernel=ard_kernel(Xs.shape[1], noise=noise, noise_floor=noise_floor),
            alpha=alpha,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(Xs, ys)
        return gp

    def fit(self, X: FloatArray, y: FloatArray, n_replicates: np.ndarray | None = None) -> GPRegressor:
        """Fit kernel hyperparameters by maximising the marginal likelihood."""
        self.x_scaler_ = StandardScaler().fit(X)
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y) or 1.0)
        ys = (y - self.y_mean_) / self.y_std_
        floor = min(MIN_LOG_NOISE_SD**2 / self.y_std_**2, 0.5)
        self.gp_ = self._fit_gp(self.x_scaler_.transform(X), ys, 1e-6, noise=True, noise_floor=floor)
        return self

    @property
    def noise_variance(self) -> float:
        """Learned white-noise variance on the original target scale."""
        return float(self.gp_.kernel_.k2.noise_level) * self.y_std_**2

    @property
    def length_scales(self) -> FloatArray:
        """Learned ARD length scales (standardised inputs)."""
        return np.asarray(self.gp_.kernel_.k1.k2.length_scale, dtype=float)

    def predict(self, X: FloatArray) -> Prediction:
        """Latent (epistemic) variance and white-noise (aleatoric) variance."""
        mean, std = self.gp_.predict(self.x_scaler_.transform(X), return_std=True)
        total = std**2 * self.y_std_**2
        noise = self.noise_variance
        return Prediction(
            mean * self.y_std_ + self.y_mean_, np.maximum(total - noise, 1e-12), np.full(len(mean), noise)
        )


class HeteroscedasticGP(GPRegressor):
    """Most-likely heteroscedastic GP (Kersting et al., 2007).

    1. Fit a homoscedastic GP. 2. Estimate per-point noise from posterior samples of the
    residuals. 3. Fit a second GP to the log noise. 4. Refit the first GP with the
    predicted per-point noise. Steps 2-4 are repeated ``n_iter`` times.
    """

    name = "het_gp"

    def __init__(self, n_restarts: int = 1, random_state: int = 0, n_iter: int = 2, n_samples: int = 100) -> None:
        super().__init__(n_restarts, random_state)
        self.n_iter = n_iter
        self.n_samples = n_samples

    def fit(self, X: FloatArray, y: FloatArray, n_replicates: np.ndarray | None = None) -> HeteroscedasticGP:
        """Iterative most-likely-noise fitting."""
        super().fit(X, y, n_replicates)
        Xs = self.x_scaler_.transform(X)
        ys = (y - self.y_mean_) / self.y_std_
        rng = np.random.default_rng(self.random_state)
        gp = self.gp_
        for _ in range(self.n_iter):
            mean, cov = gp.predict(Xs, return_cov=True)
            draws = rng.multivariate_normal(
                mean, cov + 1e-9 * np.eye(len(mean)), size=self.n_samples, method="cholesky"
            )
            resid_var = np.mean((ys[None, :] - draws) ** 2, axis=0)
            log_noise = np.log(np.maximum(resid_var, 1e-6))
            self.noise_mean_ = float(log_noise.mean())
            self.noise_gp_ = self._fit_gp(Xs, log_noise - self.noise_mean_, 1e-6, noise=True)
            floor = MIN_LOG_NOISE_SD**2 / self.y_std_**2
            alpha = np.maximum(np.exp(self.noise_gp_.predict(Xs) + self.noise_mean_), floor)
            gp = self._fit_gp(Xs, ys, alpha, noise=False)
        self.gp_ = gp
        return self

    def predict(self, X: FloatArray) -> Prediction:
        """Latent variance from the main GP; input-dependent noise from the noise GP."""
        Xs = self.x_scaler_.transform(X)
        mean, std = self.gp_.predict(Xs, return_std=True)
        noise = np.maximum(np.exp(self.noise_gp_.predict(Xs) + self.noise_mean_) * self.y_std_**2, MIN_LOG_NOISE_SD**2)
        return Prediction(mean * self.y_std_ + self.y_mean_, std**2 * self.y_std_**2, noise)


class HybridGP(GPRegressor):
    """GP on the residual of a physics prior: y = prior(x) + f(x) + noise."""

    name = "hybrid_gp"

    def __init__(self, prior: Callable[[FloatArray], FloatArray], n_restarts: int = 2, random_state: int = 0) -> None:
        super().__init__(n_restarts, random_state)
        self.prior = prior

    def fit(self, X: FloatArray, y: FloatArray, n_replicates: np.ndarray | None = None) -> HybridGP:
        """Fit the GP to y - prior(X)."""
        super().fit(X, y - self.prior(X), n_replicates)
        return self

    def predict(self, X: FloatArray) -> Prediction:
        """Prior plus GP residual prediction."""
        base = super().predict(X)
        return Prediction(base.mean + self.prior(X), base.var_epistemic, base.var_aleatoric)
