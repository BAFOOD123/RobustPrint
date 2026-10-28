"""Compositional-data utilities: closure, clr/ilr transforms and Dirichlet moments.

Compositions are vectors of non-negative fractions that sum to one. Their sample space
is the simplex, so Euclidean distances on raw fractions are misleading (a change from
0.01 to 0.02 doubles a component, a change from 0.50 to 0.51 barely matters). Following
Egozcue et al. (2003), models in RobustPrint see compositions through the isometric
log-ratio (ilr) transform, which maps the D-part simplex isometrically onto R^(D-1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import digamma, polygamma

FloatArray = NDArray[np.float64]

#: Fractions below this value are replaced before taking logarithms.
ZERO_REPLACEMENT = 1e-4


def closure(x: ArrayLike) -> FloatArray:
    """Rescale rows of ``x`` so that they sum to one."""
    arr = np.asarray(x, dtype=float)
    total = arr.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("Every composition must have a positive total.")
    return arr / total


def replace_zeros(x: ArrayLike, delta: float = ZERO_REPLACEMENT) -> FloatArray:
    """Multiplicative zero replacement: set zeros to ``delta`` and shrink the rest.

    This keeps ratios between the non-zero parts unchanged, which is the standard
    approach for rounded zeros in compositional data.
    """
    arr = closure(x)
    is_small = arr < delta
    n_small = is_small.sum(axis=-1, keepdims=True)
    rest = np.where(is_small, 0.0, arr)
    rest_total = rest.sum(axis=-1, keepdims=True)
    return np.where(is_small, delta, rest * (1.0 - n_small * delta) / rest_total)


def helmert_basis(n_parts: int) -> FloatArray:
    """Return a (D, D-1) orthonormal basis of the clr hyperplane (Helmert contrasts).

    Column j contrasts the first j+1 parts against part j+2, so the columns are
    orthonormal and orthogonal to the vector of ones.
    """
    if n_parts < 2:
        raise ValueError("A composition needs at least two parts.")
    basis = np.zeros((n_parts, n_parts - 1))
    for j in range(1, n_parts):
        basis[:j, j - 1] = 1.0 / j
        basis[j, j - 1] = -1.0
        basis[:, j - 1] *= np.sqrt(j / (j + 1.0))
    return basis


def clr(x: ArrayLike) -> FloatArray:
    """Centred log-ratio transform."""
    logx = np.log(replace_zeros(x))
    return logx - logx.mean(axis=-1, keepdims=True)


def ilr(x: ArrayLike) -> FloatArray:
    """Isometric log-ratio transform using the Helmert basis."""
    arr = np.asarray(x, dtype=float)
    return clr(arr) @ helmert_basis(arr.shape[-1])


def ilr_inverse(z: ArrayLike) -> FloatArray:
    """Inverse ilr transform back to the simplex."""
    arr = np.asarray(z, dtype=float)
    clr_coords = arr @ helmert_basis(arr.shape[-1] + 1).T
    expx = np.exp(clr_coords - clr_coords.max(axis=-1, keepdims=True))
    return closure(expx)


def dirichlet_params(mean: ArrayLike, concentration: float) -> FloatArray:
    """Dirichlet parameter vector alpha = concentration * mean."""
    mu = closure(mean)
    if concentration <= 0:
        raise ValueError("Concentration must be positive.")
    return np.asarray(concentration * mu, dtype=float)


def dirichlet_std(mean: ArrayLike, concentration: float) -> FloatArray:
    """Marginal standard deviation of each part under Dirichlet(concentration * mean)."""
    mu = closure(mean)
    return np.asarray(np.sqrt(mu * (1.0 - mu) / (concentration + 1.0)), dtype=float)


def concentration_from_std(mean: ArrayLike, std: ArrayLike) -> float:
    """Moment-match a single Dirichlet concentration to per-part standard deviations.

    Each part gives an estimate ``mu(1-mu)/sd^2 - 1``; we take the median over parts with
    a positive standard deviation, which is robust to one badly specified entry.
    """
    mu = closure(mean)
    sd = np.asarray(std, dtype=float)
    valid = sd > 0
    if not np.any(valid):
        raise ValueError("At least one positive standard deviation is required.")
    estimates = mu[valid] * (1.0 - mu[valid]) / sd[valid] ** 2 - 1.0
    return float(max(np.median(estimates), 1.0))


def sample_dirichlet(
    mean: ArrayLike, concentration: float, n: int, rng: np.random.Generator
) -> FloatArray:
    """Draw ``n`` compositions from Dirichlet(concentration * mean), zero-safe."""
    alpha = dirichlet_params(mean, concentration)
    draws = rng.dirichlet(np.maximum(alpha, 1e-3), size=n)
    return replace_zeros(draws)


def dirichlet_ilr_moments(alpha: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Exact mean and covariance of ilr(c) for c ~ Dirichlet(alpha).

    Uses E[log c_i] = psi(a_i) - psi(a_0) and
    Cov[log c_i, log c_j] = psi'(a_i) delta_ij - psi'(a_0). Because the ilr basis V is
    orthogonal to the ones vector, ilr(c) = V^T log c and the constant terms cancel.
    """
    a = np.asarray(alpha, dtype=float)
    a0 = a.sum()
    mean_log = digamma(a) - digamma(a0)
    cov_log = np.diag(polygamma(1, a)) - polygamma(1, a0)
    basis = helmert_basis(a.size)
    return mean_log @ basis, basis.T @ cov_log @ basis
