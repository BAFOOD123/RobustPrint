"""Compositional transforms: unit and property-based tests."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from robustprint.physics import composition as comp

positive_rows = arrays(
    np.float64, st.integers(2, 7), elements=st.floats(1e-3, 1.0, allow_nan=False)
)


@given(positive_rows)
@settings(max_examples=200, deadline=None)
def test_ilr_roundtrip(x: np.ndarray) -> None:
    c = comp.closure(x)
    back = comp.ilr_inverse(comp.ilr(c))
    np.testing.assert_allclose(back, comp.replace_zeros(c), rtol=1e-8, atol=1e-10)


@given(positive_rows)
@settings(max_examples=100, deadline=None)
def test_closure_sums_to_one(x: np.ndarray) -> None:
    assert np.isclose(comp.closure(x).sum(), 1.0)


@given(st.integers(2, 8))
def test_helmert_basis_orthonormal(d: int) -> None:
    v = comp.helmert_basis(d)
    np.testing.assert_allclose(v.T @ v, np.eye(d - 1), atol=1e-12)
    np.testing.assert_allclose(np.ones(d) @ v, np.zeros(d - 1), atol=1e-12)


@given(positive_rows, positive_rows)
@settings(max_examples=100, deadline=None)
def test_ilr_is_isometry_of_aitchison_distance(x: np.ndarray, y: np.ndarray) -> None:
    n = min(x.size, y.size)
    a, b = comp.closure(x[:n]), comp.closure(y[:n])
    if n < 2:
        return
    aitchison = np.linalg.norm(comp.clr(a) - comp.clr(b))
    np.testing.assert_allclose(np.linalg.norm(comp.ilr(a) - comp.ilr(b)), aitchison, rtol=1e-9, atol=1e-9)


def test_replace_zeros_keeps_sum_and_positivity() -> None:
    out = comp.replace_zeros(np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]]))
    assert np.all(out > 0)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert np.isclose(out[0, 1], out[0, 2])


def test_closure_rejects_zero_total() -> None:
    with pytest.raises(ValueError):
        comp.closure(np.zeros(3))


def test_dirichlet_ilr_moments_match_monte_carlo() -> None:
    alpha = np.array([12.0, 8.0, 5.0, 3.0, 2.0])
    mean, cov = comp.dirichlet_ilr_moments(alpha)
    draws = comp.ilr(np.random.default_rng(0).dirichlet(alpha, size=200_000))
    np.testing.assert_allclose(draws.mean(0), mean, atol=0.01)
    np.testing.assert_allclose(np.cov(draws.T), cov, atol=0.01)


def test_concentration_from_std_inverts_dirichlet_std() -> None:
    mean = np.array([0.5, 0.3, 0.2])
    sd = comp.dirichlet_std(mean, 40.0)
    assert np.isclose(comp.concentration_from_std(mean, sd), 40.0)


def test_sample_dirichlet_mean() -> None:
    mean = np.array([0.45, 0.25, 0.15, 0.08, 0.07])
    draws = comp.sample_dirichlet(mean, 35.0, 50_000, np.random.default_rng(1))
    np.testing.assert_allclose(draws.mean(0), mean, atol=0.005)
