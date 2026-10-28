"""Mixing models: limits, bounds and monotonicity."""

from __future__ import annotations

import numpy as np

from robustprint.physics import mixing


def test_rule_of_mixtures_bounds_and_pure_limits() -> None:
    f = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    v = np.array([1.0, 3.0])
    voigt = mixing.rule_of_mixtures(f, v)
    reuss = mixing.inverse_rule_of_mixtures(f, v)
    np.testing.assert_allclose(voigt[:2], v)
    np.testing.assert_allclose(reuss[:2], v)
    assert reuss[2] <= mixing.hill_average(f, v)[2] <= voigt[2]


def test_immiscibility_zero_for_pure_and_half_for_equal_binary() -> None:
    chi = np.array([[0.0, 1.0], [1.0, 0.0]])
    w = np.array([[1.0, 0.0], [0.5, 0.5]])
    np.testing.assert_allclose(mixing.immiscibility_index(w, chi), [0.0, 0.5])


def test_blend_strength_penalised_below_rule_of_mixtures() -> None:
    chi = np.array([[0.0, 1.0], [1.0, 0.0]])
    w = np.array([[0.5, 0.5]])
    s = np.array([20.0, 60.0])
    assert mixing.blend_strength(w, s, chi, 0.5)[0] < mixing.rule_of_mixtures(w, s)[0]
    np.testing.assert_allclose(mixing.blend_strength(w, s, np.zeros((2, 2)), 0.5), [40.0])


def test_halpin_tsai_limits_and_monotonicity() -> None:
    vf = np.linspace(0.0, 0.3, 7)
    em = np.full(7, 2.0)
    e = mixing.halpin_tsai_modulus(em, 70.0, vf, 4.0)
    assert np.isclose(e[0], 2.0)
    assert np.all(np.diff(e) > 0)
    voigt = em * (1 - vf) + 70.0 * vf
    assert np.all(e <= voigt + 1e-12)


def test_inclusion_factor_decreasing() -> None:
    phi = np.linspace(0, 0.4, 9)
    f = mixing.inclusion_strength_factor(phi)
    assert f[0] == 1.0
    assert np.all(np.diff(f) < 0)


def test_mass_to_volume_fractions() -> None:
    v = mixing.mass_to_volume_fractions(np.array([[0.5, 0.5]]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(v, [[2 / 3, 1 / 3]])
