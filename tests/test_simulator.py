"""Simulator invariants, physically expected monotonic trends and determinism."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from robustprint import space
from robustprint.physics import process
from robustprint.physics.composition import closure
from robustprint.physics.simulator import (
    ISS_NOMINAL,
    ISS_SHIFTED,
    realise,
    sample_batches,
    true_outcomes,
)

MEAN = ISS_NOMINAL.mean_array()
GOOD = np.array([265.0, 100.0, 30.0, 0.15, 100.0, 0.0, 100.0, 20.0])


def _sweep(index: int, values: np.ndarray, comp: np.ndarray = MEAN) -> dict[str, np.ndarray]:
    x = np.tile(GOOD, (len(values), 1))
    x[:, index] = values
    return true_outcomes(np.tile(comp, (len(values), 1)), x, np.full(len(values), 0.3))


@given(
    st.lists(st.floats(0.01, 1.0), min_size=5, max_size=5),
    st.lists(st.floats(0.0, 1.0), min_size=8, max_size=8),
    st.floats(0.02, 3.0),
)
@settings(max_examples=200, deadline=None)
def test_outputs_are_finite_positive_and_probabilities(c: list[float], u: list[float], m: float) -> None:
    out = true_outcomes(closure(np.array([c])), space.from_unit(np.array([u])), np.array([m]))
    for key in ("strength_MPa", "modulus_GPa", "elongation_pct"):
        assert np.isfinite(out[key]).all() and (out[key] > 0).all()
    assert 0.0 <= out["p_success"][0] <= 1.0
    assert 0.0 <= out["bond"][0] <= 1.0
    assert 0.0 <= out["porosity"][0] <= 0.6


def test_strength_increases_with_infill() -> None:
    s = _sweep(4, np.linspace(20, 100, 9))["strength_MPa"]
    assert np.all(np.diff(s) > 0)


def test_zero_degree_raster_strongest() -> None:
    s = _sweep(5, np.linspace(0, 90, 10))["strength_MPa"]
    assert s.argmax() == 0 and s[0] > s[-1]


def test_fan_reduces_bonding() -> None:
    bond = _sweep(7, np.linspace(0, 100, 5))["bond"]
    assert np.all(np.diff(bond) < 0)


def test_higher_bed_reduces_warping() -> None:
    warp = _sweep(1, np.linspace(40, 110, 8))["p_warp"]
    assert np.all(np.diff(warp) < 0)


def test_low_temperature_leaves_polymer_unmelted_and_clogs() -> None:
    out = _sweep(0, np.array([210.0, 270.0]))
    assert out["unmelted"][0] > out["unmelted"][1]
    assert out["p_clog"][0] > out["p_clog"][1]


def test_contaminant_reduces_strength_and_raises_clogging() -> None:
    clean = closure(np.array([0.47, 0.26, 0.16, 0.08, 0.03]))
    dirty = closure(np.array([0.41, 0.23, 0.14, 0.08, 0.14]))
    out = true_outcomes(np.stack([clean, dirty]), np.tile(GOOD, (2, 1)), np.full(2, 0.3))
    assert out["strength_MPa"][0] > out["strength_MPa"][1]
    assert out["p_clog"][0] < out["p_clog"][1]


def test_fibre_raises_modulus_and_clogging() -> None:
    low = closure(np.array([0.5, 0.25, 0.15, 0.03, 0.07]))
    high = closure(np.array([0.4, 0.2, 0.13, 0.20, 0.07]))
    out = true_outcomes(np.stack([low, high]), np.tile(GOOD, (2, 1)), np.full(2, 0.3))
    assert out["modulus_GPa"][1] > out["modulus_GPa"][0]
    assert out["p_clog"][1] > out["p_clog"][0]


def test_pure_polymer_has_no_immiscibility() -> None:
    pure = closure(np.array([0.9996, 1e-4, 1e-4, 1e-4, 1e-4]))
    assert true_outcomes(pure[None], GOOD[None], np.array([0.3]))["immiscibility"][0] < 1e-3


def test_moisture_reduces_strength_at_high_temperature() -> None:
    x = np.tile(GOOD, (2, 1))
    out = true_outcomes(np.tile(MEAN, (2, 1)), x, np.array([0.1, 2.0]))
    assert out["strength_MPa"][0] > out["strength_MPa"][1]


def test_extrusion_limit_reached_at_high_flow() -> None:
    ratio, demand = process.extrusion_ratio(
        np.array([220.0]), np.array([80.0]), np.array([0.35]), np.array([110.0]),
        np.array([0.1]), np.array([0.0]), process.ProcessConstants(),
    )
    assert demand[0] > 1.0 and ratio[0] < 0.9


def test_batches_and_realisation_are_deterministic_given_seed() -> None:
    a = sample_batches(ISS_NOMINAL, 20, np.random.default_rng(7))
    b = sample_batches(ISS_NOMINAL, 20, np.random.default_rng(7))
    np.testing.assert_array_equal(a.true_composition, b.true_composition)
    x = np.tile(GOOD, (20, 1))
    ra = realise(a.true_composition, x, a.moisture_pct, a.batch_effect, np.random.default_rng(3))
    rb = realise(b.true_composition, x, b.moisture_pct, b.batch_effect, np.random.default_rng(3))
    np.testing.assert_array_equal(ra["success"], rb["success"])
    np.testing.assert_array_equal(np.nan_to_num(ra["strength_MPa"]), np.nan_to_num(rb["strength_MPa"]))


def test_batches_valid_compositions() -> None:
    b = sample_batches(ISS_SHIFTED, 500, np.random.default_rng(0))
    np.testing.assert_allclose(b.true_composition.sum(1), 1.0)
    np.testing.assert_allclose(b.reported_composition.sum(1), 1.0)
    assert (b.true_composition > 0).all()
    assert np.abs(b.true_composition.mean(0) - ISS_SHIFTED.mean_array()).max() < 0.02


def test_failed_prints_have_no_strength() -> None:
    b = sample_batches(ISS_NOMINAL, 200, np.random.default_rng(1))
    bad = np.tile(np.array([210.0, 40.0, 80.0, 0.35, 100.0, 0.0, 120.0, 100.0]), (200, 1))
    r = realise(b.true_composition, bad, b.moisture_pct, b.batch_effect, np.random.default_rng(2))
    assert np.isnan(r["strength_MPa"][~r["success"]]).all()
    assert (~r["success"]).mean() > 0.5


def test_regression_values_fixed_seed() -> None:
    """Guards against silent changes to the frozen v1.0 simulator."""
    out = true_outcomes(MEAN[None], GOOD[None], np.array([0.3]))
    np.testing.assert_allclose(out["strength_MPa"][0], REGRESSION_STRENGTH, rtol=1e-6)
    np.testing.assert_allclose(out["p_success"][0], REGRESSION_PSUCCESS, rtol=1e-6)


REGRESSION_STRENGTH = 9.429437164053875
REGRESSION_PSUCCESS = 0.9220128808822595
