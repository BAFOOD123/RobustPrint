"""Schema validation, synthetic generation and loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robustprint.data.loaders import estimate_feedstock, fit_dirichlet, load_experiments, to_arrays
from robustprint.data.synthetic import design_points, generate_dataset
from robustprint.data.validation import coerce_bool, validate_frame
from robustprint.physics.simulator import ISS_NOMINAL


def test_synthetic_dataset_validates_cleanly(small_dataset: pd.DataFrame) -> None:
    clean, report = validate_frame(small_dataset)
    assert report.ok, report.summary()
    assert (clean["data_origin"] == "synthetic").all()
    assert clean["source_description"].str.startswith("SYNTHETIC").all()
    fracs = clean.filter(like="frac_").to_numpy()
    np.testing.assert_allclose(fracs.sum(1), 1.0, atol=1e-9)


def test_failed_jobs_have_no_strength(small_dataset: pd.DataFrame) -> None:
    failed = ~small_dataset["print_success"].astype(bool)
    assert small_dataset.loc[failed, "tensile_strength_MPa"].isna().all()
    assert small_dataset.loc[~failed, "tensile_strength_MPa"].notna().all()


def test_generation_is_deterministic() -> None:
    a, _ = generate_dataset(ISS_NOMINAL, 5, 3, np.random.default_rng(9))
    b, _ = generate_dataset(ISS_NOMINAL, 5, 3, np.random.default_rng(9))
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("design", ["sobol", "lhs", "random"])
def test_design_points_in_unit_cube(design: str) -> None:
    pts = design_points(37, np.random.default_rng(0), design)
    assert pts.shape == (37, 8)
    assert (pts >= 0).all() and (pts <= 1).all()


def test_unknown_design_rejected() -> None:
    with pytest.raises(ValueError):
        design_points(4, np.random.default_rng(0), "grid")


def _broken(frame: pd.DataFrame) -> pd.DataFrame:
    bad = frame.copy()
    for col in ("print_success", "n_replicates", "infill_pct"):
        bad[col] = bad[col].astype(object)
    bad.loc[0, "nozzle_temp_C"] = 500.0
    bad.loc[1, "frac_pe"] = bad.loc[1, "frac_pe"] + 0.3
    bad.loc[2, "print_success"] = "maybe"
    bad.loc[3, "data_origin"] = "made-up"
    bad.loc[4, "date"] = "not a date"
    bad.loc[5, "experiment_id"] = bad.loc[6, "experiment_id"]
    bad.loc[7, "n_replicates"] = 2.5
    bad.loc[8, "infill_pct"] = "abc"
    return bad


def test_validation_reports_each_problem_readably(small_dataset: pd.DataFrame) -> None:
    _, report = validate_frame(_broken(small_dataset))
    text = report.summary()
    assert not report.ok
    assert "nozzle_temp_C=500 is outside the physically valid range [150, 350] °C" in text
    assert "sum to" in text and "divide by 100" in text
    assert "print_success='maybe'" in text
    assert "data_origin='made-up'" in text
    assert "not a valid date" in text
    assert "experiment_id is not unique" in text
    assert "n_replicates must be a whole number" in text
    assert "infill_pct='abc' is not a number" in text
    assert "rows disagree on composition" in text


def test_missing_columns_reported(small_dataset: pd.DataFrame) -> None:
    _, report = validate_frame(small_dataset.drop(columns=["frac_pe", "nozzle_temp_C"]))
    assert "Missing required column(s): frac_pe, nozzle_temp_C" in report.summary()


def test_warnings_for_out_of_search_range_and_mixed_origin(small_dataset: pd.DataFrame) -> None:
    df = small_dataset.copy()
    df.loc[0, "nozzle_temp_C"] = 300.0
    df.loc[1, "data_origin"] = "experimental"
    df["extra_col"] = 1
    _, report = validate_frame(df)
    assert report.ok
    text = report.summary()
    assert "outside the optimiser search range" in text
    assert "mixes experimental and synthetic" in text
    assert "unrecognised column(s): extra_col" in text


def test_success_strength_mismatch_warnings(small_dataset: pd.DataFrame) -> None:
    df = small_dataset.copy()
    ok_idx = df.index[df["print_success"]][0]
    fail_idx = df.index[~df["print_success"]][0]
    df.loc[ok_idx, "tensile_strength_MPa"] = np.nan
    df.loc[fail_idx, "tensile_strength_MPa"] = 5.0
    _, report = validate_frame(df)
    assert report.ok
    assert "used for the success model only" in report.summary()
    assert "the strength is ignored" in report.summary()


def test_coerce_bool_spellings() -> None:
    s, bad = coerce_bool(pd.Series(["TRUE", "no", 1, True, "y", "?"]))
    assert s.tolist()[:5] == [True, False, True, True, True]
    assert bad == [5]


def test_to_arrays_and_estimate_feedstock(small_dataset: pd.DataFrame) -> None:
    clean, _ = validate_frame(small_dataset)
    arrays = to_arrays(clean)
    assert len(arrays) == len(clean)
    assert arrays.strength_rows.sum() == clean["print_success"].sum()
    sub = arrays.subset(arrays.strength_rows)
    assert np.isfinite(sub.strength).all()
    fs = estimate_feedstock(arrays)
    assert np.abs(np.array(fs.mean) - ISS_NOMINAL.mean_array()).max() < 0.05
    assert 10 < fs.concentration < 120


def test_moisture_imputation(small_dataset: pd.DataFrame) -> None:
    df = small_dataset.copy()
    df.loc[:5, "moisture_pct"] = np.nan
    clean, _ = validate_frame(df)
    arr = to_arrays(clean)
    assert np.isfinite(arr.moisture).all()
    arr2 = to_arrays(clean.drop(columns=["moisture_pct"]))
    assert np.allclose(arr2.moisture, 0.3)


def test_fit_dirichlet_recovers_parameters() -> None:
    alpha = np.array([15.0, 9.0, 5.0, 3.0, 2.0])
    x = np.random.default_rng(0).dirichlet(alpha, size=5000)
    np.testing.assert_allclose(fit_dirichlet(x), alpha, rtol=0.06)


def test_fit_dirichlet_needs_two_rows() -> None:
    with pytest.raises(ValueError):
        fit_dirichlet(np.array([[0.5, 0.5]]))


def test_load_experiments_roundtrip_and_error(small_dataset: pd.DataFrame, tmp_path) -> None:
    good = tmp_path / "good.csv"
    small_dataset.to_csv(good, index=False)
    arrays, report = load_experiments(str(good))
    assert report.ok and len(arrays) == len(small_dataset)
    bad = tmp_path / "bad.csv"
    _broken(small_dataset).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="failed validation"):
        load_experiments(str(bad))
