"""Validation of experiment tables with human-readable error messages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from robustprint.data.schema import (
    COMPOSITION_SUM_TOLERANCE,
    DATA_ORIGINS,
    DEFAULT_SCHEMA,
    VALID_RANGES,
    Schema,
)
from robustprint.space import PARAMETERS

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


@dataclass
class ValidationReport:
    """Errors make a file unusable; warnings are worth a look but do not block use."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no errors."""
        return not self.errors

    def summary(self) -> str:
        """Multi-line, human-readable summary."""
        lines = [f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _row_label(df: pd.DataFrame, idx: object) -> str:
    """'row 3 (experiment_id=E0003)' with 1-based CSV data-row numbering."""
    position = list(df.index).index(idx) + 1
    exp = str(df.at[idx, "experiment_id"]) if "experiment_id" in df.columns else "?"
    return f"row {position} (experiment_id={exp})"


def coerce_bool(series: pd.Series) -> tuple[pd.Series, list[int]]:
    """Convert common spellings of booleans; returns the series and bad positions."""
    out: list[bool | None] = []
    bad: list[int] = []
    for i, value in enumerate(series.tolist()):
        if isinstance(value, bool | np.bool_):
            out.append(bool(value))
            continue
        text = str(value).strip().lower()
        if text in _TRUE:
            out.append(True)
        elif text in _FALSE:
            out.append(False)
        else:
            out.append(None)
            bad.append(i)
    return pd.Series(out, index=series.index, dtype="object"), bad


def _check_numeric(
    df: pd.DataFrame,
    report: ValidationReport,
    column: str,
    low: float,
    high: float,
    unit: str = "",
    required: bool = False,
) -> None:
    """Range-check a numeric column, reporting each offending row."""
    if column not in df.columns:
        return
    values = pd.to_numeric(df[column], errors="coerce")
    non_numeric = df[column].notna() & values.isna()
    for idx in df.index[non_numeric]:
        report.errors.append(f"{_row_label(df, idx)}: {column}={df.at[idx, column]!r} is not a number.")
    if required:
        for idx in df.index[df[column].isna()]:
            report.errors.append(f"{_row_label(df, idx)}: {column} is missing.")
    out = values.notna() & ((values < low) | (values > high))
    for idx in df.index[out]:
        report.errors.append(
            f"{_row_label(df, idx)}: {column}={values[idx]:g} is outside the physically valid "
            f"range [{low:g}, {high:g}] {unit}".rstrip()
            + "."
        )
    df[column] = values


def validate_frame(df: pd.DataFrame, schema: Schema = DEFAULT_SCHEMA) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and normalise an experiment table.

    Returns a cleaned copy (numeric columns coerced, ``print_success`` as bool) and a
    :class:`ValidationReport`. The cleaned frame is only safe to use when ``report.ok``.
    """
    report = ValidationReport()
    df = df.copy()
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required column(s): {', '.join(missing)}.")
        return df, report
    unknown = [c for c in df.columns if c not in schema.all_columns]
    if unknown:
        report.warnings.append(f"Ignoring unrecognised column(s): {', '.join(map(str, unknown))}.")

    dup = df["experiment_id"].astype(str).duplicated(keep=False)
    for idx in df.index[dup]:
        report.errors.append(f"{_row_label(df, idx)}: experiment_id is not unique.")

    origins = df["data_origin"].astype(str).str.strip().str.lower()
    for idx in df.index[~origins.isin(DATA_ORIGINS)]:
        report.errors.append(
            f"{_row_label(df, idx)}: data_origin={df.at[idx, 'data_origin']!r} must be one of {DATA_ORIGINS}."
        )
    df["data_origin"] = origins
    if origins.isin(DATA_ORIGINS).all() and origins.nunique() > 1:
        report.warnings.append("File mixes experimental and synthetic rows; keep them separate when reporting.")

    try:
        dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    except (TypeError, ValueError):
        dates = pd.to_datetime(df["date"], errors="coerce")
    for idx in df.index[dates.isna()]:
        report.errors.append(
            f"{_row_label(df, idx)}: date={df.at[idx, 'date']!r} is not a valid date (use YYYY-MM-DD)."
        )

    for col in schema.fraction_columns:
        _check_numeric(df, report, col, 0.0, 1.0, required=True)
    fractions = df[schema.fraction_columns].astype(float)
    sums = fractions.sum(axis=1)
    bad_sum = fractions.notna().all(axis=1) & ((sums - 1.0).abs() > COMPOSITION_SUM_TOLERANCE)
    for idx in df.index[bad_sum]:
        report.errors.append(
            f"{_row_label(df, idx)}: composition fractions sum to {sums[idx]:.4f}; they must sum to 1 "
            f"(±{COMPOSITION_SUM_TOLERANCE}). If you recorded percentages, divide by 100."
        )
    for col in schema.sd_columns:
        _check_numeric(df, report, col, 0.0, 0.5)

    for param in PARAMETERS:
        _check_numeric(df, report, param.name, param.hard_low, param.hard_high, param.unit, required=True)
        values = df[param.name]
        outside = values.notna() & ((values < param.low) | (values > param.high))
        if outside.any():
            report.warnings.append(
                f"{int(outside.sum())} row(s) have {param.name} outside the optimiser search range "
                f"[{param.low:g}, {param.high:g}] {param.unit}; models will extrapolate there."
            )

    for column, (low, high) in VALID_RANGES.items():
        _check_numeric(df, report, column, low, high)

    success, bad = coerce_bool(df["print_success"])
    for pos in bad:
        idx = df.index[pos]
        report.errors.append(
            f"{_row_label(df, idx)}: print_success={df.at[idx, 'print_success']!r} must be true/false."
        )
    df["print_success"] = success
    ok_success = success.notna()
    no_strength = ok_success & success.astype(bool) & df["tensile_strength_MPa"].isna()
    for idx in df.index[no_strength]:
        report.warnings.append(
            f"{_row_label(df, idx)}: print succeeded but tensile_strength_MPa is empty; row is used "
            "for the success model only."
        )
    failed_with_strength = ok_success & ~success.astype(bool) & df["tensile_strength_MPa"].notna()
    for idx in df.index[failed_with_strength]:
        report.warnings.append(
            f"{_row_label(df, idx)}: print_success is false but a strength is given; the strength is ignored."
        )

    if "n_replicates" in df.columns:
        reps = df["n_replicates"]
        non_int = reps.notna() & (reps != np.round(reps))
        for idx in df.index[non_int]:
            report.errors.append(f"{_row_label(df, idx)}: n_replicates must be a whole number.")

    _check_batch_consistency(df, schema, report)
    if report.ok:
        df["print_success"] = df["print_success"].astype(bool)
        df["n_replicates"] = df["n_replicates"].astype(int)
    return df, report


def _check_batch_consistency(df: pd.DataFrame, schema: Schema, report: ValidationReport) -> None:
    """Rows of the same batch must report the same composition."""
    cols = schema.fraction_columns
    grouped = df.groupby(df["batch_id"].astype(str))[cols]
    spread = grouped.max() - grouped.min()
    for batch_id, row in spread.iterrows():
        if (row.astype(float) > 1e-6).any():
            report.errors.append(
                f"batch {batch_id}: rows disagree on composition; every row of a batch must repeat the same fractions."
            )


def validate_csv(path: str, schema: Schema = DEFAULT_SCHEMA) -> tuple[pd.DataFrame, ValidationReport]:
    """Read and validate a CSV file."""
    df = pd.read_csv(path)
    return validate_frame(df, schema)
