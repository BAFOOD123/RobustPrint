"""Definition of the print-parameter design space shared by every module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustprint.physics.composition import FloatArray


@dataclass(frozen=True)
class Parameter:
    """One controllable print parameter.

    Attributes:
        name: Column name in the data schema.
        low: Lower bound of the search space.
        high: Upper bound of the search space.
        unit: Unit string for display.
        default: A conventional starting value.
        hard_low: Physically valid lower limit used for data validation.
        hard_high: Physically valid upper limit used for data validation.
    """

    name: str
    low: float
    high: float
    unit: str
    default: float
    hard_low: float
    hard_high: float


PARAMETERS: tuple[Parameter, ...] = (
    Parameter("nozzle_temp_C", 210.0, 290.0, "°C", 230.0, 150.0, 350.0),
    Parameter("bed_temp_C", 40.0, 110.0, "°C", 70.0, 15.0, 150.0),
    Parameter("print_speed_mm_s", 10.0, 80.0, "mm/s", 40.0, 1.0, 300.0),
    Parameter("layer_height_mm", 0.10, 0.35, "mm", 0.2, 0.02, 1.0),
    Parameter("infill_pct", 20.0, 100.0, "%", 100.0, 0.0, 100.0),
    Parameter("raster_angle_deg", 0.0, 90.0, "deg", 45.0, -90.0, 90.0),
    Parameter("flow_rate_pct", 85.0, 120.0, "%", 100.0, 50.0, 200.0),
    Parameter("cooling_fan_pct", 0.0, 100.0, "%", 50.0, 0.0, 100.0),
)

PARAM_NAMES: list[str] = [p.name for p in PARAMETERS]
N_PARAMS: int = len(PARAMETERS)


def bounds() -> tuple[FloatArray, FloatArray]:
    """Lower and upper search bounds as arrays in ``PARAM_NAMES`` order."""
    low = np.array([p.low for p in PARAMETERS])
    high = np.array([p.high for p in PARAMETERS])
    return low, high


def to_unit(x: FloatArray) -> FloatArray:
    """Map parameters from physical units to the unit hypercube."""
    low, high = bounds()
    return (np.asarray(x, dtype=float) - low) / (high - low)


def from_unit(u: FloatArray) -> FloatArray:
    """Map points in the unit hypercube to physical parameter values."""
    low, high = bounds()
    return low + np.clip(np.asarray(u, dtype=float), 0.0, 1.0) * (high - low)


def defaults() -> FloatArray:
    """Conventional default settings."""
    return np.array([p.default for p in PARAMETERS])


def as_dict(x: FloatArray) -> dict[str, float]:
    """Convert one parameter vector to a ``{name: value}`` dictionary."""
    return {name: float(v) for name, v in zip(PARAM_NAMES, np.asarray(x, dtype=float), strict=True)}
