"""Column definitions for experimental and synthetic data (see data/schema.md)."""

from __future__ import annotations

from dataclasses import dataclass

from robustprint.physics.materials import DEFAULT_LIBRARY, MaterialLibrary
from robustprint.space import PARAM_NAMES

#: Allowed values of the ``data_origin`` column. Synthetic rows must never be mixed into a
#: dataset that is reported as experimental.
DATA_ORIGINS = ("experimental", "synthetic")

BATCH_COLUMNS_REQUIRED = ("batch_id", "date", "source_description")
BATCH_COLUMNS_OPTIONAL = (
    "composition_concentration",
    "moisture_pct",
    "melt_flow_index_g_10min",
    "particle_size_mm",
)
OUTCOME_COLUMNS = (
    "tensile_strength_MPa",
    "tensile_strength_sd_MPa",
    "youngs_modulus_GPa",
    "elongation_pct",
    "print_success",
    "n_replicates",
    "defect_notes",
)
EXPERIMENT_COLUMNS = ("experiment_id", "data_origin")

#: Physically valid ranges for numeric non-parameter columns (inclusive).
VALID_RANGES: dict[str, tuple[float, float]] = {
    "composition_concentration": (1.0, 1e6),
    "moisture_pct": (0.0, 20.0),
    "melt_flow_index_g_10min": (0.0, 500.0),
    "particle_size_mm": (0.0, 50.0),
    "tensile_strength_MPa": (0.0, 500.0),
    "tensile_strength_sd_MPa": (0.0, 200.0),
    "youngs_modulus_GPa": (0.0, 100.0),
    "elongation_pct": (0.0, 1000.0),
    "n_replicates": (1.0, 1000.0),
}

#: Tolerance on the sum of composition fractions.
COMPOSITION_SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class Schema:
    """Resolved schema for a given material library."""

    library: MaterialLibrary = DEFAULT_LIBRARY

    @property
    def fraction_columns(self) -> list[str]:
        """``frac_<component>`` columns, one per component, in library order."""
        return [f"frac_{name}" for name in self.library.names]

    @property
    def sd_columns(self) -> list[str]:
        """Optional per-component composition standard deviation columns."""
        return [f"sd_{name}" for name in self.library.names]

    @property
    def required_columns(self) -> list[str]:
        """Columns that must be present in every file."""
        return [
            *EXPERIMENT_COLUMNS,
            *BATCH_COLUMNS_REQUIRED,
            *self.fraction_columns,
            *PARAM_NAMES,
            "print_success",
            "n_replicates",
            "tensile_strength_MPa",
        ]

    @property
    def all_columns(self) -> list[str]:
        """Every recognised column in canonical order."""
        return [
            *EXPERIMENT_COLUMNS,
            *BATCH_COLUMNS_REQUIRED,
            *self.fraction_columns,
            *self.sd_columns,
            *BATCH_COLUMNS_OPTIONAL,
            *PARAM_NAMES,
            *OUTCOME_COLUMNS,
        ]


DEFAULT_SCHEMA = Schema()
