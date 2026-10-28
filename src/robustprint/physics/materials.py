"""Representative component properties for the default ISS-like mixed waste stream.

Every number has a ``source`` tag. ``ETB`` values fall inside the ranges listed by the
Engineering ToolBox reference tables (see docs/REFERENCES.md, [ETB-MODULUS] and
[ETB-POLYMER]). ``ASSUMPTION`` values are engineering judgements that are listed and
justified in docs/DECISIONS.md (ADR-007). None of these values is grade-specific: they
set the plausible scale of the synthetic data, not the properties of any real feedstock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robustprint.physics.composition import FloatArray


@dataclass(frozen=True)
class Component:
    """Bulk properties of one feedstock component.

    Attributes:
        name: Short identifier used in CSV column names (``frac_<name>``).
        description: Human-readable description of what the component represents.
        kind: ``"polymer"`` (melts and forms matrix), ``"fiber"`` or ``"inclusion"``.
        density_g_cm3: Density.
        modulus_GPa: Young's modulus of the bulk material.
        strength_MPa: Ultimate tensile strength of the bulk material (for fibres: the
            effective strength contribution, see ADR-007).
        elongation_pct: Elongation at break of a *printed* specimen of the neat material.
        t_min_C: Lowest nozzle temperature that fully melts the component in FDM.
        t_crit_C: Critical interface temperature above which neighbouring beads sinter.
        t_deg_C: Nozzle temperature at which thermal degradation becomes noticeable.
        sources: Mapping from attribute name to its source tag.
    """

    name: str
    description: str
    kind: str
    density_g_cm3: float
    modulus_GPa: float
    strength_MPa: float
    elongation_pct: float
    t_min_C: float
    t_crit_C: float
    t_deg_C: float
    sources: dict[str, str] = field(default_factory=dict)


DEFAULT_COMPONENTS: tuple[Component, ...] = (
    Component(
        name="pe",
        description="Polyethylene films and bags (packaging)",
        kind="polymer",
        density_g_cm3=0.95,
        modulus_GPa=0.8,
        strength_MPa=20.0,
        elongation_pct=12.0,
        t_min_C=190.0,
        t_crit_C=150.0,
        t_deg_C=300.0,
        sources={
            "density_g_cm3": "ETB-POLYMER",
            "modulus_GPa": "ETB-MODULUS",
            "strength_MPa": "ETB-MODULUS/ETB-POLYMER (15-30 MPa)",
            "elongation_pct": "ASSUMPTION",
            "t_min_C": "ASSUMPTION",
            "t_crit_C": "ASSUMPTION",
            "t_deg_C": "ASSUMPTION",
        },
    ),
    Component(
        name="pet",
        description="Polyethylene terephthalate (food-packaging laminates, bottles)",
        kind="polymer",
        density_g_cm3=1.38,
        modulus_GPa=2.4,
        strength_MPa=55.0,
        elongation_pct=4.0,
        t_min_C=255.0,
        t_crit_C=230.0,
        t_deg_C=295.0,
        sources={
            "density_g_cm3": "ASSUMPTION",
            "modulus_GPa": "ETB-MODULUS (2-2.7 GPa)",
            "strength_MPa": "ETB-MODULUS",
            "elongation_pct": "ASSUMPTION",
            "t_min_C": "ASSUMPTION",
            "t_crit_C": "ASSUMPTION",
            "t_deg_C": "ASSUMPTION",
        },
    ),
    Component(
        name="pa",
        description="Polyamide / nylon (textile layers, hook-and-loop, film layers)",
        kind="polymer",
        density_g_cm3=1.16,
        modulus_GPa=2.5,
        strength_MPa=60.0,
        elongation_pct=15.0,
        t_min_C=230.0,
        t_crit_C=200.0,
        t_deg_C=285.0,
        sources={
            "density_g_cm3": "ETB-POLYMER",
            "modulus_GPa": "ETB-MODULUS (2-4 GPa)",
            "strength_MPa": "ETB-MODULUS (45-90 MPa)",
            "elongation_pct": "ASSUMPTION",
            "t_min_C": "ASSUMPTION",
            "t_crit_C": "ASSUMPTION",
            "t_deg_C": "ASSUMPTION",
        },
    ),
    Component(
        name="fiber",
        description="Short aramid-type fibres from shredded cloth and cargo bags",
        kind="fiber",
        density_g_cm3=1.44,
        modulus_GPa=70.0,
        strength_MPa=400.0,
        elongation_pct=3.0,
        t_min_C=float("nan"),
        t_crit_C=float("nan"),
        t_deg_C=400.0,
        sources={
            "density_g_cm3": "ASSUMPTION",
            "modulus_GPa": "ETB-MODULUS (aramid 70-112 GPa, lower end)",
            "strength_MPa": "ASSUMPTION (effective contribution, short poorly bonded fibres)",
            "elongation_pct": "ASSUMPTION",
            "t_deg_C": "ASSUMPTION",
        },
    ),
    Component(
        name="contaminant",
        description="Foam fragments, cellulose, food and adhesive residues",
        kind="inclusion",
        density_g_cm3=1.2,
        modulus_GPa=0.05,
        strength_MPa=2.0,
        elongation_pct=1.0,
        t_min_C=float("nan"),
        t_crit_C=float("nan"),
        t_deg_C=235.0,
        sources={
            "density_g_cm3": "ASSUMPTION",
            "modulus_GPa": "ASSUMPTION",
            "strength_MPa": "ASSUMPTION",
            "elongation_pct": "ASSUMPTION",
            "t_deg_C": "ASSUMPTION",
        },
    ),
)

#: Pairwise incompatibility chi_ij in [0, 1] between matrix polymers (ASSUMPTION, ADR-008).
#: 1 means fully immiscible with no interfacial adhesion; 0 means fully compatible.
DEFAULT_INCOMPATIBILITY: dict[frozenset[str], float] = {
    frozenset({"pe", "pet"}): 1.0,
    frozenset({"pe", "pa"}): 0.9,
    frozenset({"pet", "pa"}): 0.5,
}


@dataclass(frozen=True)
class MaterialLibrary:
    """An ordered set of components plus their pairwise blend incompatibilities."""

    components: tuple[Component, ...] = DEFAULT_COMPONENTS
    incompatibility: dict[frozenset[str], float] = field(
        default_factory=lambda: dict(DEFAULT_INCOMPATIBILITY)
    )

    @property
    def names(self) -> list[str]:
        """Component names in column order."""
        return [c.name for c in self.components]

    @property
    def n(self) -> int:
        """Number of components."""
        return len(self.components)

    def mask(self, kind: str) -> FloatArray:
        """1.0 for components of the given kind, else 0.0."""
        return np.array([1.0 if c.kind == kind else 0.0 for c in self.components])

    def vector(self, attribute: str) -> FloatArray:
        """Return one attribute for all components as an array."""
        return np.array([float(getattr(c, attribute)) for c in self.components])

    def chi_matrix(self) -> FloatArray:
        """Symmetric incompatibility matrix with zeros on the diagonal."""
        names = self.names
        chi = np.zeros((self.n, self.n))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i != j:
                    chi[i, j] = self.incompatibility.get(frozenset({a, b}), 0.0)
        return chi


DEFAULT_LIBRARY = MaterialLibrary()
