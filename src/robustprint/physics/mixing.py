"""Blend and composite mixing models.

Functions are vectorised over the leading axis: a composition array of shape (n, D)
returns arrays of shape (n,). See docs/METHODS.md section 2 for the equations.
"""

from __future__ import annotations

import numpy as np

from robustprint.physics.composition import FloatArray


def mass_to_volume_fractions(mass_frac: FloatArray, density: FloatArray) -> FloatArray:
    """Convert mass fractions to volume fractions given component densities."""
    vol = np.asarray(mass_frac, dtype=float) / np.asarray(density, dtype=float)
    return vol / vol.sum(axis=-1, keepdims=True)


def rule_of_mixtures(fractions: FloatArray, values: FloatArray) -> FloatArray:
    """Voigt (iso-strain) average: sum_i f_i v_i."""
    return np.asarray(fractions @ values, dtype=float)


def inverse_rule_of_mixtures(fractions: FloatArray, values: FloatArray) -> FloatArray:
    """Reuss (iso-stress) average: 1 / sum_i (f_i / v_i)."""
    return np.asarray(1.0 / (fractions @ (1.0 / np.asarray(values, dtype=float))), dtype=float)


def hill_average(fractions: FloatArray, values: FloatArray) -> FloatArray:
    """Arithmetic mean of the Voigt and Reuss bounds."""
    return 0.5 * (rule_of_mixtures(fractions, values) + inverse_rule_of_mixtures(fractions, values))


def immiscibility_index(weights: FloatArray, chi: FloatArray) -> FloatArray:
    """Pairwise incompatibility-weighted interface measure, in [0, 1].

    I(w) = sum_{i != j} chi_ij w_i w_j for polymer weights w that sum to one. It is zero
    for a single polymer or a fully compatible blend, 0.5 for an equal two-part blend
    with chi = 1, and never exceeds 1 - 1/D.
    """
    w = np.asarray(weights, dtype=float)
    return np.asarray(np.einsum("ni,ij,nj->n", w, chi, w), dtype=float)


def blend_strength(weights: FloatArray, strengths: FloatArray, chi: FloatArray, kappa: float) -> FloatArray:
    """Polymer-blend strength: rule of mixtures with an immiscibility penalty.

    sigma_blend = (sum_i w_i sigma_i) * (1 - kappa * I(w)), with I from
    :func:`immiscibility_index`. kappa is the strength loss of a 50/50 fully immiscible
    blend relative to the mixture rule, times two (ASSUMPTION, ADR-008).
    """
    penalty = 1.0 - kappa * immiscibility_index(weights, chi)
    return rule_of_mixtures(weights, strengths) * np.clip(penalty, 0.05, 1.0)


def halpin_tsai_modulus(
    matrix_modulus: FloatArray, fiber_modulus: float, fiber_volume: FloatArray, xi: float
) -> FloatArray:
    """Halpin–Tsai composite modulus (Halpin & Kardos, 1976).

    E_c = E_m (1 + xi * eta * V_f) / (1 - eta * V_f),
    eta = (E_f / E_m - 1) / (E_f / E_m + xi).
    """
    ratio = fiber_modulus / np.asarray(matrix_modulus, dtype=float)
    eta = (ratio - 1.0) / (ratio + xi)
    vf = np.asarray(fiber_volume, dtype=float)
    return np.asarray(matrix_modulus * (1.0 + xi * eta * vf) / (1.0 - eta * vf), dtype=float)


def short_fiber_strength(
    matrix_strength: FloatArray,
    fiber_strength: float,
    fiber_volume: FloatArray,
    efficiency: float,
    agglomeration_vf: float,
) -> FloatArray:
    """Modified rule of mixtures for short, randomly oriented, poorly bonded fibres.

    sigma_c = efficiency * sigma_f * V_f + sigma_m * (1 - V_f), multiplied by a
    smooth agglomeration penalty exp(-(V_f / V_agg)^2) that represents fibre clumping
    and voids at high loading (ASSUMPTION, ADR-009).
    """
    vf = np.asarray(fiber_volume, dtype=float)
    base = efficiency * fiber_strength * vf + matrix_strength * (1.0 - vf)
    return np.asarray(base * np.exp(-((vf / agglomeration_vf) ** 2)), dtype=float)


def inclusion_strength_factor(inclusion_volume: FloatArray, coefficient: float = 1.21) -> FloatArray:
    """Strength reduction by non-adhering inclusions, 1 - a * phi^(2/3).

    The functional form is commonly attributed to Nicolais & Narkis; the coefficient is
    treated as an assumption (see docs/REFERENCES.md, [NICOLAIS-1970]).
    """
    phi = np.clip(np.asarray(inclusion_volume, dtype=float), 0.0, 0.6)
    return np.asarray(np.clip(1.0 - coefficient * phi ** (2.0 / 3.0), 0.05, 1.0), dtype=float)
