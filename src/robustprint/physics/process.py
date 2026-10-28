"""Process-parameter effect models: interlayer bonding, extrusion limits and porosity.

All coefficients live in :class:`ProcessConstants`. Their values are assumptions chosen to
reproduce qualitative trends reported in the FDM literature (docs/DECISIONS.md, ADR-010);
they are not fitted to any measured data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustprint.physics.composition import FloatArray

GAS_CONSTANT = 8.314
KELVIN = 273.15


@dataclass(frozen=True)
class ProcessConstants:
    """Coefficients of the process model (all ASSUMPTION, see ADR-010)."""

    ambient_C: float = 25.0
    bed_coupling: float = 0.3
    tau_cool_s: float = 1.5
    fan_cooling_gain: float = 2.0
    layer_path_mm: float = 600.0
    tau_layer_s: float = 8.0
    t_rep_ref_s: float = 0.4
    t_rep_ref_C: float = 230.0
    activation_J_mol: float = 60_000.0
    bond_softmin_power: float = 4.0
    line_width_mm: float = 0.45
    nozzle_diameter_mm: float = 0.4
    q_max_ref_mm3_s: float = 9.0
    q_max_ref_C: float = 220.0
    q_max_temp_scale_C: float = 60.0
    fiber_viscosity_gain: float = 6.0
    unmelted_viscosity_gain: float = 3.0
    extrusion_softmin_power: float = 6.0
    base_porosity: float = 0.03
    layer_porosity_gain: float = 0.08
    underextrusion_porosity_gain: float = 0.9
    moisture_porosity_gain: float = 0.03
    max_porosity: float = 0.6


def softplus(x: FloatArray) -> FloatArray:
    """Numerically stable log(1 + exp(x))."""
    arr = np.asarray(x, dtype=float)
    return np.asarray(np.logaddexp(0.0, arr), dtype=float)


def sigmoid(x: FloatArray) -> FloatArray:
    """Logistic function."""
    arr = np.asarray(x, dtype=float)
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * arr)), dtype=float)


def smooth_min_one(r: FloatArray, power: float) -> FloatArray:
    """Smooth approximation of min(r, 1) for r >= 0: r / (1 + r^p)^(1/p)."""
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    return np.asarray(r / (1.0 + r**power) ** (1.0 / power), dtype=float)


def interface_start_temperature(
    bed_temp: FloatArray, print_speed: FloatArray, nozzle_temp: FloatArray, k: ProcessConstants
) -> FloatArray:
    """Temperature of the previous layer when the next bead is deposited.

    The surroundings are warmed by the bed (envelope = ambient + coupling * (bed - ambient)).
    The previous layer has cooled for one layer time t_layer = path / speed towards the
    envelope with time constant tau_layer. Faster printing therefore leaves a hotter
    substrate, which improves bonding (qualitative trend from [SUN-2008]).
    """
    envelope = k.ambient_C + k.bed_coupling * (bed_temp - k.ambient_C)
    t_layer = k.layer_path_mm / np.maximum(print_speed, 1e-3)
    decay = np.exp(-t_layer / k.tau_layer_s)
    return np.asarray(envelope + (nozzle_temp - envelope) * decay, dtype=float)


def bond_efficiency(
    nozzle_temp: FloatArray,
    bed_temp: FloatArray,
    print_speed: FloatArray,
    layer_height: FloatArray,
    fan_pct: FloatArray,
    t_crit: FloatArray,
    k: ProcessConstants,
) -> FloatArray:
    """Degree of interlayer healing beta in [0, 1].

    The interface cools exponentially from the nozzle temperature towards the substrate
    temperature with time constant tau = tau0 * (h / 0.2) / (1 + g * fan). The time spent
    above the critical sintering temperature t_above drives reptation; following the
    t^(1/4) healing law [WOOL-1981], beta = softmin(1, (t_above / t_rep)^(1/4)), where the
    reptation time t_rep follows an Arrhenius law evaluated at the mean interface
    temperature while above T_crit.
    """
    substrate = interface_start_temperature(bed_temp, print_speed, nozzle_temp, k)
    tau = k.tau_cool_s * (layer_height / 0.2) / (1.0 + k.fan_cooling_gain * fan_pct / 100.0)
    excess = np.maximum(nozzle_temp - substrate, 1e-6)
    crit_excess = np.maximum(t_crit - substrate, 1e-6)
    t_above = tau * np.log(np.maximum(excess / crit_excess, 1.0))
    t_mean_K = 0.5 * (nozzle_temp + np.maximum(t_crit, substrate)) + KELVIN
    t_ref_K = k.t_rep_ref_C + KELVIN
    t_rep = k.t_rep_ref_s * np.exp(k.activation_J_mol / GAS_CONSTANT * (1.0 / t_mean_K - 1.0 / t_ref_K))
    ratio = (t_above / t_rep) ** 0.25
    return smooth_min_one(ratio, k.bond_softmin_power)


def extrusion_ratio(
    nozzle_temp: FloatArray,
    print_speed: FloatArray,
    layer_height: FloatArray,
    flow_pct: FloatArray,
    fiber_volume: FloatArray,
    unmelted_fraction: FloatArray,
    k: ProcessConstants,
) -> tuple[FloatArray, FloatArray]:
    """Delivered/commanded flow ratio and the commanded volumetric flow.

    The hot end can melt at most Q_max = Q_ref * exp((T - T_ref) / s) / v mm^3/s, where
    v = 1 + a * V_f + b * unmelted is a viscosity factor for fibres and unmelted
    particles. The commanded flow is Q = speed * h * w * flow/100. The delivered ratio is
    softmin(1, Q_max / Q). Hot-end melt-rate limits of order 10 mm^3/s are typical of
    desktop printers (ASSUMPTION, ADR-010).
    """
    q_demand = print_speed * layer_height * k.line_width_mm * flow_pct / 100.0
    viscosity = 1.0 + k.fiber_viscosity_gain * fiber_volume + k.unmelted_viscosity_gain * unmelted_fraction
    q_max = k.q_max_ref_mm3_s * np.exp((nozzle_temp - k.q_max_ref_C) / k.q_max_temp_scale_C) / viscosity
    ratio = smooth_min_one(q_max / np.maximum(q_demand, 1e-9), k.extrusion_softmin_power)
    return ratio, np.asarray(q_demand / q_max, dtype=float)


def porosity(
    layer_height: FloatArray,
    delivered_flow: FloatArray,
    moisture_pct: FloatArray,
    degradation: FloatArray,
    k: ProcessConstants,
) -> FloatArray:
    """Void fraction from layer geometry, under-extrusion, moisture and degassing.

    delivered_flow is the fraction of nominal material actually deposited
    (flow/100 * extrusion ratio).
    """
    under = softplus(20.0 * (1.0 - delivered_flow)) / 20.0
    p = (
        k.base_porosity
        + k.layer_porosity_gain * (layer_height / k.nozzle_diameter_mm) ** 2
        + k.underextrusion_porosity_gain * under
        + k.moisture_porosity_gain * moisture_pct
        + 0.05 * degradation
    )
    return np.asarray(np.clip(p, 0.0, k.max_porosity), dtype=float)


def tsai_hill_factor(angle_deg: FloatArray, transverse_ratio: FloatArray, shear_ratio: FloatArray) -> FloatArray:
    """Off-axis strength relative to the longitudinal strength (Tsai–Hill criterion).

    1 / s(theta)^2 = cos^4 + (1/q^2 - 1) cos^2 sin^2 + sin^4 / t^2, where t and q are the
    transverse and shear strengths divided by the longitudinal strength. Raster angle is
    measured from the load axis; 0 degrees is strongest, as reported by [AHN-2002].
    """
    theta = np.deg2rad(np.asarray(angle_deg, dtype=float))
    c2 = np.cos(theta) ** 2
    s2 = np.sin(theta) ** 2
    t = np.maximum(np.asarray(transverse_ratio, dtype=float), 1e-6)
    q = np.maximum(np.asarray(shear_ratio, dtype=float), 1e-6)
    inv_sq = c2**2 + (1.0 / q**2 - 1.0) * c2 * s2 + s2**2 / t**2
    return np.asarray(1.0 / np.sqrt(inv_sq), dtype=float)


def section_fraction(infill_pct: FloatArray, perimeter_fraction: float = 0.35) -> FloatArray:
    """Load-bearing cross-section fraction of a tensile bar with solid perimeters."""
    return np.asarray(perimeter_fraction + (1.0 - perimeter_fraction) * infill_pct / 100.0, dtype=float)


def print_time_proxy(print_speed: FloatArray, layer_height: FloatArray, infill_pct: FloatArray) -> FloatArray:
    """Relative print time for a fixed part: deposited volume / volumetric rate.

    Normalised so that 40 mm/s, 0.2 mm, 100 % infill gives 1.0.
    """
    volume = section_fraction(infill_pct)
    rate = print_speed * layer_height
    return np.asarray(volume / rate / (1.0 / (40.0 * 0.2)), dtype=float)


def energy_proxy(
    nozzle_temp: FloatArray,
    bed_temp: FloatArray,
    print_speed: FloatArray,
    layer_height: FloatArray,
    infill_pct: FloatArray,
) -> FloatArray:
    """Relative energy: print time times heater load, with load proportional to the
    temperature rise above ambient (bed weighted 3x for its larger area; ASSUMPTION).

    Normalised so that 230 °C nozzle and 70 °C bed at the reference time give 1.0.
    """
    load = (nozzle_temp - 25.0) + 3.0 * (bed_temp - 25.0)
    ref_load = (230.0 - 25.0) + 3.0 * (70.0 - 25.0)
    return np.asarray(print_time_proxy(print_speed, layer_height, infill_pct) * load / ref_load, dtype=float)
