"""Ground-truth simulator: composition + print settings -> mechanical outcomes.

This is the "true world" of all synthetic experiments. It is deliberately transparent:
every step is a short, documented function (see docs/METHODS.md, section 2). It is
**not** a validated model of any real material. Its purpose is to produce data with the
structure we expect from real recycled feedstock (blend immiscibility, fibre and
contaminant effects, thermal bonding, extrusion limits, batch effects, heteroscedastic
noise, print failures) and, crucially, to provide a known ground truth so that the
realised performance and regret of every optimiser can be computed exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robustprint.physics import mixing, process
from robustprint.physics.composition import FloatArray, closure
from robustprint.physics.materials import DEFAULT_LIBRARY, MaterialLibrary
from robustprint.physics.process import ProcessConstants, sigmoid, softplus


@dataclass(frozen=True)
class SimulatorConfig:
    """All coefficients of the ground-truth simulator.

    Values marked ASSUMPTION in docs/DECISIONS.md (ADR-007 to ADR-012).
    """

    library: MaterialLibrary = DEFAULT_LIBRARY
    process: ProcessConstants = field(default_factory=ProcessConstants)
    kappa_strength: float = 0.5
    kappa_modulus: float = 0.25
    kappa_elongation: float = 1.5
    melt_width_C: float = 6.0
    fiber_efficiency: float = 0.2
    fiber_agglomeration_vf: float = 0.2
    halpin_tsai_xi: float = 4.0
    degradation_scale_C: float = 10.0
    degradation_weight: float = 1.0
    hydrolysis_sensitivity: dict[str, float] = field(default_factory=lambda: {"pet": 1.5, "pa": 1.5})
    hydrolysis_onset_C: float = 240.0
    clog_intercept: float = -6.0
    clog_fiber: float = 18.0
    clog_contaminant: float = 10.0
    clog_unmelted: float = 12.0
    clog_flow_limit: float = 3.0
    clog_fiber_flow: float = 40.0
    clog_degradation: float = 1.0
    warp_intercept: float = -4.5
    warp_pe: float = 3.0
    warp_bed_per_C: float = 0.06
    warp_bed_ref_C: float = 70.0
    warp_fan_pe: float = 3.0
    noise_cv_base: float = 0.04
    noise_cv_contaminant: float = 0.3
    noise_cv_fiber: float = 0.2
    noise_cv_porosity: float = 0.3
    batch_effect_sd: float = 0.05


DEFAULT_CONFIG = SimulatorConfig()


@dataclass(frozen=True)
class Feedstock:
    """A feedstock source: the distribution that batch compositions are drawn from.

    Attributes:
        name: Identifier.
        mean: Mean composition (fractions in library order).
        concentration: Dirichlet concentration; larger means less batch-to-batch spread.
        moisture_median_pct: Median moisture content of a batch.
        moisture_log_sd: Log-normal spread of moisture between batches.
    """

    name: str
    mean: tuple[float, ...]
    concentration: float
    moisture_median_pct: float = 0.3
    moisture_log_sd: float = 0.6

    def mean_array(self) -> FloatArray:
        """Mean composition as a closed array."""
        return closure(np.array(self.mean))


#: ISS-like mixed waste: mostly films, some laminates, textile, fibres and residues.
ISS_NOMINAL = Feedstock("iss_nominal", (0.45, 0.25, 0.15, 0.08, 0.07), concentration=35.0)
#: Shifted (out-of-distribution) stream: more fibre and contaminant, less polyethylene.
ISS_SHIFTED = Feedstock("iss_shifted", (0.30, 0.30, 0.12, 0.16, 0.12), concentration=35.0)

FEEDSTOCKS: dict[str, Feedstock] = {f.name: f for f in (ISS_NOMINAL, ISS_SHIFTED)}


def _column(lib: MaterialLibrary, composition: FloatArray, name: str) -> FloatArray:
    """Fraction of one named component (zeros if the library lacks it)."""
    if name not in lib.names:
        return np.zeros(composition.shape[0])
    return np.asarray(composition[:, lib.names.index(name)], dtype=float)


def material_state(
    composition: FloatArray, nozzle_temp: FloatArray, cfg: SimulatorConfig = DEFAULT_CONFIG
) -> dict[str, FloatArray]:
    """Phase state of the melt at a given nozzle temperature.

    Returns volume fractions of fibre (``vf``) and inclusions (``vp``: contaminants plus
    unmelted polymer), polymer volume weights of the molten matrix (``w``), the unmelted
    mass fraction and the matrix critical sintering temperature.
    """
    lib = cfg.library
    c = closure(np.atleast_2d(composition))
    t = np.asarray(nozzle_temp, dtype=float).reshape(-1, 1)
    is_poly = lib.mask("polymer")
    t_min = np.nan_to_num(lib.vector("t_min_C"), nan=0.0)
    unmelted = c * is_poly * sigmoid((t_min - t) / cfg.melt_width_C)
    melted = c * is_poly - unmelted
    density = lib.vector("density_g_cm3")
    vol_melted = melted / density
    vol_unmelted = unmelted / density
    vol_fiber = c * lib.mask("fiber") / density
    vol_incl = c * lib.mask("inclusion") / density
    total = vol_melted.sum(1) + vol_unmelted.sum(1) + vol_fiber.sum(1) + vol_incl.sum(1)
    matrix_vol = vol_melted.sum(1, keepdims=True)
    w = np.where(matrix_vol > 1e-9, vol_melted / np.maximum(matrix_vol, 1e-12), is_poly / is_poly.sum())
    t_crit = np.nan_to_num(lib.vector("t_crit_C"), nan=0.0)
    return {
        "vf": vol_fiber.sum(1) / total,
        "vp": (vol_incl.sum(1) + vol_unmelted.sum(1)) / total,
        "w": w,
        "unmelted": unmelted.sum(1),
        "unmelted_vol": vol_unmelted.sum(1) / total,
        "t_crit": w @ t_crit,
    }


def true_outcomes(
    composition: FloatArray,
    params: FloatArray,
    moisture_pct: FloatArray,
    batch_effect: FloatArray | None = None,
    cfg: SimulatorConfig = DEFAULT_CONFIG,
) -> dict[str, FloatArray]:
    """Noise-free expected outcomes for each (composition, settings) row.

    Args:
        composition: (n, D) true composition fractions.
        params: (n, 8) print parameters in ``robustprint.space.PARAM_NAMES`` order.
        moisture_pct: (n,) batch moisture content in percent.
        batch_effect: (n,) log-scale multiplicative batch effect on strength (default 0).
        cfg: Simulator configuration.

    Returns:
        Dictionary with ``strength_MPa``, ``modulus_GPa``, ``elongation_pct``,
        ``p_success``, the specimen-level coefficient of variation ``noise_cv`` and the
        intermediate physical quantities (``bond``, ``porosity``, ``extrusion`` ...).
    """
    lib = cfg.library
    k = cfg.process
    c = closure(np.atleast_2d(np.asarray(composition, dtype=float)))
    x = np.atleast_2d(np.asarray(params, dtype=float))
    n = c.shape[0]
    moisture = np.broadcast_to(np.asarray(moisture_pct, dtype=float), (n,))
    b = np.zeros(n) if batch_effect is None else np.broadcast_to(np.asarray(batch_effect, dtype=float), (n,))
    temp, bed, speed, height, infill, angle, flow, fan = (x[:, i] for i in range(8))

    state = material_state(c, temp, cfg)
    vf, vp, w = state["vf"], state["vp"], state["w"]
    chi = lib.chi_matrix()
    imm = mixing.immiscibility_index(w, chi)

    sigma_matrix = mixing.blend_strength(w, lib.vector("strength_MPa"), chi, cfg.kappa_strength)
    modulus_matrix = mixing.hill_average(w, lib.vector("modulus_GPa")) * (1.0 - cfg.kappa_modulus * imm)
    fiber_idx = lib.mask("fiber").astype(bool)
    fiber_strength = float(lib.vector("strength_MPa")[fiber_idx][0]) if fiber_idx.any() else 0.0
    fiber_modulus = float(lib.vector("modulus_GPa")[fiber_idx][0]) if fiber_idx.any() else 1.0

    sigma_comp = mixing.short_fiber_strength(
        sigma_matrix, fiber_strength, vf, cfg.fiber_efficiency, cfg.fiber_agglomeration_vf
    )
    sigma_comp = sigma_comp * mixing.inclusion_strength_factor(vp)
    modulus_comp = mixing.halpin_tsai_modulus(modulus_matrix, fiber_modulus, vf, cfg.halpin_tsai_xi)
    modulus_comp = modulus_comp * (1.0 - vp) ** 2

    t_deg = lib.vector("t_deg_C")
    degradation = (c * softplus((temp[:, None] - t_deg) / cfg.degradation_scale_C)).sum(1)
    sensitivity = sum(
        (cfg.hydrolysis_sensitivity.get(name, 0.0) * _column(lib, c, name) for name in lib.names),
        start=np.zeros(n),
    )
    hydrolysis = moisture * sensitivity * sigmoid((temp - cfg.hydrolysis_onset_C) / 10.0)
    chemistry = np.exp(-cfg.degradation_weight * degradation - hydrolysis)

    extrusion, demand_ratio = process.extrusion_ratio(temp, speed, height, flow, vf, state["unmelted_vol"], k)
    delivered = flow / 100.0 * extrusion
    pores = process.porosity(height, delivered, moisture, degradation, k)
    over = softplus(20.0 * (delivered - 1.08)) / 20.0
    bond = process.bond_efficiency(temp, bed, speed, height, fan, state["t_crit"], k)

    dense = (1.0 - pores) ** 1.5
    longitudinal = 0.55 + 0.40 * bond
    orient = process.tsai_hill_factor(angle, 0.75 * bond / longitudinal, 0.5 * bond / longitudinal)
    section = process.section_fraction(infill)
    strength = sigma_comp * chemistry * longitudinal * dense * orient * section * np.exp(-4.0 * over) * np.exp(b)

    mod_orient = process.tsai_hill_factor(angle, np.full(n, 0.8), np.full(n, 0.6))
    modulus = (
        modulus_comp
        * chemistry**0.3
        * (1.0 - pores) ** 2
        * (0.85 + 0.15 * bond)
        * mod_orient
        * section
        * np.exp(0.5 * b)
    )

    log_elong = w @ np.log(lib.vector("elongation_pct"))
    elongation = (
        np.exp(log_elong)
        * np.clip(1.0 - imm, 0.0, 1.0) ** cfg.kappa_elongation
        * np.exp(-10.0 * vf)
        * mixing.inclusion_strength_factor(vp) ** 2
        * (0.3 + 0.7 * bond)
        * (1.0 - pores) ** 2
        * chemistry
    )

    c_fiber = (c * lib.mask("fiber")).sum(1)
    c_incl = (c * lib.mask("inclusion")).sum(1)
    c_pe = _column(lib, c, "pe")
    z_clog = (
        cfg.clog_intercept
        + cfg.clog_fiber * c_fiber
        + cfg.clog_contaminant * c_incl
        + cfg.clog_unmelted * state["unmelted"]
        + cfg.clog_flow_limit * softplus(4.0 * (demand_ratio - 1.0)) / 4.0
        + cfg.clog_fiber_flow * c_fiber * demand_ratio
        + cfg.clog_degradation * degradation
    )
    z_warp = (
        cfg.warp_intercept
        + cfg.warp_pe * c_pe
        - cfg.warp_bed_per_C * (bed - cfg.warp_bed_ref_C)
        + cfg.warp_fan_pe * c_pe * fan / 100.0
    )
    p_success = (1.0 - sigmoid(z_clog)) * (1.0 - sigmoid(z_warp))

    noise_cv = (
        cfg.noise_cv_base
        + cfg.noise_cv_contaminant * c_incl
        + cfg.noise_cv_fiber * c_fiber
        + cfg.noise_cv_porosity * pores
    )
    return {
        "strength_MPa": strength,
        "modulus_GPa": modulus,
        "elongation_pct": elongation,
        "p_success": p_success,
        "p_clog": sigmoid(z_clog),
        "p_warp": sigmoid(z_warp),
        "noise_cv": noise_cv,
        "bond": bond,
        "porosity": pores,
        "extrusion": extrusion,
        "immiscibility": imm,
        "degradation": degradation,
        "unmelted": state["unmelted"],
        "fiber_volume": vf,
        "inclusion_volume": vp,
    }


@dataclass(frozen=True)
class Batches:
    """A set of simulated feedstock batches.

    Attributes:
        batch_id: String identifiers.
        true_composition: (n, D) composition used by the ground truth.
        reported_composition: (n, D) composition a lab analysis would report (noisy).
        moisture_pct: (n,) moisture content.
        batch_effect: (n,) unobserved log-scale strength effect.
    """

    batch_id: list[str]
    true_composition: FloatArray
    reported_composition: FloatArray
    moisture_pct: FloatArray
    batch_effect: FloatArray

    def __len__(self) -> int:
        return len(self.batch_id)


def sample_batches(
    feedstock: Feedstock,
    n: int,
    rng: np.random.Generator,
    measurement_concentration: float = 300.0,
    cfg: SimulatorConfig = DEFAULT_CONFIG,
    prefix: str = "B",
) -> Batches:
    """Draw ``n`` batches from a feedstock.

    True compositions are Dirichlet(concentration * mean). Reported compositions add
    analysis error as Dirichlet(measurement_concentration * true). Moisture is
    log-normal and clipped to [0.02, 3] %. The batch effect is N(0, batch_effect_sd^2).
    """
    alpha = np.maximum(feedstock.concentration * feedstock.mean_array(), 1e-3)
    true_comp = closure(np.maximum(rng.dirichlet(alpha, size=n), 1e-6))
    reported = closure(
        np.maximum(
            np.array([rng.dirichlet(np.maximum(measurement_concentration * row, 1e-3)) for row in true_comp]),
            1e-6,
        )
    )
    moisture = np.clip(
        feedstock.moisture_median_pct * np.exp(feedstock.moisture_log_sd * rng.standard_normal(n)), 0.02, 3.0
    )
    effect = cfg.batch_effect_sd * rng.standard_normal(n)
    ids = [f"{prefix}{i:04d}" for i in range(n)]
    return Batches(ids, true_comp, reported, moisture, effect)


def realise(
    composition: FloatArray,
    params: FloatArray,
    moisture_pct: FloatArray,
    batch_effect: FloatArray,
    rng: np.random.Generator,
    cfg: SimulatorConfig = DEFAULT_CONFIG,
) -> dict[str, FloatArray]:
    """Draw one realised specimen per row: success flag and noisy measured properties.

    Failed prints have NaN mechanical properties.
    """
    truth = true_outcomes(composition, params, moisture_pct, batch_effect, cfg)
    n = truth["strength_MPa"].shape[0]
    success = rng.random(n) < truth["p_success"]
    cv = truth["noise_cv"]
    strength = truth["strength_MPa"] * np.exp(cv * rng.standard_normal(n) - 0.5 * cv**2)
    modulus = truth["modulus_GPa"] * np.exp(0.6 * cv * rng.standard_normal(n))
    elongation = truth["elongation_pct"] * np.exp(1.5 * cv * rng.standard_normal(n))
    nan = np.full(n, np.nan)
    return {
        "success": success,
        "strength_MPa": np.where(success, strength, nan),
        "modulus_GPa": np.where(success, modulus, nan),
        "elongation_pct": np.where(success, elongation, nan),
    }


def print_job(
    composition: FloatArray,
    params: FloatArray,
    moisture_pct: FloatArray,
    batch_effect: FloatArray,
    rng: np.random.Generator,
    n_replicates: int,
    cfg: SimulatorConfig = DEFAULT_CONFIG,
) -> dict[str, FloatArray]:
    """Simulate print jobs of ``n_replicates`` specimens each.

    A job succeeds or fails as a whole (one Bernoulli draw per job). The specimens of a
    successful job differ only by measurement noise. Arrays of specimen properties have
    shape (n_replicates, n) and are NaN for failed jobs.
    """
    truth = true_outcomes(composition, params, moisture_pct, batch_effect, cfg)
    n = truth["strength_MPa"].shape[0]
    success = rng.random(n) < truth["p_success"]
    cv = truth["noise_cv"]
    z = rng.standard_normal((3, n_replicates, n))
    mask = np.where(success, 1.0, np.nan)
    return {
        "success": success,
        "strength_MPa": truth["strength_MPa"] * np.exp(cv * z[0] - 0.5 * cv**2) * mask,
        "modulus_GPa": truth["modulus_GPa"] * np.exp(0.6 * cv * z[1]) * mask,
        "elongation_pct": truth["elongation_pct"] * np.exp(1.5 * cv * z[2]) * mask,
    }
