/**
 * Ground-truth simulator (exact port of robustprint.physics.{mixing,process,simulator}).
 *
 * Every function is scalar (one composition + one setting). Parity with the Python
 * implementation is checked against golden vectors in tests/simulator.test.ts.
 * This is a SYNTHETIC world with assumed coefficients, not a validated material model.
 */

import { closure } from "./composition";
import { CHI, COMPONENTS, type Feedstock, N_COMPONENTS } from "./materials";
import { sigmoid, smoothMinOne, softplus } from "./math";
import type { Rng } from "./rng";

export const CONFIG = {
  kappaStrength: 0.5,
  kappaModulus: 0.25,
  kappaElongation: 1.5,
  meltWidthC: 6,
  fiberEfficiency: 0.2,
  fiberAgglomerationVf: 0.2,
  halpinTsaiXi: 4,
  degradationScaleC: 10,
  degradationWeight: 1,
  hydrolysisSensitivity: { pet: 1.5, pa: 1.5 } as Record<string, number>,
  hydrolysisOnsetC: 240,
  clogIntercept: -6,
  clogFiber: 18,
  clogContaminant: 10,
  clogUnmelted: 12,
  clogFlowLimit: 3,
  clogFiberFlow: 40,
  clogDegradation: 1,
  warpIntercept: -4.5,
  warpPe: 3,
  warpBedPerC: 0.06,
  warpBedRefC: 70,
  warpFanPe: 3,
  noiseCvBase: 0.04,
  noiseCvContaminant: 0.3,
  noiseCvFiber: 0.2,
  noiseCvPorosity: 0.3,
  batchEffectSd: 0.05,
};

export const PROCESS = {
  ambientC: 25,
  bedCoupling: 0.3,
  tauCoolS: 1.5,
  fanCoolingGain: 2,
  layerPathMm: 600,
  tauLayerS: 8,
  tRepRefS: 0.4,
  tRepRefC: 230,
  activationJMol: 60000,
  bondSoftminPower: 4,
  lineWidthMm: 0.45,
  nozzleDiameterMm: 0.4,
  qMaxRef: 9,
  qMaxRefC: 220,
  qMaxTempScaleC: 60,
  fiberViscosityGain: 6,
  unmeltedViscosityGain: 3,
  extrusionSoftminPower: 6,
  basePorosity: 0.03,
  layerPorosityGain: 0.08,
  underextrusionPorosityGain: 0.9,
  moisturePorosityGain: 0.03,
  maxPorosity: 0.6,
};

const GAS_CONSTANT = 8.314;
const KELVIN = 273.15;
const IDX: Record<string, number> = Object.fromEntries(COMPONENTS.map((c, i) => [c.name, i]));

export interface Outcome {
  strengthMPa: number;
  modulusGPa: number;
  elongationPct: number;
  pSuccess: number;
  pClog: number;
  pWarp: number;
  noiseCv: number;
  bond: number;
  porosity: number;
  extrusion: number;
  demandRatio: number;
  immiscibility: number;
  degradation: number;
  hydrolysis: number;
  unmelted: number;
  fiberVolume: number;
  inclusionVolume: number;
  /** Multiplicative strength factors, in the order they are applied (for explanation). */
  factors: { blend: number; fibre: number; inclusions: number; chemistry: number; bonding: number; density: number; raster: number; infill: number; overExtrusion: number; batch: number };
}

export function tsaiHillFactor(angleDeg: number, transverse: number, shear: number): number {
  const th = (angleDeg * Math.PI) / 180;
  const c2 = Math.cos(th) ** 2;
  const s2 = Math.sin(th) ** 2;
  const t = Math.max(transverse, 1e-6);
  const q = Math.max(shear, 1e-6);
  return 1 / Math.sqrt(c2 * c2 + (1 / (q * q) - 1) * c2 * s2 + (s2 * s2) / (t * t));
}

export function sectionFraction(infillPct: number, perimeter = 0.35): number {
  return perimeter + ((1 - perimeter) * infillPct) / 100;
}

export function printTimeProxy(speed: number, layerHeight: number, infillPct: number): number {
  return sectionFraction(infillPct) / (speed * layerHeight) / (1 / (40 * 0.2));
}

export function energyProxy(nozzle: number, bed: number, speed: number, layerHeight: number, infillPct: number): number {
  const load = nozzle - 25 + 3 * (bed - 25);
  const ref = 230 - 25 + 3 * (70 - 25);
  return (printTimeProxy(speed, layerHeight, infillPct) * load) / ref;
}

function bondEfficiency(T: number, bed: number, speed: number, h: number, fan: number, tCrit: number): number {
  const k = PROCESS;
  const envelope = k.ambientC + k.bedCoupling * (bed - k.ambientC);
  const tLayer = k.layerPathMm / Math.max(speed, 1e-3);
  const substrate = envelope + (T - envelope) * Math.exp(-tLayer / k.tauLayerS);
  const tau = (k.tauCoolS * (h / 0.2)) / (1 + (k.fanCoolingGain * fan) / 100);
  const excess = Math.max(T - substrate, 1e-6);
  const critExcess = Math.max(tCrit - substrate, 1e-6);
  const tAbove = tau * Math.log(Math.max(excess / critExcess, 1));
  const tMeanK = 0.5 * (T + Math.max(tCrit, substrate)) + KELVIN;
  const tRefK = k.tRepRefC + KELVIN;
  const tRep = k.tRepRefS * Math.exp((k.activationJMol / GAS_CONSTANT) * (1 / tMeanK - 1 / tRefK));
  return smoothMinOne(Math.pow(tAbove / tRep, 0.25), k.bondSoftminPower);
}

/** Noise-free expected outcomes for one (composition, settings, moisture, batch effect). */
export function trueOutcome(composition: ArrayLike<number>, params: ArrayLike<number>, moisture: number, batchEffect = 0): Outcome {
  const cfg = CONFIG;
  const k = PROCESS;
  const c = closure(composition);
  const [T, bed, speed, h, infill, angle, flow, fan] = Array.from(params);

  // Phase state of the melt.
  let volMeltedTot = 0;
  let volUnmelted = 0;
  let volFiber = 0;
  let volIncl = 0;
  let unmeltedMass = 0;
  const volMelted = new Array<number>(N_COMPONENTS).fill(0);
  let nPoly = 0;
  COMPONENTS.forEach((comp, i) => {
    if (comp.kind === "polymer") {
      nPoly += 1;
      const un = c[i] * sigmoid((comp.tMin - T) / cfg.meltWidthC);
      unmeltedMass += un;
      volMelted[i] = (c[i] - un) / comp.density;
      volMeltedTot += volMelted[i];
      volUnmelted += un / comp.density;
    } else if (comp.kind === "fiber") volFiber += c[i] / comp.density;
    else volIncl += c[i] / comp.density;
  });
  const total = volMeltedTot + volUnmelted + volFiber + volIncl;
  const w = COMPONENTS.map((comp, i) =>
    volMeltedTot > 1e-9 ? volMelted[i] / Math.max(volMeltedTot, 1e-12) : comp.kind === "polymer" ? 1 / nPoly : 0,
  );
  const vf = volFiber / total;
  const vp = (volIncl + volUnmelted) / total;
  const unmeltedVol = volUnmelted / total;
  const tCrit = COMPONENTS.reduce((s, comp, i) => s + w[i] * (Number.isNaN(comp.tCrit) ? 0 : comp.tCrit), 0);

  // Blend and composite mechanics.
  let imm = 0;
  for (let i = 0; i < N_COMPONENTS; i++) for (let j = 0; j < N_COMPONENTS; j++) imm += w[i] * CHI[i][j] * w[j];
  const rom = COMPONENTS.reduce((s, comp, i) => s + w[i] * comp.strengthMPa, 0);
  const penalty = Math.min(Math.max(1 - cfg.kappaStrength * imm, 0.05), 1);
  const sigmaMatrix = rom * penalty;
  const voigt = COMPONENTS.reduce((s, comp, i) => s + w[i] * comp.modulusGPa, 0);
  const reuss = 1 / COMPONENTS.reduce((s, comp, i) => s + w[i] / comp.modulusGPa, 0);
  const modulusMatrix = 0.5 * (voigt + reuss) * (1 - cfg.kappaModulus * imm);
  const fiber = COMPONENTS[IDX.fiber];

  const sigmaFibre = (cfg.fiberEfficiency * fiber.strengthMPa * vf + sigmaMatrix * (1 - vf)) * Math.exp(-((vf / cfg.fiberAgglomerationVf) ** 2));
  const phi = Math.min(Math.max(vp, 0), 0.6);
  const inclusionFactor = Math.min(Math.max(1 - 1.21 * Math.pow(phi, 2 / 3), 0.05), 1);
  const sigmaComp = sigmaFibre * inclusionFactor;
  const ratio = fiber.modulusGPa / modulusMatrix;
  const eta = (ratio - 1) / (ratio + cfg.halpinTsaiXi);
  const modulusComp = ((modulusMatrix * (1 + cfg.halpinTsaiXi * eta * vf)) / (1 - eta * vf)) * (1 - vp) ** 2;

  // Chemistry: degradation and hydrolysis.
  const degradation = COMPONENTS.reduce((s, comp, i) => s + c[i] * softplus((T - comp.tDeg) / cfg.degradationScaleC), 0);
  const sens = COMPONENTS.reduce((s, comp, i) => s + (cfg.hydrolysisSensitivity[comp.name] ?? 0) * c[i], 0);
  const hydrolysis = moisture * sens * sigmoid((T - cfg.hydrolysisOnsetC) / 10);
  const chemistry = Math.exp(-cfg.degradationWeight * degradation - hydrolysis);

  // Extrusion, porosity, bonding.
  const qDemand = (speed * h * k.lineWidthMm * flow) / 100;
  const viscosity = 1 + k.fiberViscosityGain * vf + k.unmeltedViscosityGain * unmeltedVol;
  const qMax = (k.qMaxRef * Math.exp((T - k.qMaxRefC) / k.qMaxTempScaleC)) / viscosity;
  const extrusion = smoothMinOne(qMax / Math.max(qDemand, 1e-9), k.extrusionSoftminPower);
  const demandRatio = qDemand / qMax;
  const delivered = (flow / 100) * extrusion;
  const under = softplus(20 * (1 - delivered)) / 20;
  const pores = Math.min(
    Math.max(
      k.basePorosity + k.layerPorosityGain * (h / k.nozzleDiameterMm) ** 2 + k.underextrusionPorosityGain * under + k.moisturePorosityGain * moisture + 0.05 * degradation,
      0,
    ),
    k.maxPorosity,
  );
  const over = softplus(20 * (delivered - 1.08)) / 20;
  const bond = bondEfficiency(T, bed, speed, h, fan, tCrit);

  const dense = (1 - pores) ** 1.5;
  const longitudinal = 0.55 + 0.4 * bond;
  const orient = tsaiHillFactor(angle, (0.75 * bond) / longitudinal, (0.5 * bond) / longitudinal);
  const section = sectionFraction(infill);
  const overFactor = Math.exp(-4 * over);
  const batch = Math.exp(batchEffect);
  const strength = sigmaComp * chemistry * longitudinal * dense * orient * section * overFactor * batch;

  const modOrient = tsaiHillFactor(angle, 0.8, 0.6);
  const modulus = modulusComp * chemistry ** 0.3 * (1 - pores) ** 2 * (0.85 + 0.15 * bond) * modOrient * section * Math.exp(0.5 * batchEffect);

  const logElong = COMPONENTS.reduce((s, comp, i) => s + w[i] * Math.log(comp.elongationPct), 0);
  const elongation =
    Math.exp(logElong) *
    Math.min(Math.max(1 - imm, 0), 1) ** cfg.kappaElongation *
    Math.exp(-10 * vf) *
    inclusionFactor ** 2 *
    (0.3 + 0.7 * bond) *
    (1 - pores) ** 2 *
    chemistry;

  const cFiber = c[IDX.fiber];
  const cIncl = c[IDX.contaminant];
  const cPe = c[IDX.pe];
  const zClog =
    cfg.clogIntercept +
    cfg.clogFiber * cFiber +
    cfg.clogContaminant * cIncl +
    cfg.clogUnmelted * unmeltedMass +
    (cfg.clogFlowLimit * softplus(4 * (demandRatio - 1))) / 4 +
    cfg.clogFiberFlow * cFiber * demandRatio +
    cfg.clogDegradation * degradation;
  const zWarp = cfg.warpIntercept + cfg.warpPe * cPe - cfg.warpBedPerC * (bed - cfg.warpBedRefC) + (cfg.warpFanPe * cPe * fan) / 100;
  const pClog = sigmoid(zClog);
  const pWarp = sigmoid(zWarp);
  const noiseCv = cfg.noiseCvBase + cfg.noiseCvContaminant * cIncl + cfg.noiseCvFiber * cFiber + cfg.noiseCvPorosity * pores;

  return {
    strengthMPa: strength,
    modulusGPa: modulus,
    elongationPct: elongation,
    pSuccess: (1 - pClog) * (1 - pWarp),
    pClog,
    pWarp,
    noiseCv,
    bond,
    porosity: pores,
    extrusion,
    demandRatio,
    immiscibility: imm,
    degradation,
    hydrolysis,
    unmelted: unmeltedMass,
    fiberVolume: vf,
    inclusionVolume: vp,
    factors: {
      blend: sigmaMatrix,
      fibre: sigmaFibre / sigmaMatrix,
      inclusions: inclusionFactor,
      chemistry,
      bonding: longitudinal,
      density: dense,
      raster: orient,
      infill: section,
      overExtrusion: overFactor,
      batch,
    },
  };
}

export interface Batch {
  id: string;
  trueComposition: number[];
  reportedComposition: number[];
  moisture: number;
  batchEffect: number;
}

function clampedDirichlet(rng: Rng, alpha: number[]): number[] {
  return closure(rng.dirichlet(alpha.map((a) => Math.max(a, 1e-3))).map((v) => Math.max(v, 1e-6)));
}

/** Draw batches from a feedstock (true composition, noisy reported composition, moisture, batch effect). */
export function sampleBatches(feedstock: Feedstock, n: number, rng: Rng, measurementConcentration = 300, prefix = "B"): Batch[] {
  const mu = closure(feedstock.mean);
  const out: Batch[] = [];
  for (let i = 0; i < n; i++) {
    const trueComposition = clampedDirichlet(rng, mu.map((m) => feedstock.concentration * m));
    const reportedComposition = clampedDirichlet(rng, trueComposition.map((v) => measurementConcentration * v));
    const moisture = Math.min(Math.max(feedstock.moistureMedianPct * Math.exp(feedstock.moistureLogSd * rng.normal()), 0.02), 3);
    const batchEffect = CONFIG.batchEffectSd * rng.normal();
    out.push({ id: `${prefix}${String(i).padStart(4, "0")}`, trueComposition, reportedComposition, moisture, batchEffect });
  }
  return out;
}

export interface Specimen {
  success: boolean;
  strengthMPa: number;
  modulusGPa: number;
  elongationPct: number;
}

/** One realised specimen: success draw and noisy measured properties (NaN if failed). */
export function realise(batch: Batch, params: ArrayLike<number>, rng: Rng, outcome?: Outcome): Specimen {
  const t = outcome ?? trueOutcome(batch.trueComposition, params, batch.moisture, batch.batchEffect);
  const success = rng.random() < t.pSuccess;
  const cv = t.noiseCv;
  const strength = t.strengthMPa * Math.exp(cv * rng.normal() - 0.5 * cv * cv);
  const modulus = t.modulusGPa * Math.exp(0.6 * cv * rng.normal());
  const elongation = t.elongationPct * Math.exp(1.5 * cv * rng.normal());
  return success
    ? { success, strengthMPa: strength, modulusGPa: modulus, elongationPct: elongation }
    : { success, strengthMPa: NaN, modulusGPa: NaN, elongationPct: NaN };
}

export interface JobResult {
  success: boolean;
  strengthMPa: number[];
  modulusGPa: number[];
  elongationPct: number[];
}

/**
 * One print job of n specimens: the job succeeds or fails as a whole; specimens of a
 * successful job differ only by measurement noise (same semantics as Python print_job).
 */
export function printJob(batch: Batch, params: ArrayLike<number>, rng: Rng, nReplicates: number, outcome?: Outcome): JobResult {
  const t = outcome ?? trueOutcome(batch.trueComposition, params, batch.moisture, batch.batchEffect);
  const success = rng.random() < t.pSuccess;
  const cv = t.noiseCv;
  const strengthMPa: number[] = [];
  const modulusGPa: number[] = [];
  const elongationPct: number[] = [];
  for (let r = 0; r < nReplicates; r++) {
    strengthMPa.push(success ? t.strengthMPa * Math.exp(cv * rng.normal() - 0.5 * cv * cv) : NaN);
    modulusGPa.push(success ? t.modulusGPa * Math.exp(0.6 * cv * rng.normal()) : NaN);
    elongationPct.push(success ? t.elongationPct * Math.exp(1.5 * cv * rng.normal()) : NaN);
  }
  return { success, strengthMPa, modulusGPa, elongationPct };
}
