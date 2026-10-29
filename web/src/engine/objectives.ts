/**
 * Robust objectives by Monte Carlo with common random numbers (port of
 * robustprint.optimize.objectives). Strength samples are weighted by the predicted success
 * probability (conditioning on parts that print); all candidates share the same draws.
 */

import { closure } from "./composition";
import type { Feedstock } from "./materials";
import type { Predictor } from "./predictor";
import { Rng } from "./rng";

export const OBJECTIVES = ["nominal", "expected", "quantile", "cvar", "chance"] as const;
export type Objective = (typeof OBJECTIVES)[number];

export const OBJECTIVE_LABELS: Record<Objective, string> = {
  nominal: "Nominal (mean composition)",
  expected: "Expected over composition",
  quantile: "5 % quantile",
  cvar: "CVaR (worst 10 %)",
  chance: "Chance-constrained",
};

export interface ObjectiveSettings {
  quantileLevel: number;
  cvarAlpha: number;
  pMin: number;
  chanceThresholdMPa: number;
  chanceProb: number;
  penaltyMPa: number;
}

export const DEFAULT_SETTINGS: ObjectiveSettings = {
  quantileLevel: 0.05,
  cvarAlpha: 0.1,
  pMin: 0.9,
  chanceThresholdMPa: 5,
  chanceProb: 0.9,
  penaltyMPa: 100,
};

export interface FeedstockSamples {
  comps: number[][];
  moisture: number[];
}

export function sampleFeedstock(feedstock: Feedstock, n: number, rng: Rng): FeedstockSamples {
  const mu = closure(feedstock.mean);
  const comps: number[][] = [];
  const moisture: number[] = [];
  for (let i = 0; i < n; i++) {
    comps.push(closure(rng.dirichlet(mu.map((m) => Math.max(feedstock.concentration * m, 1e-3))).map((v) => Math.max(v, 1e-6))));
    moisture.push(Math.min(Math.max(feedstock.moistureMedianPct * Math.exp(feedstock.moistureLogSd * rng.normal()), 0.02), 3));
  }
  return { comps, moisture };
}

/** Weighted lower quantile of values (weights need not sum to one). */
export function weightedQuantile(values: number[], weights: number[], q: number): number {
  const idx = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const total = weights.reduce((s, w) => s + w, 0);
  let cum = 0;
  for (let k = 0; k < idx.length; k++) {
    cum += weights[idx[k]] / total;
    if (cum >= q) return values[idx[k]];
  }
  return values[idx[idx.length - 1]];
}

/** Weighted lower-tail CVaR: mean of the lowest alpha probability mass. */
export function weightedCvar(values: number[], weights: number[], alpha: number): number {
  const idx = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const total = weights.reduce((s, w) => s + w, 0);
  let cum = 0;
  let acc = 0;
  for (let k = 0; k < idx.length; k++) {
    const w = weights[idx[k]] / total;
    const take = Math.min(Math.max(alpha - cum, 0), w);
    acc += take * values[idx[k]];
    cum += w;
    if (cum >= alpha) break;
  }
  return acc / alpha;
}

export interface MixtureStats {
  mean: number;
  quantile: number;
  cvar: number;
  pExceed: number;
  pSuccess: number;
  /** Variance decomposition of log strength over the feedstock draws. */
  varInput: number;
  varEpistemic: number;
  varAleatoric: number;
  meanLog: number;
}

export interface NominalStats {
  nominalMean: number;
  nominalPSuccess: number;
}

export class RobustProblem {
  readonly samples: FeedstockSamples;
  readonly normals: number[][];
  readonly settings: ObjectiveSettings;
  evaluations = 0;

  constructor(
    readonly predictor: Predictor,
    readonly feedstock: Feedstock,
    settings: Partial<ObjectiveSettings> = {},
    nCompositions = 64,
    nNoise = 16,
    seed = 0,
  ) {
    this.settings = { ...DEFAULT_SETTINGS, ...settings };
    const rng = new Rng(seed);
    this.samples = sampleFeedstock(feedstock, nCompositions, rng);
    this.normals = Array.from({ length: nCompositions }, () => Array.from({ length: nNoise }, () => rng.normal()));
  }

  mixture(paramsBatch: number[][]): MixtureStats[] {
    const K = this.samples.comps.length;
    const comps: number[][] = [];
    const moist: number[] = [];
    const params: number[][] = [];
    for (const x of paramsBatch)
      for (let k = 0; k < K; k++) {
        comps.push(this.samples.comps[k]);
        moist.push(this.samples.moisture[k]);
        params.push(x);
      }
    const pred = this.predictor.predict(comps, moist, params);
    this.evaluations += paramsBatch.length;
    const s = this.settings;
    return paramsBatch.map((_, p) => {
      const values: number[] = [];
      const weights: number[] = [];
      let pSum = 0;
      let mLog = 0;
      let vE = 0;
      let vA = 0;
      for (let k = 0; k < K; k++) {
        const i = p * K + k;
        const sd = Math.sqrt(pred.varEpistemic[i] + pred.varAleatoric[i]);
        const w = Math.max(pred.pSuccess[i], 1e-9);
        pSum += pred.pSuccess[i];
        mLog += pred.mean[i];
        vE += pred.varEpistemic[i];
        vA += pred.varAleatoric[i];
        for (const z of this.normals[k]) {
          values.push(Math.exp(pred.mean[i] + sd * z));
          weights.push(w);
        }
      }
      mLog /= K;
      let vIn = 0;
      for (let k = 0; k < K; k++) vIn += (pred.mean[p * K + k] - mLog) ** 2;
      const wTot = weights.reduce((a, b) => a + b, 0);
      let mean = 0;
      let exceed = 0;
      for (let i = 0; i < values.length; i++) {
        mean += weights[i] * values[i];
        if (values[i] >= s.chanceThresholdMPa) exceed += weights[i];
      }
      return {
        mean: mean / wTot,
        quantile: weightedQuantile(values, weights, s.quantileLevel),
        cvar: weightedCvar(values, weights, s.cvarAlpha),
        pExceed: exceed / wTot,
        pSuccess: pSum / K,
        varInput: vIn / K,
        varEpistemic: vE / K,
        varAleatoric: vA / K,
        meanLog: mLog,
      };
    });
  }

  nominal(paramsBatch: number[][]): NominalStats[] {
    const mu = closure(this.feedstock.mean);
    const pred = this.predictor.predict(
      paramsBatch.map(() => mu),
      paramsBatch.map(() => this.feedstock.moistureMedianPct),
      paramsBatch,
    );
    this.evaluations += paramsBatch.length;
    return paramsBatch.map((_, i) => ({
      nominalMean: Math.exp(pred.mean[i] + 0.5 * (pred.varEpistemic[i] + pred.varAleatoric[i])),
      nominalPSuccess: pred.pSuccess[i],
    }));
  }

  /** Penalised objective value to maximise (computes only what the objective needs). */
  score(objective: Objective, paramsBatch: number[][]): number[] {
    const s = this.settings;
    const pen = s.penaltyMPa;
    if (objective === "nominal") {
      return this.nominal(paramsBatch).map((n) => n.nominalMean - pen * Math.max(0, s.pMin - n.nominalPSuccess));
    }
    return this.mixture(paramsBatch).map((m) => {
      const successPen = pen * Math.max(0, s.pMin - m.pSuccess);
      switch (objective) {
        case "expected":
          return m.mean - successPen;
        case "quantile":
          return m.quantile - successPen;
        case "cvar":
          return m.cvar - successPen;
        default:
          return m.mean - successPen - pen * Math.max(0, s.chanceProb - m.pExceed);
      }
    });
  }
}
