/**
 * Predictors: map (composition, moisture, settings) to a Gaussian on log strength (split
 * into epistemic and aleatoric variance) and a success probability.
 */

import { features, physicsPrior } from "./features";
import { GaussianProcess, type GPHyper } from "./gp";
import { GbmSuccessClassifier } from "./gbm";
import { CONFIG, trueOutcome } from "./simulator";

export interface PredictionBatch {
  mean: Float64Array;
  varEpistemic: Float64Array;
  varAleatoric: Float64Array;
  pSuccess: Float64Array;
}

export interface Predictor {
  readonly kind: "model" | "oracle";
  predict(comps: number[][], moisture: number[], params: number[][]): PredictionBatch;
}

/** The ground-truth simulator as a predictor (no epistemic uncertainty). */
export class OraclePredictor implements Predictor {
  readonly kind = "oracle" as const;

  predict(comps: number[][], moisture: number[], params: number[][]): PredictionBatch {
    const n = comps.length;
    const out: PredictionBatch = { mean: new Float64Array(n), varEpistemic: new Float64Array(n), varAleatoric: new Float64Array(n), pSuccess: new Float64Array(n) };
    for (let i = 0; i < n; i++) {
      const o = trueOutcome(comps[i], params[i], moisture[i]);
      out.mean[i] = Math.log(o.strengthMPa);
      out.varAleatoric[i] = o.noiseCv ** 2 + CONFIG.batchEffectSd ** 2;
      out.pSuccess[i] = o.pSuccess;
    }
    return out;
  }
}

export interface TrainingJob {
  composition: number[];
  params: number[];
  moisture: number;
  success: boolean;
  strengthMPa: number;
  strengthSdMPa: number;
  nReplicates: number;
}

export interface ModelSummary {
  nJobs: number;
  nStrength: number;
  lml: number;
  noiseVariance: number;
  lengthScales: number[];
  specimenLogVar: number;
  successRate: number;
}

/** Physics-residual GP + boosted-tree success model, with the single-part noise correction. */
export class ModelPredictor implements Predictor {
  readonly kind = "model" as const;
  gp = new GaussianProcess();
  clf = new GbmSuccessClassifier();
  specimenLogVar = 0;
  meanReplicates = 1;
  summary: ModelSummary | null = null;

  fit(jobs: TrainingJob[], opts: { seed?: number; restarts?: number; iterations?: number; onProgress?: (f: number) => void } = {}): this {
    const ok = jobs.filter((j) => j.success && Number.isFinite(j.strengthMPa) && j.strengthMPa > 0);
    if (ok.length < 5) throw new Error(`Need at least 5 successful jobs with a strength; got ${ok.length}.`);
    const X = ok.map((j) => features(j.composition, j.params, j.moisture));
    const y = ok.map((j) => Math.log(j.strengthMPa) - physicsPrior(j.composition, j.params));
    this.gp.fit(X, y, { seed: opts.seed, restarts: opts.restarts ?? 2, iterations: opts.iterations ?? 150, onProgress: opts.onProgress });
    this.clf.fit(jobs.map((j) => features(j.composition, j.params, j.moisture)), jobs.map((j) => j.success));
    const withSd = ok.filter((j) => j.nReplicates > 1 && Number.isFinite(j.strengthSdMPa));
    const dof = withSd.reduce((s, j) => s + j.nReplicates - 1, 0);
    this.specimenLogVar = dof > 0 ? withSd.reduce((s, j) => s + (j.nReplicates - 1) * (j.strengthSdMPa / j.strengthMPa) ** 2, 0) / dof : 0;
    this.meanReplicates = ok.reduce((s, j) => s + j.nReplicates, 0) / ok.length;
    this.summary = {
      nJobs: jobs.length,
      nStrength: ok.length,
      lml: this.gp.lml,
      noiseVariance: this.gp.noiseVariance,
      lengthScales: this.gp.lengthScales,
      specimenLogVar: this.specimenLogVar,
      successRate: jobs.filter((j) => j.success).length / jobs.length,
    };
    return this;
  }

  predict(comps: number[][], moisture: number[], params: number[][]): PredictionBatch {
    const X = comps.map((c, i) => features(c, params[i], moisture[i]));
    const g = this.gp.predict(X);
    const extra = this.meanReplicates > 1 ? this.specimenLogVar * (1 - 1 / this.meanReplicates) : 0;
    const n = comps.length;
    const mean = new Float64Array(n);
    const varA = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      mean[i] = g.mean[i] + physicsPrior(comps[i], params[i]);
      varA[i] = g.varAleatoric[i] + extra;
    }
    return { mean, varEpistemic: g.varEpistemic, varAleatoric: varA, pSuccess: this.clf.predictProba(X) };
  }

  /** Serializable state (for transferring a fitted model between worker and page). */
  hyper(): GPHyper {
    return this.gp.hyper;
  }
}
