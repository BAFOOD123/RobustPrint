/** Deployment of fixed settings on fresh, unseen batches using the ground truth. */

import type { Feedstock } from "./materials";
import { quantile } from "./math";
import { Rng } from "./rng";
import { type Batch, realise, sampleBatches, trueOutcome } from "./simulator";

export interface DeploymentResult {
  meanStrength: number;
  q05Strength: number;
  cvar10Strength: number;
  failureRate: number;
  expectedUtility: number;
  pMeetThreshold: number;
  /** Realised strength of every part (NaN for failed prints), in batch order. */
  strengths: number[];
}

export interface Deployment {
  batches: Batch[];
  seed: number;
}

export function drawDeployment(feedstock: Feedstock, n: number, seed: number): Deployment {
  return { batches: sampleBatches(feedstock, n, new Rng(seed), 300, "D"), seed };
}

/** Print one part per fresh batch. The same random numbers are reused for every setting (paired). */
export function deploy(params: number[], dep: Deployment, thresholdMPa = 5): DeploymentResult {
  const rng = new Rng(dep.seed + 1);
  const strengths = dep.batches.map((b) => {
    const t = trueOutcome(b.trueComposition, params, b.moisture, b.batchEffect);
    return realise(b, params, rng, t).strengthMPa;
  });
  const ok = strengths.filter((s) => Number.isFinite(s));
  const sorted = ok.slice().sort((a, b) => a - b);
  const tail = sorted.slice(0, Math.max(1, Math.ceil(0.1 * sorted.length)));
  const utility = strengths.map((s) => (Number.isFinite(s) ? s : 0));
  return {
    meanStrength: ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : 0,
    q05Strength: ok.length ? quantile(ok, 0.05) : 0,
    cvar10Strength: ok.length ? tail.reduce((a, b) => a + b, 0) / tail.length : 0,
    failureRate: 1 - ok.length / strengths.length,
    expectedUtility: utility.reduce((a, b) => a + b, 0) / utility.length,
    pMeetThreshold: utility.filter((u) => u >= thresholdMPa).length / utility.length,
    strengths,
  };
}
