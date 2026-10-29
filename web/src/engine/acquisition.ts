/**
 * Next-experiment selection with the GP posterior covariance (port of
 * robustprint.active.acquisition): greedy batch selection with exact fantasy updates.
 */

import { features } from "./features";
import type { Feedstock } from "./materials";
import { latinHypercube, safeBounds } from "./optimise";
import { sampleFeedstock } from "./objectives";
import type { ModelPredictor } from "./predictor";
import type { Rng } from "./rng";
import { printTimeProxy } from "./simulator";
import { fromUnit } from "./space";

export const STRATEGIES = ["targeted_ivr", "cost_aware_ivr", "max_variance", "space_filling", "random"] as const;
export type Strategy = (typeof STRATEGIES)[number];

export const STRATEGY_LABELS: Record<Strategy, string> = {
  targeted_ivr: "Targeted variance reduction",
  cost_aware_ivr: "Cost-aware variance reduction",
  max_variance: "Maximum uncertainty",
  space_filling: "Space-filling (Latin hypercube)",
  random: "Random",
};

export function experimentCost(params: number[], setup = 1): number {
  return setup + printTimeProxy(params[2], params[3], params[4]);
}

export interface Candidate {
  composition: number[];
  moisture: number;
  params: number[];
  score: number;
  cost: number;
}

/** Choose n experiments (composition + settings) by the given strategy. */
export function selectExperiments(
  predictor: ModelPredictor | null,
  feedstock: Feedstock,
  n: number,
  strategy: Strategy,
  rng: Rng,
  recommended: number[] | null,
  nCandidates = 256,
  nReference = 24,
  nReplicates = 3,
): Candidate[] {
  const fs = sampleFeedstock(feedstock, nCandidates, rng);
  const { low, high } = safeBounds(feedstock);
  const unit = strategy === "random" ? Array.from({ length: nCandidates }, () => low.map(() => rng.random())) : latinHypercube(nCandidates, low.length, rng);
  const params = unit.map((u) => fromUnit(u, low, high));
  const cands = params.map((p, i) => ({ composition: fs.comps[i], moisture: fs.moisture[i], params: p, score: NaN, cost: experimentCost(p) }));
  if (strategy === "random" || strategy === "space_filling" || !predictor) {
    if (strategy === "space_filling") {
      const lhs = latinHypercube(n, low.length, rng).map((u) => fromUnit(u, low, high));
      return lhs.map((p, i) => ({ ...cands[i], params: p, cost: experimentCost(p) }));
    }
    return cands.slice(0, n);
  }
  const gp = predictor.gp;
  const X = cands.map((c) => features(c.composition, c.params, c.moisture));
  const pSucc = predictor.clf.predictProba(X);
  const noise = Math.exp(gp.hyper.logNoise) / nReplicates;
  const cc = gp.posteriorCov(X, X);
  let cross: number[][] = [];
  const useRef = strategy !== "max_variance";
  if (useRef) {
    if (!recommended) throw new Error("Variance-reduction strategies need a recommended setting.");
    const ref = sampleFeedstock(feedstock, nReference, rng);
    const R = ref.comps.map((c, i) => features(c, recommended, ref.moisture[i]));
    cross = gp.posteriorCov(R, X);
  }
  let varC = cc.map((r, i) => Math.max(r[i], 1e-12));
  const available = cands.map(() => true);
  const chosen: Candidate[] = [];
  for (let step = 0; step < Math.min(n, cands.length); step++) {
    let best = -1;
    let bestVal = -Infinity;
    for (let j = 0; j < cands.length; j++) {
      if (!available[j]) continue;
      let gain = varC[j];
      if (useRef) {
        let s = 0;
        for (const row of cross) s += row[j] * row[j];
        gain = s / (varC[j] + noise);
      }
      let value = gain * pSucc[j];
      if (strategy === "cost_aware_ivr") value /= cands[j].cost;
      if (value > bestVal) {
        bestVal = value;
        best = j;
      }
    }
    available[best] = false;
    chosen.push({ ...cands[best], score: bestVal });
    const denom = varC[best] + noise;
    const kj = cc.map((r) => r[best]);
    if (useRef) cross = cross.map((row) => row.map((v, j) => v - (row[best] * kj[j]) / denom));
    for (let a = 0; a < cc.length; a++) for (let b = 0; b < cc.length; b++) cc[a][b] -= (kj[a] * kj[b]) / denom;
    varC = cc.map((r, i) => Math.max(r[i], 1e-12));
  }
  return chosen;
}
