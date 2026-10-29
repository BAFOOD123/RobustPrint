/**
 * Maximisation over a box: Latin-hypercube warm start followed by CMA-ES (Hansen 2016,
 * default strategy parameters). Bounds are handled by evaluating at the projection onto
 * the box plus a quadratic penalty on the distance outside it.
 */

import { COMPONENTS, type Feedstock } from "./materials";
import { closure } from "./composition";
import { symmetricEigen } from "./linalg";
import type { Objective, RobustProblem } from "./objectives";
import { Rng } from "./rng";
import { HIGH, LOW } from "./space";

/** Margin below the dominant polymer's degradation onset (ADR-016). */
export const DEGRADATION_MARGIN_C = 10;

/** Search bounds with the nozzle window limited by the dominant polymer. */
export function safeBounds(feedstock: Feedstock): { low: number[]; high: number[]; dominant: string } {
  const mu = closure(feedstock.mean);
  let dominant = -1;
  COMPONENTS.forEach((c, i) => {
    if (c.kind === "polymer" && (dominant < 0 || mu[i] > mu[dominant])) dominant = i;
  });
  const comp = COMPONENTS[dominant];
  const low = LOW.slice();
  const high = HIGH.slice();
  low[0] = Math.max(low[0], comp.tMin);
  high[0] = Math.min(high[0], comp.tDeg - DEGRADATION_MARGIN_C);
  return { low, high, dominant: comp.name };
}

export function latinHypercube(n: number, d: number, rng: Rng): number[][] {
  const cols = Array.from({ length: d }, () => rng.shuffle(Array.from({ length: n }, (_, i) => i)));
  return Array.from({ length: n }, (_, i) => cols.map((col) => (col[i] + rng.random()) / n));
}

export interface OptimisationResult {
  params: number[];
  score: number;
  evaluations: number;
  trace: number[];
}

export interface MaximiseOptions {
  budget?: number;
  popsize?: number;
  nInit?: number;
  seed?: number;
  sigma0?: number;
  onProgress?: (fraction: number, best: number) => void;
}

/** Maximise a batched score (rows in physical units) over [low, high]. */
export function maximise(score: (xs: number[][]) => number[], low: number[], high: number[], opts: MaximiseOptions = {}): OptimisationResult {
  const d = low.length;
  const budget = opts.budget ?? 800;
  const lambda = opts.popsize ?? 16;
  const nInit = opts.nInit ?? 64;
  const rng = new Rng(opts.seed ?? 0);
  const toX = (u: number[]) => u.map((v, i) => low[i] + Math.min(Math.max(v, 0), 1) * (high[i] - low[i]));
  const trace: number[] = [];

  const u0 = latinHypercube(nInit, d, rng);
  const s0 = score(u0.map(toX));
  let bestI = 0;
  s0.forEach((v, i) => {
    if (v > s0[bestI]) bestI = i;
  });
  let bestU = u0[bestI].slice();
  let bestS = s0[bestI];
  let evals = nInit;
  trace.push(bestS);

  // CMA-ES strategy parameters.
  const mu = Math.floor(lambda / 2);
  const rawW = Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1));
  const sumW = rawW.reduce((a, b) => a + b, 0);
  const w = rawW.map((v) => v / sumW);
  const mueff = 1 / w.reduce((a, b) => a + b * b, 0);
  const cc = (4 + mueff / d) / (d + 4 + (2 * mueff) / d);
  const cs = (mueff + 2) / (d + mueff + 5);
  const c1 = 2 / ((d + 1.3) ** 2 + mueff);
  const cmu = Math.min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((d + 2) ** 2 + mueff));
  const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (d + 1)) - 1) + cs;
  const chiN = Math.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d));

  let mean = bestU.slice();
  let sigma = opts.sigma0 ?? 0.2;
  const pc = new Array<number>(d).fill(0);
  const ps = new Array<number>(d).fill(0);
  let C: number[][] = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)));
  let B = C.map((r) => r.slice());
  let D = new Array<number>(d).fill(1);
  let gen = 0;

  while (evals + lambda <= budget) {
    gen += 1;
    const zs: number[][] = [];
    const ys: number[][] = [];
    const xs: number[][] = [];
    for (let k = 0; k < lambda; k++) {
      const z = Array.from({ length: d }, () => rng.normal());
      const y = B.map((row) => row.reduce((s, b, j) => s + b * D[j] * z[j], 0));
      zs.push(z);
      ys.push(y);
      xs.push(mean.map((m, i) => m + sigma * y[i]));
    }
    const clipped = xs.map((x) => x.map((v) => Math.min(Math.max(v, 0), 1)));
    const raw = score(clipped.map(toX));
    evals += lambda;
    const fit = raw.map((v, k) => v - 1e3 * xs[k].reduce((s, x, i) => s + (x - clipped[k][i]) ** 2, 0));
    raw.forEach((v, k) => {
      if (v > bestS) {
        bestS = v;
        bestU = clipped[k].slice();
      }
    });
    trace.push(bestS);
    opts.onProgress?.(Math.min(evals / budget, 1), bestS);

    const order = fit.map((_, k) => k).sort((a, b) => fit[b] - fit[a]);
    const oldMean = mean;
    mean = Array.from({ length: d }, (_, i) => order.slice(0, mu).reduce((s, k, r) => s + w[r] * xs[k][i], 0));
    const yw = mean.map((m, i) => (m - oldMean[i]) / sigma);
    // C^{-1/2} yw = B D^{-1} B^T yw
    const bty = B[0].map((_, j) => B.reduce((s, row, i) => s + row[j] * yw[i], 0));
    const invSqrt = B.map((row) => row.reduce((s, b, j) => s + (b * bty[j]) / D[j], 0));
    for (let i = 0; i < d; i++) ps[i] = (1 - cs) * ps[i] + Math.sqrt(cs * (2 - cs) * mueff) * invSqrt[i];
    const psNorm = Math.sqrt(ps.reduce((s, v) => s + v * v, 0));
    const hsig = psNorm / Math.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (d + 1) ? 1 : 0;
    for (let i = 0; i < d; i++) pc[i] = (1 - cc) * pc[i] + hsig * Math.sqrt(cc * (2 - cc) * mueff) * yw[i];
    const sel = order.slice(0, mu).map((k) => ys[k]);
    C = C.map((row, i) =>
      row.map((cij, j) => {
        let rankMu = 0;
        for (let r = 0; r < mu; r++) rankMu += w[r] * sel[r][i] * sel[r][j];
        return (1 - c1 - cmu) * cij + c1 * (pc[i] * pc[j] + (1 - hsig) * cc * (2 - cc) * cij) + cmu * rankMu;
      }),
    );
    sigma *= Math.exp((cs / damps) * (psNorm / chiN - 1));
    sigma = Math.min(Math.max(sigma, 1e-8), 1);
    const eig = symmetricEigen(C);
    B = eig.vectors;
    D = eig.values.map((v) => Math.sqrt(Math.max(v, 1e-20)));
    if (sigma * Math.max(...D) < 1e-6) break;
  }
  return { params: toX(bestU), score: bestS, evaluations: evals, trace };
}

/** Recommend settings maximising an objective for the problem's feedstock. */
export function recommend(problem: RobustProblem, objective: Objective, opts: MaximiseOptions = {}): OptimisationResult {
  const { low, high } = safeBounds(problem.feedstock);
  return maximise((xs) => problem.score(objective, xs), low, high, opts);
}
