/** Compositional-data utilities (port of robustprint.physics.composition). */

import { digamma, trigamma } from "./math";

export const ZERO_REPLACEMENT = 1e-4;

export function closure(x: ArrayLike<number>): number[] {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  if (!(s > 0)) throw new Error("Every composition must have a positive total.");
  return Array.from(x, (v) => v / s);
}

/** Multiplicative zero replacement: parts below delta become delta, the rest shrink. */
export function replaceZeros(x: ArrayLike<number>, delta = ZERO_REPLACEMENT): number[] {
  const a = closure(x);
  const small = a.map((v) => v < delta);
  const nSmall = small.filter(Boolean).length;
  const restTotal = a.reduce((s, v, i) => s + (small[i] ? 0 : v), 0);
  return a.map((v, i) => (small[i] ? delta : (v * (1 - nSmall * delta)) / restTotal));
}

/** (D, D-1) Helmert basis of the clr hyperplane, column-major as basis[i][j]. */
export function helmertBasis(nParts: number): number[][] {
  const basis = Array.from({ length: nParts }, () => new Array<number>(nParts - 1).fill(0));
  for (let j = 1; j < nParts; j++) {
    const scale = Math.sqrt(j / (j + 1));
    for (let i = 0; i < j; i++) basis[i][j - 1] = scale / j;
    basis[j][j - 1] = -scale;
  }
  return basis;
}

const basisCache = new Map<number, number[][]>();
function basisFor(n: number): number[][] {
  let b = basisCache.get(n);
  if (!b) {
    b = helmertBasis(n);
    basisCache.set(n, b);
  }
  return b;
}

export function clr(x: ArrayLike<number>): number[] {
  const logx = replaceZeros(x).map(Math.log);
  const m = logx.reduce((s, v) => s + v, 0) / logx.length;
  return logx.map((v) => v - m);
}

export function ilr(x: ArrayLike<number>): number[] {
  const c = clr(x);
  const b = basisFor(c.length);
  const out = new Array<number>(c.length - 1).fill(0);
  for (let j = 0; j < c.length - 1; j++) for (let i = 0; i < c.length; i++) out[j] += c[i] * b[i][j];
  return out;
}

export function ilrInverse(z: ArrayLike<number>): number[] {
  const d = z.length + 1;
  const b = basisFor(d);
  const c = new Array<number>(d).fill(0);
  for (let i = 0; i < d; i++) for (let j = 0; j < d - 1; j++) c[i] += b[i][j] * z[j];
  const mx = Math.max(...c);
  return closure(c.map((v) => Math.exp(v - mx)));
}

/** Exact mean and covariance of ilr(c) for c ~ Dirichlet(alpha). */
export function dirichletIlrMoments(alpha: number[]): { mean: number[]; cov: number[][] } {
  const d = alpha.length;
  const a0 = alpha.reduce((s, v) => s + v, 0);
  const meanLog = alpha.map((a) => digamma(a) - digamma(a0));
  const covLog = alpha.map((ai, i) => alpha.map((_, j) => (i === j ? trigamma(ai) : 0) - trigamma(a0)));
  const b = basisFor(d);
  const mean = new Array<number>(d - 1).fill(0);
  for (let j = 0; j < d - 1; j++) for (let i = 0; i < d; i++) mean[j] += meanLog[i] * b[i][j];
  const cov = Array.from({ length: d - 1 }, () => new Array<number>(d - 1).fill(0));
  for (let p = 0; p < d - 1; p++)
    for (let q = 0; q < d - 1; q++) {
      let s = 0;
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) s += b[i][p] * covLog[i][j] * b[j][q];
      cov[p][q] = s;
    }
  return { mean, cov };
}

/** Marginal standard deviation of each part under Dirichlet(concentration * mean). */
export function dirichletStd(meanComp: number[], concentration: number): number[] {
  const mu = closure(meanComp);
  return mu.map((m) => Math.sqrt((m * (1 - m)) / (concentration + 1)));
}
