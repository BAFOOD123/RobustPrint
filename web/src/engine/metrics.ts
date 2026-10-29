/** Accuracy, scoring-rule and calibration metrics (port of robustprint.uncertainty.metrics). */

import { normCdf } from "./math";

export interface RegressionReport {
  rmseLog: number;
  rmseMPa: number;
  maeMPa: number;
  nll: number;
  crps: number;
  qce: number;
  coverage90: number;
  width90: number;
}

const Z90 = 1.6448536269514722;

export function regressionReport(y: number[], mean: number[], variance: number[]): RegressionReport {
  const n = y.length;
  let se = 0, seM = 0, aeM = 0, nll = 0, crps = 0, cover = 0, width = 0;
  const pit: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = Math.max(variance[i], 1e-12);
    const sd = Math.sqrt(v);
    const r = y[i] - mean[i];
    se += r * r;
    const yM = Math.exp(y[i]);
    seM += (yM - Math.exp(mean[i] + 0.5 * v)) ** 2;
    aeM += Math.abs(yM - Math.exp(mean[i]));
    nll += 0.5 * Math.log(2 * Math.PI * v) + (0.5 * r * r) / v;
    const z = r / sd;
    const pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
    const cdf = normCdf(z);
    crps += sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / Math.sqrt(Math.PI));
    pit.push(cdf);
    if (Math.abs(r) <= Z90 * sd) cover += 1;
    width += 2 * Z90 * sd;
  }
  let qce = 0;
  const levels = Array.from({ length: 19 }, (_, k) => 0.05 + 0.05 * k);
  for (const p of levels) qce += Math.abs(pit.filter((u) => u <= p).length / n - p);
  return {
    rmseLog: Math.sqrt(se / n),
    rmseMPa: Math.sqrt(seM / n),
    maeMPa: aeM / n,
    nll: nll / n,
    crps: crps / n,
    qce: qce / levels.length,
    coverage90: cover / n,
    width90: width / n,
  };
}

/** Observed frequency of y <= predicted p-quantile, for each level p. */
export function calibrationCurve(y: number[], mean: number[], variance: number[]): { level: number; observed: number }[] {
  const pit = y.map((v, i) => normCdf((v - mean[i]) / Math.sqrt(Math.max(variance[i], 1e-12))));
  return Array.from({ length: 19 }, (_, k) => {
    const level = 0.05 + 0.05 * k;
    return { level, observed: pit.filter((u) => u <= level).length / pit.length };
  });
}

export function brierScore(y: boolean[], p: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += (p[i] - (y[i] ? 1 : 0)) ** 2;
  return s / y.length;
}

export function reliabilityCurve(y: boolean[], p: ArrayLike<number>, nBins = 10): { meanP: number; freq: number; count: number }[] {
  const bins = Array.from({ length: nBins }, () => ({ sp: 0, sy: 0, count: 0 }));
  for (let i = 0; i < y.length; i++) {
    const b = Math.min(Math.floor(p[i] * nBins), nBins - 1);
    bins[b].sp += p[i];
    bins[b].sy += y[i] ? 1 : 0;
    bins[b].count += 1;
  }
  return bins.filter((b) => b.count > 0).map((b) => ({ meanP: b.sp / b.count, freq: b.sy / b.count, count: b.count }));
}
