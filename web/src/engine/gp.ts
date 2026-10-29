/**
 * Gaussian-process regression with an ARD Matérn-5/2 kernel plus white noise
 * (port of robustprint.models.gp.GPRegressor). Inputs and outputs are standardised;
 * hyperparameters maximise the log marginal likelihood (analytic gradient, Adam with
 * restarts, box constraints identical to the Python bounds). The algebra (prediction for
 * fixed hyperparameters) is checked against scikit-learn in tests/gp.test.ts.
 */

import { choleskyJitter, cholInverse, cholSolve, solveLower } from "./linalg";
import { Rng } from "./rng";

const SQRT5 = Math.sqrt(5);
/** Smallest noise SD on the target (log strength): 3 % CV (ADR-014). */
export const MIN_LOG_NOISE_SD = 0.03;
const JITTER = 1e-6;

export interface GPHyper {
  logAmp: number;
  logLs: number[];
  logNoise: number;
}

export interface GPFitOptions {
  restarts?: number;
  iterations?: number;
  seed?: number;
  noiseFloorRaw?: number;
  onProgress?: (fraction: number) => void;
}

export interface GPPrediction {
  mean: Float64Array;
  varEpistemic: Float64Array;
  varAleatoric: Float64Array;
}

function matern52(r: number): number {
  return (1 + SQRT5 * r + (5 / 3) * r * r) * Math.exp(-SQRT5 * r);
}

export class GaussianProcess {
  d = 0;
  n = 0;
  xMean: number[] = [];
  xStd: number[] = [];
  yMean = 0;
  yStd = 1;
  X = new Float64Array(0);
  y = new Float64Array(0);
  hyper: GPHyper = { logAmp: 0, logLs: [], logNoise: Math.log(0.1) };
  L = new Float64Array(0);
  alpha = new Float64Array(0);
  bounds = { amp: [Math.log(1e-2), Math.log(1e2)], ls: [Math.log(5e-2), Math.log(1e3)], noise: [Math.log(1e-5), Math.log(1)] };
  lml = -Infinity;

  /** Standardise inputs (population SD, zero SD -> 1) and outputs. */
  private setData(Xraw: number[][], yraw: number[]): void {
    this.n = Xraw.length;
    this.d = Xraw[0].length;
    this.xMean = Array.from({ length: this.d }, (_, j) => Xraw.reduce((s, r) => s + r[j], 0) / this.n);
    this.xStd = Array.from({ length: this.d }, (_, j) => {
      const v = Xraw.reduce((s, r) => s + (r[j] - this.xMean[j]) ** 2, 0) / this.n;
      return v > 0 ? Math.sqrt(v) : 1;
    });
    this.X = new Float64Array(this.n * this.d);
    Xraw.forEach((r, i) => r.forEach((v, j) => (this.X[i * this.d + j] = (v - this.xMean[j]) / this.xStd[j])));
    this.yMean = yraw.reduce((s, v) => s + v, 0) / this.n;
    const sd = Math.sqrt(yraw.reduce((s, v) => s + (v - this.yMean) ** 2, 0) / this.n);
    this.yStd = sd > 0 ? sd : 1;
    this.y = Float64Array.from(yraw, (v) => (v - this.yMean) / this.yStd);
  }

  private scaled(row: ArrayLike<number>): Float64Array {
    const out = new Float64Array(this.d);
    for (let j = 0; j < this.d; j++) out[j] = (row[j] - this.xMean[j]) / this.xStd[j];
    return out;
  }

  /** Log marginal likelihood and its gradient with respect to [logAmp, logLs..., logNoise]. */
  lmlAndGrad(h: GPHyper, withGrad = true): { lml: number; grad: number[] } {
    const { n, d, X, y } = this;
    const amp = Math.exp(h.logAmp);
    const noise = Math.exp(h.logNoise);
    const invLs2 = h.logLs.map((l) => Math.exp(-2 * l));
    const K = new Float64Array(n * n);
    const G = withGrad ? new Float64Array(n * n) : null;
    for (let a = 0; a < n; a++) {
      K[a * n + a] = amp + noise + JITTER;
      for (let b = 0; b < a; b++) {
        let r2 = 0;
        for (let k = 0; k < d; k++) {
          const diff = X[a * d + k] - X[b * d + k];
          r2 += diff * diff * invLs2[k];
        }
        const r = Math.sqrt(r2);
        const e = Math.exp(-SQRT5 * r);
        const kv = amp * (1 + SQRT5 * r + (5 / 3) * r2) * e;
        K[a * n + b] = kv;
        K[b * n + a] = kv;
        if (G) {
          const g = amp * (5 / 3) * (1 + SQRT5 * r) * e;
          G[a * n + b] = g;
          G[b * n + a] = g;
        }
      }
    }
    const { L } = choleskyJitter(K, n);
    const alpha = cholSolve(L, n, y);
    let logDet = 0;
    for (let i = 0; i < n; i++) logDet += Math.log(L[i * n + i]);
    let fit = 0;
    for (let i = 0; i < n; i++) fit += y[i] * alpha[i];
    const lml = -0.5 * fit - logDet - 0.5 * n * Math.log(2 * Math.PI);
    if (!G) return { lml, grad: [] };
    const Kinv = cholInverse(L, n);
    const grad = new Array<number>(d + 2).fill(0);
    let trW = 0;
    for (let a = 0; a < n; a++) {
      const waa = alpha[a] * alpha[a] - Kinv[a * n + a];
      trW += waa;
      grad[0] += 0.5 * waa * amp;
      for (let b = 0; b < a; b++) {
        const w = alpha[a] * alpha[b] - Kinv[a * n + b];
        grad[0] += w * K[a * n + b];
        const g = w * G[a * n + b];
        for (let k = 0; k < d; k++) {
          const diff = X[a * d + k] - X[b * d + k];
          grad[1 + k] += g * diff * diff * invLs2[k];
        }
      }
    }
    grad[d + 1] = 0.5 * noise * trW;
    return { lml, grad };
  }

  private clampHyper(v: number[]): number[] {
    const { amp, ls, noise } = this.bounds;
    return v.map((x, i) => {
      const [lo, hi] = i === 0 ? amp : i === v.length - 1 ? noise : ls;
      return Math.min(Math.max(x, lo), hi);
    });
  }

  private toHyper(v: number[]): GPHyper {
    return { logAmp: v[0], logLs: v.slice(1, v.length - 1), logNoise: v[v.length - 1] };
  }

  /** Projected Adam ascent on the log marginal likelihood from one start; returns best point. */
  private optimise(start: number[], iterations: number): { v: number[]; lml: number } {
    let v = this.clampHyper(start);
    const m = new Array<number>(v.length).fill(0);
    const s = new Array<number>(v.length).fill(0);
    let best = { v: v.slice(), lml: -Infinity };
    const b1 = 0.9;
    const b2 = 0.999;
    for (let t = 1; t <= iterations; t++) {
      const { lml, grad } = this.lmlAndGrad(this.toHyper(v));
      if (lml > best.lml) best = { v: v.slice(), lml };
      const lr = 0.08 * (0.05 + 0.95 * (1 - t / iterations));
      for (let i = 0; i < v.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * grad[i];
        s[i] = b2 * s[i] + (1 - b2) * grad[i] * grad[i];
        const mh = m[i] / (1 - b1 ** t);
        const sh = s[i] / (1 - b2 ** t);
        v[i] += (lr * mh) / (Math.sqrt(sh) + 1e-8);
      }
      v = this.clampHyper(v);
    }
    const final = this.lmlAndGrad(this.toHyper(v), false).lml;
    if (final > best.lml) best = { v: v.slice(), lml: final };
    return best;
  }

  /** Fit hyperparameters by maximum marginal likelihood (raw-scale targets). */
  fit(Xraw: number[][], yraw: number[], opts: GPFitOptions = {}): this {
    this.setData(Xraw, yraw);
    const floorRaw = opts.noiseFloorRaw ?? MIN_LOG_NOISE_SD ** 2;
    const floor = Math.min(floorRaw / this.yStd ** 2, 0.5);
    this.bounds.noise = [Math.log(floor), Math.log(1)];
    const restarts = opts.restarts ?? 2;
    const iterations = opts.iterations ?? 150;
    const rng = new Rng(opts.seed ?? 0);
    const starts: number[][] = [[0, ...new Array<number>(this.d).fill(0), Math.log(Math.max(0.1, 2 * floor))]];
    for (let r = 0; r < restarts; r++) {
      starts.push([
        Math.log(0.3) + rng.random() * Math.log(10),
        ...Array.from({ length: this.d }, () => Math.log(0.3) + rng.random() * Math.log(100)),
        Math.log(floor) + rng.random() * (Math.log(1) - Math.log(floor)),
      ]);
    }
    let best = { v: starts[0], lml: -Infinity };
    starts.forEach((st, i) => {
      const res = this.optimise(st, iterations);
      if (res.lml > best.lml) best = res;
      opts.onProgress?.((i + 1) / starts.length);
    });
    return this.setHyper(this.toHyper(best.v));
  }

  /** Condition on the stored data with fixed hyperparameters (standardised units). */
  setHyper(h: GPHyper, Xraw?: number[][], yraw?: number[]): this {
    if (Xraw && yraw) this.setData(Xraw, yraw);
    this.hyper = h;
    const { n, d, X } = this;
    const amp = Math.exp(h.logAmp);
    const noise = Math.exp(h.logNoise);
    const K = new Float64Array(n * n);
    for (let a = 0; a < n; a++)
      for (let b = 0; b <= a; b++) {
        const kv = a === b ? amp + noise + JITTER : this.kernelScaled(X.subarray(a * d, a * d + d), X.subarray(b * d, b * d + d));
        K[a * n + b] = kv;
        K[b * n + a] = kv;
      }
    this.L = choleskyJitter(K, n).L;
    this.alpha = cholSolve(this.L, n, this.y);
    this.lml = this.lmlAndGrad(h, false).lml;
    return this;
  }

  /** Latent kernel between two standardised rows. */
  kernelScaled(a: ArrayLike<number>, b: ArrayLike<number>): number {
    let r2 = 0;
    for (let k = 0; k < this.d; k++) {
      const diff = (a[k] - b[k]) / Math.exp(this.hyper.logLs[k]);
      r2 += diff * diff;
    }
    return Math.exp(this.hyper.logAmp) * matern52(Math.sqrt(r2));
  }

  /** Cross-covariance vector k(x, X_train) for a standardised row. */
  private crossScaled(xs: Float64Array): Float64Array {
    const { n, d, X } = this;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.kernelScaled(xs, X.subarray(i * d, i * d + d));
    return out;
  }

  /** Learned white-noise variance on the original target scale. */
  get noiseVariance(): number {
    return Math.exp(this.hyper.logNoise) * this.yStd ** 2;
  }

  get lengthScales(): number[] {
    return this.hyper.logLs.map(Math.exp);
  }

  /** Predictive mean, latent (epistemic) variance and noise (aleatoric) variance, raw scale. */
  predict(rows: ArrayLike<number>[]): GPPrediction {
    const m = rows.length;
    const mean = new Float64Array(m);
    const varE = new Float64Array(m);
    const varA = new Float64Array(m).fill(this.noiseVariance);
    const amp = Math.exp(this.hyper.logAmp);
    for (let i = 0; i < m; i++) {
      const ks = this.crossScaled(this.scaled(rows[i]));
      let mu = 0;
      for (let j = 0; j < this.n; j++) mu += ks[j] * this.alpha[j];
      const v = solveLower(this.L, this.n, ks);
      let vv = 0;
      for (let j = 0; j < this.n; j++) vv += v[j] * v[j];
      mean[i] = mu * this.yStd + this.yMean;
      varE[i] = Math.max(amp - vv, 1e-12) * this.yStd ** 2;
    }
    return { mean, varEpistemic: varE, varAleatoric: varA };
  }

  /** Standardised rows (for posterior covariance computations). */
  scaleRows(rows: ArrayLike<number>[]): Float64Array[] {
    return rows.map((r) => this.scaled(r));
  }

  /** Posterior latent covariance between two sets of raw rows (standardised y units). */
  posteriorCov(A: ArrayLike<number>[], B: ArrayLike<number>[]): number[][] {
    const sa = this.scaleRows(A);
    const sb = this.scaleRows(B);
    const va = sa.map((x) => solveLower(this.L, this.n, this.crossScaled(x)));
    const vb = sb.map((x) => solveLower(this.L, this.n, this.crossScaled(x)));
    return sa.map((xa, i) =>
      sb.map((xb, j) => {
        let s = 0;
        for (let k = 0; k < this.n; k++) s += va[i][k] * vb[j][k];
        return this.kernelScaled(xa, xb) - s;
      }),
    );
  }
}
