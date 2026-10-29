/** Scalar numerical helpers shared by the engine. */

export function sigmoid(x: number): number {
  return 0.5 * (1 + Math.tanh(0.5 * x));
}

/** Numerically stable log(1 + exp(x)). */
export function softplus(x: number): number {
  return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
}

/** Smooth approximation of min(r, 1) for r >= 0: r / (1 + r^p)^(1/p). */
export function smoothMinOne(r: number, power: number): number {
  const v = Math.max(r, 0);
  return v / Math.pow(1 + Math.pow(v, power), 1 / power);
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}

/** Digamma function (asymptotic series after recurrence to x >= 6). */
export function digamma(x: number): number {
  let result = 0;
  let v = x;
  while (v < 6) {
    result -= 1 / v;
    v += 1;
  }
  const f = 1 / (v * v);
  return (
    result +
    Math.log(v) -
    0.5 / v -
    f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f * (1 / 132)))))
  );
}

/** Trigamma function (asymptotic series after recurrence to x >= 6). */
export function trigamma(x: number): number {
  let result = 0;
  let v = x;
  while (v < 6) {
    result += 1 / (v * v);
    v += 1;
  }
  const f = 1 / (v * v);
  return result + 1 / v + f / 2 + (f / v) * (1 / 6 - f * (1 / 30 - f * (1 / 42 - f / 30)));
}

export function mean(xs: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

export function variance(xs: ArrayLike<number>, ddof = 0): number {
  const m = mean(xs);
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += (xs[i] - m) ** 2;
  return s / (xs.length - ddof);
}

/** Linear-interpolated quantile of an unsorted sample (numpy's default "linear" method). */
export function quantile(xs: ArrayLike<number>, q: number): number {
  const a = Array.from(xs).sort((u, v) => u - v);
  if (a.length === 0) return NaN;
  const pos = (a.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

/** Standard normal CDF (Abramowitz-Stegun 7.1.26 via erf, |error| < 1.5e-7). */
export function normCdf(z: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
  const y =
    1 -
    t *
      (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) *
      Math.exp(-(z * z) / 2);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

/** Inverse standard normal CDF (Acklam's rational approximation, relative error < 1.2e-9). */
export function normPpf(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
