/**
 * Seeded pseudo-random numbers for reproducible simulations in the browser.
 *
 * Uses sfc32 (a small, fast, well-tested 128-bit-state generator) seeded through
 * splitmix32. Streams are deterministic given the seed but are NOT the same numbers as
 * NumPy's PCG64, so browser simulations reproduce each other, not the Python results.
 */
export class Rng {
  private a: number;
  private b: number;
  private c: number;
  private d: number;
  private spare: number | null = null;

  constructor(seed: number) {
    let s = seed >>> 0;
    const next = (): number => {
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.a = next();
    this.b = next();
    this.c = next();
    this.d = next();
    for (let i = 0; i < 12; i++) this.uint32();
  }

  uint32(): number {
    const t = (((this.a + this.b) >>> 0) + this.d) >>> 0;
    this.d = (this.d + 1) >>> 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) >>> 0;
    this.c = ((this.c << 21) | (this.c >>> 11)) >>> 0;
    this.c = (this.c + t) >>> 0;
    return t;
  }

  /** Uniform on [0, 1) with 53-bit resolution. */
  random(): number {
    const hi = this.uint32() >>> 5;
    const lo = this.uint32() >>> 6;
    return (hi * 67108864 + lo) / 9007199254740992;
  }

  /** Standard normal (Marsaglia polar method). */
  normal(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = 2 * this.random() - 1;
      v = 2 * this.random() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * f;
    return u * f;
  }

  /** Gamma(shape, 1) (Marsaglia & Tsang 2000; boosted for shape < 1). */
  gamma(shape: number): number {
    if (shape < 1) {
      const u = this.random();
      return this.gamma(shape + 1) * Math.pow(Math.max(u, 1e-300), 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x = 0;
      let v = 0;
      do {
        x = this.normal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = this.random();
      if (u < 1 - 0.0331 * x ** 4) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  /** Dirichlet(alpha) draw. */
  dirichlet(alpha: ArrayLike<number>): number[] {
    const g = Array.from(alpha, (a) => this.gamma(Math.max(a, 1e-3)));
    const s = g.reduce((p, q) => p + q, 0);
    return g.map((v) => v / s);
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.random() * n);
  }

  /** In-place Fisher-Yates shuffle. */
  shuffle<T>(xs: T[]): T[] {
    for (let i = xs.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      [xs[i], xs[j]] = [xs[j], xs[i]];
    }
    return xs;
  }
}

/** Derive a child seed from a parent seed and integer keys (deterministic). */
export function deriveSeed(seed: number, ...keys: number[]): number {
  let h = (seed ^ 0x2545f491) >>> 0;
  for (const k of keys) {
    h = Math.imul(h ^ (k >>> 0), 0x9e3779b1) >>> 0;
    h = (h ^ (h >>> 15)) >>> 0;
    h = Math.imul(h, 0x85ebca77) >>> 0;
    h = (h ^ (h >>> 13)) >>> 0;
  }
  return h >>> 0;
}
