import { describe, expect, it } from "vitest";
import { mean, variance } from "../src/engine/math";
import { deriveSeed, Rng } from "../src/engine/rng";
import { ISS_NOMINAL } from "../src/engine/materials";
import { sampleBatches } from "../src/engine/simulator";

describe("Rng", () => {
  it("is deterministic per seed and differs across seeds", () => {
    const a = new Rng(1), b = new Rng(1), c = new Rng(2);
    const xa = Array.from({ length: 5 }, () => a.random());
    expect(Array.from({ length: 5 }, () => b.random())).toEqual(xa);
    expect(Array.from({ length: 5 }, () => c.random())).not.toEqual(xa);
    expect(deriveSeed(7, 1, 2)).toBe(deriveSeed(7, 1, 2));
    expect(deriveSeed(7, 1, 2)).not.toBe(deriveSeed(7, 2, 1));
  });
  it("uniform, normal and gamma moments", () => {
    const r = new Rng(3);
    const u = Array.from({ length: 100000 }, () => r.random());
    expect(Math.abs(mean(u) - 0.5)).toBeLessThan(0.005);
    expect(Math.min(...u)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...u)).toBeLessThan(1);
    const z = Array.from({ length: 100000 }, () => r.normal());
    expect(Math.abs(mean(z))).toBeLessThan(0.01);
    expect(Math.abs(variance(z) - 1)).toBeLessThan(0.02);
    for (const shape of [0.3, 1, 4.5]) {
      const g = Array.from({ length: 50000 }, () => r.gamma(shape));
      expect(Math.abs(mean(g) - shape) / shape).toBeLessThan(0.03);
      expect(Math.abs(variance(g) - shape) / shape).toBeLessThan(0.06);
    }
  });
  it("Dirichlet batches reproduce the feedstock mean and spread", () => {
    const b = sampleBatches(ISS_NOMINAL, 20000, new Rng(4));
    ISS_NOMINAL.mean.forEach((m, j) => {
      const col = b.map((x) => x.trueComposition[j]);
      expect(Math.abs(mean(col) - m)).toBeLessThan(0.003);
      const sd = Math.sqrt((m * (1 - m)) / (ISS_NOMINAL.concentration + 1));
      expect(Math.abs(Math.sqrt(variance(col)) - sd) / sd).toBeLessThan(0.05);
    });
    b.slice(0, 100).forEach((x) => expect(Math.abs(x.reportedComposition.reduce((s, v) => s + v, 0) - 1)).toBeLessThan(1e-12));
  });
});
