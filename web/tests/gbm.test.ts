import { describe, expect, it } from "vitest";
import gold from "./golden/gbm.json";
import { Booster, GbmSuccessClassifier, Isotonic } from "../src/engine/gbm";

const excess = (p: ArrayLike<number>) => gold.test_p_true.reduce((s, t, i) => s + (p[i] - t) ** 2, 0) / gold.test_p_true.length;

describe("boosted-tree success classifier", () => {
  it("isotonic regression matches scikit-learn", () => {
    const iso = new Isotonic().fit(gold.iso_x, gold.iso_y);
    gold.iso_query.forEach((q, i) => expect(iso.predict(q)).toBeCloseTo(gold.iso_pred[i], 10));
  });
  it("raw boosted scores match HistGradientBoostingClassifier on the same data", () => {
    const y = gold.success.map((s) => (s ? 1 : 0));
    const b = new Booster().fit(gold.features, y);
    const d = gold.test_features.map((x) => b.decision(x));
    const diff = d.map((v, i) => Math.abs(v - gold.raw_decision[i]));
    // Exact thresholds equal scikit-learn's bins below 256 distinct values; residual gaps come from
    // float32 histograms and tie-breaking between equal gains.
    expect(diff.reduce((a, v) => a + v, 0) / diff.length).toBeLessThan(0.05);
  });
  it("calibrated probabilities are as close to the true P(success) as scikit-learn's", () => {
    const p = new GbmSuccessClassifier().fit(gold.features, gold.success).predictProba(gold.test_features);
    const ours = excess(p);
    const theirs = excess(gold.proba);
    expect(ours).toBeLessThan(theirs * 1.25 + 0.005);
    const mad = gold.proba.reduce((s, v, i) => s + Math.abs(v - p[i]), 0) / p.length;
    expect(mad).toBeLessThan(0.12);
  });
  it("falls back to a smoothed constant with one class", () => {
    expect(new GbmSuccessClassifier().fit([[0], [1], [2]], [false, false, false]).predictProba([[1]])[0]).toBeCloseTo(1 / 5);
  });
});
