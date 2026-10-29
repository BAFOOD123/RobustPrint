import { describe, expect, it } from "vitest";
import gold from "./golden/gp.json";
import { features, physicsPrior } from "../src/engine/features";
import { GaussianProcess } from "../src/engine/gp";

describe("features and prior parity", () => {
  it("reproduce the Python features and physics prior", () => {
    gold.composition.forEach((c, i) => {
      const f = features(c, gold.params[i], gold.moisture[i]);
      f.forEach((v, j) => expect(Math.abs(v - gold.features[i][j])).toBeLessThan(1e-9));
      expect(Math.abs(physicsPrior(c, gold.params[i]) - gold.prior[i])).toBeLessThan(1e-9);
    });
  });
});

describe("GaussianProcess", () => {
  const y = gold.log_strength.map((v, i) => v - gold.prior[i]);
  it("matches scikit-learn predictions and LML for fixed hyperparameters", () => {
    const gp = new GaussianProcess();
    gp.setHyper({ logAmp: Math.log(gold.amp), logLs: gold.length_scales.map(Math.log), logNoise: Math.log(gold.noise) }, gold.features, y);
    expect(Math.abs(gp.lml - gold.lml)).toBeLessThan(1e-6);
    const p = gp.predict(gold.test_features);
    p.mean.forEach((v, i) => expect(Math.abs(v - gold.pred_mean[i])).toBeLessThan(1e-7));
    p.varEpistemic.forEach((v, i) => expect(Math.abs(v - gold.pred_var_epistemic[i])).toBeLessThan(1e-7));
    p.varAleatoric.forEach((v, i) => expect(Math.abs(v - gold.pred_var_aleatoric[i])).toBeLessThan(1e-9));
  });
  it("has a correct analytic LML gradient", () => {
    const gp = new GaussianProcess();
    gp.setHyper({ logAmp: 0.3, logLs: gold.length_scales.map((l) => Math.log(l) + 0.2), logNoise: -3 }, gold.features, y);
    const h = { logAmp: 0.3, logLs: gp.hyper.logLs.slice(), logNoise: -3 };
    const { grad } = gp.lmlAndGrad(h);
    const flat = [h.logAmp, ...h.logLs, h.logNoise];
    const eps = 1e-5;
    flat.forEach((_, k) => {
      const up = flat.slice(); up[k] += eps;
      const dn = flat.slice(); dn[k] -= eps;
      const toH = (v: number[]) => ({ logAmp: v[0], logLs: v.slice(1, -1), logNoise: v[v.length - 1] });
      const fd = (gp.lmlAndGrad(toH(up), false).lml - gp.lmlAndGrad(toH(dn), false).lml) / (2 * eps);
      expect(Math.abs(fd - grad[k])).toBeLessThan(1e-4 * Math.max(1, Math.abs(fd)));
    });
  });
  it("fits to a marginal likelihood close to scikit-learn's optimum", () => {
    const gp = new GaussianProcess().fit(gold.features, y, { restarts: 2, iterations: 200, seed: 0 });
    expect(gp.lml).toBeGreaterThan(gold.lml - 2.0);
    expect(gp.noiseVariance).toBeGreaterThanOrEqual(0.03 ** 2 * 0.999);
    console.log(`TS LML ${gp.lml.toFixed(3)} vs sklearn ${gold.lml.toFixed(3)}`);
  });
});
