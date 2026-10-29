import { describe, expect, it } from "vitest";
import { selectExperiments } from "../src/engine/acquisition";
import { estimateFeedstock, generateCampaign, parseCsv, toCsv } from "../src/engine/campaign";
import { deploy, drawDeployment } from "../src/engine/deploy";
import { ISS_NOMINAL } from "../src/engine/materials";
import { regressionReport } from "../src/engine/metrics";
import { RobustProblem, weightedCvar, weightedQuantile } from "../src/engine/objectives";
import { maximise, recommend, safeBounds } from "../src/engine/optimise";
import { ModelPredictor, OraclePredictor } from "../src/engine/predictor";
import { Rng } from "../src/engine/rng";

describe("objectives and optimiser", () => {
  it("weighted quantile/CVaR reduce to unweighted versions", () => {
    const r = new Rng(1);
    const v = Array.from({ length: 20000 }, () => r.normal());
    const w = v.map(() => 1);
    const sorted = v.slice().sort((a, b) => a - b);
    expect(Math.abs(weightedQuantile(v, w, 0.05) - sorted[999])).toBeLessThan(1e-12);
    const tail = sorted.slice(0, 2000).reduce((a, b) => a + b, 0) / 2000;
    expect(Math.abs(weightedCvar(v, w, 0.1) - tail)).toBeLessThan(1e-9);
  });
  it("CMA-ES finds a quadratic optimum inside the box", () => {
    const res = maximise((xs) => xs.map((x) => -((x[0] - 0.3) ** 2) - (x[1] - 0.7) ** 2), [0, 0], [1, 1], { budget: 400, seed: 0 });
    expect(Math.abs(res.params[0] - 0.3)).toBeLessThan(0.01);
    expect(Math.abs(res.params[1] - 0.7)).toBeLessThan(0.01);
  });
  it("reproduces the Python oracle finding: robust settings nearly halve failures", () => {
    const prob = new RobustProblem(new OraclePredictor(), ISS_NOMINAL, {}, 128, 8, 5);
    const nom = recommend(prob, "nominal", { budget: 800, seed: 1 });
    const exp = recommend(prob, "expected", { budget: 800, seed: 1 });
    const dep = drawDeployment(ISS_NOMINAL, 3000, 9);
    const dn = deploy(nom.params, dep);
    const de = deploy(exp.params, dep);
    // Python (results/full, oracle): nominal failure 17.6 %, expected 9.6 %; EU 8.05 vs 8.52 MPa.
    expect(dn.failureRate).toBeGreaterThan(0.13);
    expect(de.failureRate).toBeLessThan(0.12);
    expect(de.expectedUtility).toBeGreaterThan(dn.expectedUtility);
    expect(de.expectedUtility).toBeGreaterThan(7.8);
    console.log("oracle nominal", dn.failureRate.toFixed(3), dn.expectedUtility.toFixed(2), "expected", de.failureRate.toFixed(3), de.expectedUtility.toFixed(2));
  });
  it("safe bounds follow the dominant polymer", () => {
    expect(safeBounds(ISS_NOMINAL).low[0]).toBe(210);
    expect(safeBounds({ ...ISS_NOMINAL, mean: [0.1, 0.6, 0.1, 0.1, 0.1] }).low[0]).toBe(255);
  });
});

describe("campaign, model and active learning", () => {
  const jobs = generateCampaign(ISS_NOMINAL, 30, 4, 3);
  it("campaign CSV round-trips through the validator", () => {
    const csv = toCsv(jobs);
    const parsed = parseCsv(csv);
    expect(parsed.errors).toEqual([]);
    expect(parsed.jobs.length).toBe(jobs.length);
    expect(parsed.jobs.filter((j) => j.success).length).toBe(jobs.filter((j) => j.success).length);
    const bad = csv.replace(/\n(E00000,synthetic,B0000,[^,]*,[^,]*,)([^,]*)/, "\n$1" + "0.9");
    expect(parseCsv(bad).errors.join(" ")).toMatch(/sum to/);
    const fs = estimateFeedstock(parsed.jobs);
    fs.mean.forEach((m, k) => expect(Math.abs(m - ISS_NOMINAL.mean[k])).toBeLessThan(0.06));
  });
  it("trains a predictor that beats a constant on held-out batches and proposes experiments", () => {
    const train = jobs.filter((j) => Number(j.batchId.slice(1)) < 22);
    const test = jobs.filter((j) => Number(j.batchId.slice(1)) >= 22 && j.success);
    const model = new ModelPredictor().fit(train, { seed: 0, restarts: 1, iterations: 120 });
    const p = model.predict(test.map((j) => j.composition), test.map((j) => j.moisture), test.map((j) => j.params));
    const y = test.map((j) => Math.log(j.strengthMPa));
    const rep = regressionReport(y, Array.from(p.mean), Array.from(p.mean, (_, i) => p.varEpistemic[i] + p.varAleatoric[i]));
    const mu = y.reduce((a, b) => a + b, 0) / y.length;
    const constRmse = Math.sqrt(y.reduce((s, v) => s + (v - mu) ** 2, 0) / y.length);
    expect(rep.rmseLog).toBeLessThan(constRmse);
    expect(rep.coverage90).toBeGreaterThan(0.5);
    const prob = new RobustProblem(model, ISS_NOMINAL, {}, 32, 8, 1);
    const rec = recommend(prob, "cvar", { budget: 300, seed: 2 });
    for (const strategy of ["targeted_ivr", "cost_aware_ivr", "max_variance", "space_filling", "random"] as const) {
      const sel = selectExperiments(model, ISS_NOMINAL, 4, strategy, new Rng(4), rec.params, 96);
      expect(sel.length).toBe(4);
      sel.forEach((c) => expect(Math.abs(c.composition.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-9));
    }
  });
});
