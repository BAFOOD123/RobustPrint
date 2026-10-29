/**
 * Compute handlers (pure functions of the request, plus the worker's fitted model).
 * Kept separate from the worker entry point so they can be unit-tested in Node.
 */

import { selectExperiments } from "../engine/acquisition";
import type { Job } from "../engine/campaign";
import { deploy, drawDeployment } from "../engine/deploy";
import { features } from "../engine/features";
import { ISS_NOMINAL, type Feedstock } from "../engine/materials";
import { brierScore, calibrationCurve, regressionReport, reliabilityCurve } from "../engine/metrics";
import { type Objective, RobustProblem } from "../engine/objectives";
import { recommend } from "../engine/optimise";
import { ModelPredictor, OraclePredictor, type Predictor } from "../engine/predictor";
import { deriveSeed, Rng } from "../engine/rng";
import { CONFIG, trueOutcome } from "../engine/simulator";
import { PARAMETERS } from "../engine/space";
import type {
  EvaluateRequest,
  EvaluateResult,
  OptimiseRequest,
  OptimiseResult,
  ParityPoint,
  ProposeRequest,
  ProposeResult,
  TrainRequest,
  TrainResult,
} from "./protocol";

export type Progress = (fraction: number, label: string) => void;

export interface ComputeState {
  model: ModelPredictor | null;
  oracleCache: Map<string, { truth: RobustProblem; expectedUtility: number }>;
}

export function createState(): ComputeState {
  return { model: null, oracleCache: new Map() };
}

const FEATURE_NAMES = ["ilr 1 (PE vs rest)", "ilr 2", "ilr 3", "ilr 4 (contam. vs rest)", ...PARAMETERS.map((p) => p.label), "log moisture"];

function fitOptions(n: number) {
  return n > 220 ? { restarts: 0, iterations: 110 } : n > 120 ? { restarts: 1, iterations: 120 } : { restarts: 2, iterations: 140 };
}

/** Train with a grouped hold-out evaluation (last quarter of batches), then refit on all data. */
export function train(state: ComputeState, req: TrainRequest, progress: Progress): TrainResult {
  const t0 = performance.now();
  const jobs = req.jobs;
  const batches = [...new Set(jobs.map((j) => j.batchId))];
  const nTest = Math.max(1, Math.floor(batches.length / 4));
  const testIds = new Set(new Rng(deriveSeed(req.seed, 7)).shuffle(batches.slice()).slice(0, nTest));
  const trainJobs = jobs.filter((j) => !testIds.has(j.batchId));
  const testJobs = jobs.filter((j) => testIds.has(j.batchId));
  let holdout: TrainResult["holdout"] = null;
  const okTrain = trainJobs.filter((j) => j.success && j.strengthMPa > 0).length;
  if (batches.length >= 4 && okTrain >= 5) {
    progress(0.02, "Fitting hold-out model");
    const evalModel = new ModelPredictor().fit(trainJobs, {
      seed: req.seed,
      ...fitOptions(okTrain),
      onProgress: (f) => progress(0.02 + 0.45 * f, "Fitting hold-out model"),
    });
    const okTest = testJobs.filter((j) => j.success && j.strengthMPa > 0);
    if (okTest.length > 0) {
      const p = evalModel.predict(okTest.map((j) => j.composition), okTest.map((j) => j.moisture), okTest.map((j) => j.params));
      const y = okTest.map((j) => Math.log(j.strengthMPa));
      const mean = Array.from(p.mean);
      const variance = mean.map((_, i) => p.varEpistemic[i] + p.varAleatoric[i]);
      const rep = regressionReport(y, mean, variance);
      const parity: ParityPoint[] = okTest.map((j, i) => ({
        measured: j.strengthMPa,
        predicted: Math.exp(mean[i]),
        lo: Math.exp(mean[i] - 1.645 * Math.sqrt(variance[i])),
        hi: Math.exp(mean[i] + 1.645 * Math.sqrt(variance[i])),
        truth: j.trueStrengthMPa ?? null,
        batchId: j.batchId,
      }));
      const pt = evalModel.predict(testJobs.map((j) => j.composition), testJobs.map((j) => j.moisture), testJobs.map((j) => j.params)).pSuccess;
      const succ = testJobs.map((j) => j.success);
      const twoClasses = succ.some(Boolean) && succ.some((s) => !s);
      holdout = {
        nTrainBatches: batches.length - nTest,
        nTestBatches: nTest,
        nTest: okTest.length,
        rmseMPa: rep.rmseMPa,
        nll: rep.nll,
        crps: rep.crps,
        coverage90: rep.coverage90,
        qce: rep.qce,
        brier: twoClasses ? brierScore(succ, pt) : null,
        baseRate: succ.filter(Boolean).length / succ.length,
        parity,
        calibration: calibrationCurve(y, mean, variance),
        reliability: reliabilityCurve(succ, pt, 5),
      };
    }
  }
  progress(0.5, "Fitting final model on all data");
  const okAll = jobs.filter((j) => j.success && j.strengthMPa > 0).length;
  const model = new ModelPredictor().fit(jobs, {
    seed: req.seed + 1,
    ...fitOptions(okAll),
    onProgress: (f) => progress(0.5 + 0.48 * f, "Fitting final model on all data"),
  });
  state.model = model;
  const ls = model.gp.lengthScales;
  const synthetic = jobs.filter((j) => j.success && j.trueComposition);
  let noiseCheck: TrainResult["noiseCheck"] = null;
  if (synthetic.length) {
    const trueNoise = synthetic.reduce((s, j) => {
      const t = trueOutcome(j.trueComposition as number[], j.params, j.moisture);
      return s + t.noiseCv ** 2 / j.nReplicates + CONFIG.batchEffectSd ** 2;
    }, 0) / synthetic.length;
    noiseCheck = { modelNoiseVar: model.gp.noiseVariance, trueNoiseVar: trueNoise };
  }
  return {
    kind: "train",
    summary: model.summary!,
    holdout,
    relevance: ls.map((l, i) => ({ name: FEATURE_NAMES[i], value: 1 / l })),
    noiseCheck,
    seconds: (performance.now() - t0) / 1000,
  };
}

function predictorFor(state: ComputeState, kind: "model" | "oracle"): Predictor {
  if (kind === "oracle") return new OraclePredictor();
  if (!state.model) throw new Error("Train a model first (Model tab).");
  return state.model;
}

function feedstockKey(f: Feedstock): string {
  return [...f.mean.map((v) => v.toFixed(4)), f.concentration, f.moistureMedianPct, f.moistureLogSd].join("|");
}

export function optimise(state: ComputeState, req: OptimiseRequest, progress: Progress): OptimiseResult {
  const t0 = performance.now();
  const predictor = predictorFor(state, req.predictor);
  const problem = new RobustProblem(predictor, req.feedstock, {}, req.nCompositions, 8, deriveSeed(req.seed, 1));
  const dep = drawDeployment(req.deployFeedstock, req.deployBatches, deriveSeed(req.seed, 2));
  const recommendations = req.objectives.map((objective: Objective, k) => {
    const res = recommend(problem, objective, {
      budget: req.budget,
      seed: deriveSeed(req.seed, 3, k),
      onProgress: (f) => progress((k + f) / req.objectives.length, `Optimising ${objective}`),
    });
    const predicted = { ...problem.mixture([res.params])[0], ...problem.nominal([res.params])[0] };
    return { objective, predictor: req.predictor, params: res.params, predicted, deployed: deploy(res.params, dep), trace: res.trace };
  });
  return { kind: "optimise", recommendations, seconds: (performance.now() - t0) / 1000 };
}

/**
 * Regret reference, as in the Python E3: the perfect model's optimum of the tracked (CVaR)
 * objective, cached per feedstock together with the large fixed truth sample it is scored on.
 */
function oracleReference(state: ComputeState, feedstock: Feedstock): { truth: RobustProblem; expectedUtility: number } {
  const key = feedstockKey(feedstock);
  const cached = state.oracleCache.get(key);
  if (cached) return cached;
  const truth = new RobustProblem(new OraclePredictor(), feedstock, {}, 256, 8, 97);
  const best = recommend(truth, "cvar", { budget: 1500, seed: 98 }).params;
  const st = truth.mixture([best])[0];
  const value = { truth, expectedUtility: st.mean * st.pSuccess };
  state.oracleCache.set(key, value);
  return value;
}

export function propose(state: ComputeState, req: ProposeRequest, progress: Progress): ProposeResult {
  const t0 = performance.now();
  if (!state.model) throw new Error("Train a model first (Model tab).");
  const model = state.model;
  progress(0.1, "Finding the current robust (CVaR) setting");
  const problem = new RobustProblem(model, req.feedstock, {}, 64, 8, deriveSeed(req.seed, 11));
  const rec = recommend(problem, "cvar", { budget: 800, seed: deriveSeed(req.seed, 12) }).params;
  progress(0.5, "Scoring candidate experiments");
  const sel = selectExperiments(model, req.feedstock, req.n, req.strategy, new Rng(deriveSeed(req.seed, 13)), rec, 256);
  const pred = model.predict(sel.map((c) => c.composition), sel.map((c) => c.moisture), sel.map((c) => c.params));
  progress(0.8, "Scoring the recommendation on the hidden truth");
  const ref = oracleReference(state, req.feedstock);
  const truth = ref.truth.mixture([rec])[0];
  const utility = truth.mean * truth.pSuccess;
  return {
    kind: "propose",
    proposals: sel.map((c, i) => {
      const sd = Math.sqrt(pred.varEpistemic[i] + pred.varAleatoric[i]);
      return {
        ...c,
        predMedian: Math.exp(pred.mean[i]),
        predLo: Math.exp(pred.mean[i] - 1.645 * sd),
        predHi: Math.exp(pred.mean[i] + 1.645 * sd),
        pSuccess: pred.pSuccess[i],
      };
    }),
    recommended: rec,
    quality: { trueCvar: truth.cvar, truePSuccess: truth.pSuccess, trueExpectedUtility: utility, regretUtility: ref.expectedUtility - utility },
    seconds: (performance.now() - t0) / 1000,
  };
}

export function evaluate(state: ComputeState, req: EvaluateRequest): EvaluateResult {
  const truthProblem = new RobustProblem(new OraclePredictor(), req.feedstock, {}, 256, 8, deriveSeed(req.seed, 5));
  const truth = { ...truthProblem.mixture([req.params])[0], ...truthProblem.nominal([req.params])[0] };
  let model = null;
  if (state.model) {
    const mp = new RobustProblem(state.model, req.feedstock, {}, 128, 8, deriveSeed(req.seed, 5));
    model = { ...mp.mixture([req.params])[0], ...mp.nominal([req.params])[0] };
  }
  return { kind: "evaluate", model, truth };
}

/** Features of jobs (exported for the Model view's relevance labels). */
export function featureNames(): string[] {
  return FEATURE_NAMES;
}

export function defaultFeedstock(): Feedstock {
  return ISS_NOMINAL;
}

export function jobFeatures(j: Job): number[] {
  return features(j.composition, j.params, j.moisture);
}
