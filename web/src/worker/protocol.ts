/** Messages between the page and the compute worker. */

import type { Strategy } from "../engine/acquisition";
import type { Job } from "../engine/campaign";
import type { DeploymentResult } from "../engine/deploy";
import type { Feedstock } from "../engine/materials";
import type { MixtureStats, NominalStats, Objective } from "../engine/objectives";
import type { ModelSummary } from "../engine/predictor";

export interface TrainRequest {
  kind: "train";
  jobs: Job[];
  seed: number;
}

export interface ParityPoint {
  measured: number;
  predicted: number;
  lo: number;
  hi: number;
  truth: number | null;
  batchId: string;
}

export interface TrainResult {
  kind: "train";
  summary: ModelSummary;
  holdout: {
    nTrainBatches: number;
    nTestBatches: number;
    nTest: number;
    rmseMPa: number;
    nll: number;
    crps: number;
    coverage90: number;
    qce: number;
    brier: number | null;
    baseRate: number | null;
    parity: ParityPoint[];
    calibration: { level: number; observed: number }[];
    reliability: { meanP: number; freq: number; count: number }[];
  } | null;
  /** Inverse length scales of the final model (larger = the model reacts more to that input). */
  relevance: { name: string; value: number }[];
  /** Synthetic data only: the model's noise estimate vs the truth. */
  noiseCheck: { modelNoiseVar: number; trueNoiseVar: number } | null;
  seconds: number;
}

export interface OptimiseRequest {
  kind: "optimise";
  objectives: Objective[];
  predictor: "model" | "oracle";
  feedstock: Feedstock;
  deployFeedstock: Feedstock;
  budget: number;
  nCompositions: number;
  deployBatches: number;
  seed: number;
}

export interface Recommendation {
  objective: Objective;
  predictor: "model" | "oracle";
  params: number[];
  predicted: MixtureStats & NominalStats;
  deployed: Omit<DeploymentResult, "strengths"> & { strengths: number[] };
  trace: number[];
}

export interface OptimiseResult {
  kind: "optimise";
  recommendations: Recommendation[];
  seconds: number;
}

export interface ProposeRequest {
  kind: "propose";
  strategy: Strategy;
  n: number;
  feedstock: Feedstock;
  seed: number;
}

export interface Proposal {
  composition: number[];
  moisture: number;
  params: number[];
  score: number;
  cost: number;
  predMedian: number;
  predLo: number;
  predHi: number;
  pSuccess: number;
}

export interface ProposeResult {
  kind: "propose";
  proposals: Proposal[];
  recommended: number[];
  /** True quality of the current CVaR recommendation (synthetic world only). */
  quality: { trueCvar: number; truePSuccess: number; trueExpectedUtility: number; regretUtility: number };
  seconds: number;
}

export interface EvaluateRequest {
  kind: "evaluate";
  params: number[];
  feedstock: Feedstock;
  seed: number;
}

export interface EvaluateResult {
  kind: "evaluate";
  model: (MixtureStats & NominalStats) | null;
  truth: MixtureStats & NominalStats;
}

export type Request = TrainRequest | OptimiseRequest | ProposeRequest | EvaluateRequest;
export type Result = TrainResult | OptimiseResult | ProposeResult | EvaluateResult;

export interface Envelope<T> {
  id: number;
  payload: T;
}

export type WorkerMessage =
  | { id: number; type: "progress"; fraction: number; label: string }
  | { id: number; type: "done"; result: Result }
  | { id: number; type: "error"; message: string };
