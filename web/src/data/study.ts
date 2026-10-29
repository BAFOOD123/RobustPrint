/** Typed view of study.json (written by scripts/export_web_data.py from results/full; synthetic). */

import raw from "./study.json";

export type Triple = [number | null, number | null, number | null];
export interface StudyTest { method: string; metric: string; diff: number | null; lo: number | null; hi: number | null; pHolm: number | null }
export interface StudyData {
  label: string;
  version: string;
  commit: string;
  e1: { seeds: number; trainJobs: number; meanSuccessfulJobs: number | null; methods: Record<string, Record<string, Triple>>; tests: StudyTest[] };
  e1b: { variant: string; method: string; expected_utility: number | null; failure_rate: number | null; q05_strength: number | null }[];
  e1bSignificant: { variant: string; method: string; metric: string; diff: number | null; pHolm: number | null }[];
  e2: {
    sizes: number[];
    crps: Record<string, (number | null)[]>;
    largest: { model: string; rmse: number | null; nll: number | null; crps: number | null; coverage: number | null }[];
    conformal: { model: string; split: string; coverage: number | null }[];
  };
  e3: Record<string, { nJobs: number; mean: number | null; lo: number | null; hi: number | null }[]>;
  e3Seeds: number;
  e3Contrasts: { strategy: string; diff: number | null; lo: number | null; hi: number | null }[];
  e4: Record<string, string | number | null>[];
  e5: { output: string; input: string; first: number | null; total: number | null }[];
}

export const study = raw as unknown as StudyData;
