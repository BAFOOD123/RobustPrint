/** Application state shared by all views (React context + reducer). */

import { createContext, type Dispatch, type ReactNode, useContext, useReducer } from "react";
import type { Strategy } from "./engine/acquisition";
import type { Job } from "./engine/campaign";
import { type Feedstock, ISS_NOMINAL } from "./engine/materials";
import { defaults } from "./engine/space";
import type { OptimiseResult, Proposal, ProposeResult, TrainResult } from "./worker/protocol";

export interface AlRound {
  round: number;
  nJobs: number;
  nSuccess: number;
  strategy: Strategy;
  trueExpectedUtility: number;
  regretUtility: number;
  trueCvar: number;
  truePSuccess: number;
  cumulativeCost: number;
}

export interface AppState {
  feedstock: Feedstock;
  settings: number[];
  jobs: Job[];
  dataSource: "none" | "synthetic" | "uploaded";
  dataLabel: string;
  trainedOn: number;
  train: TrainResult | null;
  optimise: OptimiseResult | null;
  proposals: Proposal[] | null;
  lastPropose: ProposeResult | null;
  rounds: AlRound[];
  alCost: number;
  seed: number;
}

export type Action =
  | { type: "feedstock"; feedstock: Feedstock }
  | { type: "settings"; settings: number[] }
  | { type: "setting"; index: number; value: number }
  | { type: "jobs"; jobs: Job[]; source: AppState["dataSource"]; label: string }
  | { type: "appendJobs"; jobs: Job[] }
  | { type: "trained"; result: TrainResult; nJobs: number }
  | { type: "optimised"; result: OptimiseResult }
  | { type: "proposed"; result: ProposeResult }
  | { type: "round"; round: AlRound }
  | { type: "clearProposals" }
  | { type: "addCost"; cost: number }
  | { type: "reseed" };

export const INITIAL: AppState = {
  feedstock: ISS_NOMINAL,
  settings: [265, 100, 25, 0.16, 100, 0, 100, 10],
  jobs: [],
  dataSource: "none",
  dataLabel: "",
  trainedOn: 0,
  train: null,
  optimise: null,
  proposals: null,
  lastPropose: null,
  rounds: [],
  alCost: 0,
  seed: 20261028,
};

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "feedstock":
      return { ...state, feedstock: action.feedstock };
    case "settings":
      return { ...state, settings: action.settings };
    case "setting": {
      const settings = state.settings.slice();
      settings[action.index] = action.value;
      return { ...state, settings };
    }
    case "jobs":
      return { ...state, jobs: action.jobs, dataSource: action.source, dataLabel: action.label, train: null, optimise: null, proposals: null, lastPropose: null, rounds: [], alCost: 0, trainedOn: 0 };
    case "appendJobs":
      return { ...state, jobs: [...state.jobs, ...action.jobs], proposals: null };
    case "trained":
      return { ...state, train: action.result, trainedOn: action.nJobs };
    case "optimised":
      return { ...state, optimise: action.result };
    case "proposed":
      return { ...state, proposals: action.result.proposals, lastPropose: action.result };
    case "round":
      return { ...state, rounds: [...state.rounds, action.round] };
    case "clearProposals":
      return { ...state, proposals: null };
    case "addCost":
      return { ...state, alCost: state.alCost + action.cost };
    case "reseed":
      return { ...state, seed: (state.seed * 1103515245 + 12345) >>> 0 };
  }
}

const Ctx = createContext<{ state: AppState; dispatch: Dispatch<Action> } | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL);
  return <Ctx.Provider value={{ state, dispatch }}>{children}</Ctx.Provider>;
}

export function useApp(): { state: AppState; dispatch: Dispatch<Action> } {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside AppProvider");
  return v;
}

export const DEFAULT_SETTINGS = defaults();
