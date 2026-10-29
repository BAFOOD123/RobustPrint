/// <reference lib="webworker" />
/** Web Worker entry: runs training, optimisation and experiment selection off the UI thread. */

import { createState, evaluate, optimise, propose, train } from "./compute";
import type { Envelope, Request, Result, WorkerMessage } from "./protocol";

const state = createState();
const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (event: MessageEvent<Envelope<Request>>) => {
  const { id, payload } = event.data;
  const post = (msg: WorkerMessage) => ctx.postMessage(msg);
  let last = 0;
  const progress = (fraction: number, label: string) => {
    const now = performance.now();
    if (now - last > 80 || fraction >= 1) {
      last = now;
      post({ id, type: "progress", fraction, label });
    }
  };
  try {
    let result: Result;
    switch (payload.kind) {
      case "train":
        result = train(state, payload, progress);
        break;
      case "optimise":
        result = optimise(state, payload, progress);
        break;
      case "propose":
        result = propose(state, payload, progress);
        break;
      default:
        result = evaluate(state, payload);
    }
    post({ id, type: "done", result });
  } catch (err) {
    post({ id, type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
