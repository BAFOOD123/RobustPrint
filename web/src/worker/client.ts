/** Promise-based client for the compute worker (one request at a time per call site). */

import EngineWorker from "./engine.worker?worker&inline";
import type { Envelope, Request, Result, WorkerMessage } from "./protocol";

type Pending = { resolve: (r: Result) => void; reject: (e: Error) => void; onProgress?: (f: number, label: string) => void };

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new EngineWorker();
  worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
    const msg = event.data;
    const p = pending.get(msg.id);
    if (!p) return;
    if (msg.type === "progress") p.onProgress?.(msg.fraction, msg.label);
    else {
      pending.delete(msg.id);
      if (msg.type === "done") p.resolve(msg.result);
      else p.reject(new Error(msg.message));
    }
  };
  worker.onerror = (e) => {
    pending.forEach((p) => p.reject(new Error(e.message || "The compute worker stopped.")));
    pending.clear();
  };
  return worker;
}

export function run<R extends Result>(payload: Request, onProgress?: (f: number, label: string) => void): Promise<R> {
  const w = ensureWorker();
  const id = nextId++;
  return new Promise<R>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (r: Result) => void, reject, onProgress });
    const msg: Envelope<Request> = { id, payload };
    w.postMessage(msg);
  });
}
