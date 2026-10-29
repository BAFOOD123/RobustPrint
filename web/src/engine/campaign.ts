/**
 * Virtual print campaigns in the experiment CSV schema (data/schema.md), plus CSV export,
 * parsing and validation with row-level messages (port of robustprint.data.validation).
 */

import { closure } from "./composition";
import { COMPONENTS, type Feedstock } from "./materials";
import { latinHypercube } from "./optimise";
import type { TrainingJob } from "./predictor";
import { Rng } from "./rng";
import { type Batch, printJob, sampleBatches, trueOutcome } from "./simulator";
import { PARAMETERS, fromUnit } from "./space";

export interface Job extends TrainingJob {
  experimentId: string;
  batchId: string;
  origin: "synthetic" | "experimental";
  /** Present only for simulated jobs: the hidden truth, used for scoring the model. */
  trueComposition?: number[];
  trueStrengthMPa?: number;
  truePSuccess?: number;
  modulusGPa: number;
  elongationPct: number;
  notes: string;
}

export type Design = "lhs" | "random";

/** Simulate print jobs for given batches and settings (n replicates per job). */
export function printJobs(batches: Batch[], settings: number[][], rng: Rng, nReplicates = 3, idStart = 0, idPrefix = "E"): Job[] {
  return settings.map((params, i) => {
    const batch = batches[i % batches.length];
    const truth = trueOutcome(batch.trueComposition, params, batch.moisture, batch.batchEffect);
    const job = printJob(batch, params, rng, nReplicates, truth);
    const success = job.success;
    const s = job.strengthMPa;
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    const m = mean(s);
    const sd = nReplicates > 1 ? Math.sqrt(s.reduce((a, b) => a + (b - m) ** 2, 0) / (nReplicates - 1)) : NaN;
    return {
      experimentId: `${idPrefix}${String(idStart + i).padStart(5, "0")}`,
      batchId: batch.id,
      origin: "synthetic",
      composition: batch.reportedComposition,
      params,
      moisture: batch.moisture,
      success,
      strengthMPa: success ? m : NaN,
      strengthSdMPa: success ? sd : NaN,
      nReplicates,
      modulusGPa: success ? mean(job.modulusGPa) : NaN,
      elongationPct: success ? mean(job.elongationPct) : NaN,
      notes: success ? "" : "simulated print failure",
      trueComposition: batch.trueComposition,
      trueStrengthMPa: truth.strengthMPa,
      truePSuccess: truth.pSuccess,
    };
  });
}

/** A campaign of nBatches x settingsPerBatch jobs following one space-filling design. */
export function generateCampaign(
  feedstock: Feedstock,
  nBatches: number,
  settingsPerBatch: number,
  seed: number,
  opts: { nReplicates?: number; design?: Design; low?: number[]; high?: number[]; batchPrefix?: string; idPrefix?: string; idStart?: number } = {},
): Job[] {
  const rng = new Rng(seed);
  const batches = sampleBatches(feedstock, nBatches, rng, 300, opts.batchPrefix ?? "B");
  const n = nBatches * settingsPerBatch;
  const u = opts.design === "random" ? Array.from({ length: n }, () => PARAMETERS.map(() => rng.random())) : latinHypercube(n, PARAMETERS.length, rng);
  const settings = u.map((row) => fromUnit(row, opts.low, opts.high).map((v, j) => roundTo(v, PARAMETERS[j].step)));
  const assigned = batches.flatMap((b) => Array.from({ length: settingsPerBatch }, () => b));
  return printJobs(assigned, settings, rng, opts.nReplicates ?? 3, opts.idStart ?? 0, opts.idPrefix ?? "E");
}

export function roundTo(v: number, step: number): number {
  const r = Math.round(v / step) * step;
  return Number(r.toFixed(step < 1 ? 2 : 0));
}

export const CSV_COLUMNS = [
  "experiment_id", "data_origin", "batch_id", "date", "source_description",
  ...COMPONENTS.map((c) => `frac_${c.name}`), "moisture_pct",
  ...PARAMETERS.map((p) => p.name),
  "tensile_strength_MPa", "tensile_strength_sd_MPa", "youngs_modulus_GPa", "elongation_pct",
  "print_success", "n_replicates", "defect_notes",
];

function fmt(v: number, digits = 6): string {
  return Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : "";
}

/** Serialise jobs to CSV in the repository schema (hidden truth is not exported). */
export function toCsv(jobs: Job[]): string {
  const lines = [CSV_COLUMNS.join(",")];
  const day = (id: string) => {
    const n = Number(id.replace(/\D/g, "")) || 0;
    const d = new Date(Date.UTC(2026, 0, 5 + n));
    return d.toISOString().slice(0, 10);
  };
  for (const j of jobs) {
    const desc = j.origin === "synthetic" ? "SYNTHETIC: RobustPrint Virtual Lab simulation" : "experimental";
    lines.push(
      [
        j.experimentId, j.origin, j.batchId, day(j.batchId), desc,
        ...j.composition.map((v) => fmt(v)), fmt(j.moisture, 4),
        ...j.params.map((v) => fmt(v, 5)),
        fmt(j.strengthMPa, 5), fmt(j.strengthSdMPa, 5), fmt(j.modulusGPa, 5), fmt(j.elongationPct, 5),
        j.success ? "true" : "false", String(j.nReplicates), j.notes.replace(/,/g, ";"),
      ].join(","),
    );
  }
  return lines.join("\n") + "\n";
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

export interface ParseResult {
  jobs: Job[];
  errors: string[];
  warnings: string[];
}

const TRUE = new Set(["true", "1", "yes", "y", "t"]);
const FALSE = new Set(["false", "0", "no", "n", "f"]);

/** Parse and validate a schema CSV. Rows with errors are excluded and reported. */
export function parseCsv(text: string): ParseResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "" && !l.startsWith("#"));
  if (lines.length < 2) return { jobs: [], errors: ["The file has no data rows."], warnings };
  const header = splitCsvLine(lines[0]);
  const required = ["experiment_id", "data_origin", "batch_id", ...COMPONENTS.map((c) => `frac_${c.name}`), ...PARAMETERS.map((p) => p.name), "print_success", "n_replicates", "tensile_strength_MPa"];
  const missing = required.filter((c) => !header.includes(c));
  if (missing.length) return { jobs: [], errors: [`Missing required column(s): ${missing.join(", ")}.`], warnings };
  const col = (name: string) => header.indexOf(name);
  const jobs: Job[] = [];
  const seen = new Set<string>();
  const batchComp = new Map<string, string>();
  let origins = new Set<string>();
  lines.slice(1).forEach((line, r) => {
    const cells = splitCsvLine(line);
    const get = (name: string) => (col(name) >= 0 ? cells[col(name)] ?? "" : "");
    const id = get("experiment_id");
    const label = `row ${r + 1} (experiment_id=${id})`;
    const rowErrors: string[] = [];
    const num = (name: string, lo: number, hi: number, required = true): number => {
      const raw = get(name);
      if (raw === "") {
        if (required) rowErrors.push(`${label}: ${name} is missing.`);
        return NaN;
      }
      const v = Number(raw);
      if (!Number.isFinite(v)) rowErrors.push(`${label}: ${name}='${raw}' is not a number.`);
      else if (v < lo || v > hi) rowErrors.push(`${label}: ${name}=${v} is outside the physically valid range [${lo}, ${hi}].`);
      return v;
    };
    if (seen.has(id)) rowErrors.push(`${label}: experiment_id is not unique.`);
    seen.add(id);
    const origin = get("data_origin").toLowerCase();
    if (origin !== "synthetic" && origin !== "experimental") rowErrors.push(`${label}: data_origin='${get("data_origin")}' must be 'experimental' or 'synthetic'.`);
    origins.add(origin);
    const fr = COMPONENTS.map((c) => num(`frac_${c.name}`, 0, 1));
    const sum = fr.reduce((a, b) => a + b, 0);
    if (fr.every(Number.isFinite) && Math.abs(sum - 1) > 0.01) rowErrors.push(`${label}: composition fractions sum to ${sum.toFixed(4)}; they must sum to 1 (±0.01). If you recorded percentages, divide by 100.`);
    const params = PARAMETERS.map((p) => num(p.name, p.hardLow, p.hardHigh));
    params.forEach((v, j) => {
      const p = PARAMETERS[j];
      if (Number.isFinite(v) && (v < p.low || v > p.high)) warnings.push(`${label}: ${p.name}=${v} is outside the search range [${p.low}, ${p.high}] ${p.unit}.`);
    });
    const moistureRaw = get("moisture_pct");
    const moisture = moistureRaw === "" ? NaN : num("moisture_pct", 0, 20, false);
    const succRaw = get("print_success").toLowerCase();
    let success = false;
    if (TRUE.has(succRaw)) success = true;
    else if (!FALSE.has(succRaw)) rowErrors.push(`${label}: print_success='${get("print_success")}' must be true/false.`);
    const nRep = num("n_replicates", 1, 1000);
    if (Number.isFinite(nRep) && nRep !== Math.round(nRep)) rowErrors.push(`${label}: n_replicates must be a whole number.`);
    const strength = num("tensile_strength_MPa", 0, 500, false);
    if (success && !Number.isFinite(strength)) warnings.push(`${label}: print succeeded but tensile_strength_MPa is empty; row is used for the success model only.`);
    if (!success && Number.isFinite(strength)) warnings.push(`${label}: print_success is false but a strength is given; the strength is ignored.`);
    const batchId = get("batch_id");
    const key = fr.map((v) => v.toFixed(6)).join("|");
    if (batchComp.has(batchId) && batchComp.get(batchId) !== key) rowErrors.push(`batch ${batchId}: rows disagree on composition; every row of a batch must repeat the same fractions.`);
    batchComp.set(batchId, key);
    if (rowErrors.length) {
      errors.push(...rowErrors);
      return;
    }
    jobs.push({
      experimentId: id,
      batchId,
      origin: origin as Job["origin"],
      composition: closure(fr.map((v) => Math.max(v, 1e-4))),
      params,
      moisture,
      success,
      strengthMPa: success ? strength : NaN,
      strengthSdMPa: Number(get("tensile_strength_sd_MPa") || NaN),
      nReplicates: nRep,
      modulusGPa: Number(get("youngs_modulus_GPa") || NaN),
      elongationPct: Number(get("elongation_pct") || NaN),
      notes: get("defect_notes"),
    });
  });
  const known = jobs.map((j) => j.moisture).filter(Number.isFinite).sort((a, b) => a - b);
  const median = known.length ? known[Math.floor(known.length / 2)] : 0.3;
  jobs.forEach((j) => {
    if (!Number.isFinite(j.moisture)) j.moisture = median;
  });
  origins = new Set([...origins].filter((o) => o === "synthetic" || o === "experimental"));
  if (origins.size > 1) warnings.push("File mixes experimental and synthetic rows; keep them separate when reporting.");
  return { jobs, errors, warnings };
}

/** Maximum-likelihood-free (moment-matched) feedstock estimate from one composition per batch. */
export function estimateFeedstock(jobs: Job[], name = "estimated"): Feedstock {
  const byBatch = new Map<string, Job>();
  jobs.forEach((j) => byBatch.has(j.batchId) || byBatch.set(j.batchId, j));
  const comps = [...byBatch.values()].map((j) => j.composition);
  const d = COMPONENTS.length;
  const mean = Array.from({ length: d }, (_, k) => comps.reduce((s, c) => s + c[k], 0) / comps.length);
  const concs = mean.map((m, k) => {
    const v = comps.reduce((s, c) => s + (c[k] - m) ** 2, 0) / Math.max(comps.length - 1, 1);
    return v > 0 ? (m * (1 - m)) / v - 1 : NaN;
  }).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const conc = concs.length ? Math.max(concs[Math.floor(concs.length / 2)], 1) : 35;
  const logM = [...byBatch.values()].map((j) => Math.log(Math.max(j.moisture, 1e-3)));
  const mLog = logM.reduce((a, b) => a + b, 0) / logM.length;
  const sdLog = logM.length > 1 ? Math.sqrt(logM.reduce((s, v) => s + (v - mLog) ** 2, 0) / (logM.length - 1)) : 0.5;
  return { name, mean: closure(mean), concentration: comps.length > 1 ? conc : 35, moistureMedianPct: Math.exp(mLog), moistureLogSd: Math.max(sdLog, 1e-3) };
}
