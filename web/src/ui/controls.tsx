/** Small, accessible form and display controls. */

import type { ReactNode } from "react";
import { fmt } from "../charts/kit";
import { COMPONENTS } from "../engine/materials";

export function Slider({ id, label, value, min, max, step, onChange, format, hint }: { id: string; label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string; hint?: ReactNode }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <output htmlFor={id}>{format ? format(value) : fmt(value)}</output>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

export function Readout({ label, value, unit, sub }: { label: string; value: ReactNode; unit?: string; sub?: ReactNode }) {
  return (
    <div className="readout">
      <span className="label">{label}</span>
      <span className="value">
        {value}
        {unit && <small>{unit}</small>}
      </span>
      {sub && <span className="sub">{sub}</span>}
    </div>
  );
}

export function Progress({ fraction, label }: { fraction: number; label: string }) {
  return (
    <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)} aria-label={label}>
      <div className="small muted">{label} · {Math.round(fraction * 100)}%</div>
      <div className="progress">
        <span style={{ width: `${Math.round(fraction * 100)}%` }} />
      </div>
    </div>
  );
}

export function CompositionBar({ composition, title }: { composition: number[]; title?: string }) {
  return (
    <div className="comp-bar" role="img" aria-label={title ?? COMPONENTS.map((c, i) => `${c.label} ${(composition[i] * 100).toFixed(1)}%`).join(", ")}>
      {COMPONENTS.map((c, i) => (
        <span key={c.name} style={{ width: `${composition[i] * 100}%`, background: `var(--s${i + 1})` }} title={`${c.label} ${(composition[i] * 100).toFixed(1)}%`} />
      ))}
    </div>
  );
}

export function SuccessPill({ p }: { p: number }) {
  const cls = p >= 0.9 ? "good" : p >= 0.75 ? "warn" : "bad";
  const word = p >= 0.9 ? "reliable" : p >= 0.75 ? "marginal" : "unreliable";
  return (
    <span className={`pill ${cls}`}>
      {(p * 100).toFixed(0)}% print · {word}
    </span>
  );
}

export const pct = (v: number, digits = 1) => (Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : "–");
export const mpa = (v: number, digits = 2) => (Number.isFinite(v) ? v.toFixed(digits) : "–");

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {children}
    </div>
  );
}
