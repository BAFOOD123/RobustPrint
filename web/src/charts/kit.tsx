/**
 * A small SVG chart kit: linear scales, nice ticks, axes, responsive width and a hover
 * tooltip layer. Colours come from CSS custom properties so charts follow the theme.
 */

import { type ReactNode, useLayoutEffect, useRef, useState } from "react";

export const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)", "var(--s7)", "var(--s8)"];

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
  invert(px: number): number;
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain[0] === domain[1] ? [domain[0] - 1, domain[1] + 1] : domain;
  const [r0, r1] = range;
  const f = ((v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)) as Scale;
  f.domain = [d0, d1];
  f.range = range;
  f.invert = (px: number) => d0 + ((px - r0) / (r1 - r0)) * (d1 - d0);
  return f;
}

/** Round ticks covering [lo, hi] with about `count` steps. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const span = hi - lo;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? 10 * mag;
  const start = Math.ceil(lo / step - 1e-9) * step;
  const out: number[] = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

export function niceDomain(values: number[], pad = 0.05, includeZero = false): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [0, 1];
  let lo = Math.min(...finite);
  let hi = Math.max(...finite);
  if (includeZero) lo = Math.min(lo, 0);
  const span = hi - lo || Math.abs(hi) || 1;
  lo -= span * (includeZero && lo === 0 ? 0 : pad);
  hi += span * pad;
  const t = niceTicks(lo, hi, 5);
  const step = t.length > 1 ? t[1] - t[0] : span;
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

export function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 1e5)) return v.toExponential(1);
  return Number(v.toPrecision(digits)).toLocaleString("en-US", { maximumFractionDigits: 4 });
}

/** Width of a container element, updated on resize. */
export function useWidth<T extends HTMLElement>(fallback = 600): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(el.clientWidth || fallback);
    const ro = new ResizeObserver((entries) => {
      const cw = entries[0]?.contentRect.width;
      if (cw) setW(cw);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [fallback]);
  return [ref, w];
}

export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export const MARGIN: Margin = { top: 12, right: 14, bottom: 40, left: 52 };

interface AxesProps {
  x: Scale;
  y: Scale;
  width: number;
  height: number;
  margin: Margin;
  xLabel?: string;
  yLabel?: string;
  xTicks?: number[];
  yTicks?: number[];
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
}

export function Axes({ x, y, width, height, margin, xLabel, yLabel, xTicks, yTicks, xFormat = (v) => fmt(v), yFormat = (v) => fmt(v) }: AxesProps) {
  const xt = xTicks ?? niceTicks(x.domain[0], x.domain[1], Math.max(2, Math.floor((width - margin.left - margin.right) / 80)));
  const yt = yTicks ?? niceTicks(y.domain[0], y.domain[1], Math.max(2, Math.floor((height - margin.top - margin.bottom) / 40)));
  const bottom = height - margin.bottom;
  return (
    <g className="axes">
      {yt.map((t) => (
        <g key={`y${t}`}>
          <line x1={margin.left} x2={width - margin.right} y1={y(t)} y2={y(t)} className="gridline" />
          <text x={margin.left - 6} y={y(t)} dy="0.32em" textAnchor="end" className="tick">
            {yFormat(t)}
          </text>
        </g>
      ))}
      {xt.map((t) => (
        <text key={`x${t}`} x={x(t)} y={bottom + 16} textAnchor="middle" className="tick">
          {xFormat(t)}
        </text>
      ))}
      <line x1={margin.left} x2={width - margin.right} y1={bottom} y2={bottom} className="baseline" />
      {xLabel && (
        <text x={(margin.left + width - margin.right) / 2} y={height - 6} textAnchor="middle" className="axis-label">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text transform={`translate(12,${(margin.top + bottom) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">
          {yLabel}
        </text>
      )}
    </g>
  );
}

export interface TooltipState {
  x: number;
  y: number;
  content: ReactNode;
}

/** Positioned tooltip inside a relatively positioned chart container; flips left near the edge. */
export function Tooltip({ tip, width }: { tip: TooltipState | null; width: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [tipWidth, setTipWidth] = useState(160);
  useLayoutEffect(() => {
    if (ref.current) setTipWidth(ref.current.offsetWidth);
  }, [tip]);
  if (!tip) return null;
  const right = tip.x + 14;
  const left = right + tipWidth > width ? Math.max(tip.x - tipWidth - 14, 4) : right;
  return (
    <div ref={ref} className="tooltip" style={{ left, top: Math.max(tip.y - 12, 0) }} role="status">
      {tip.content}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label} className="legend-item">
          <svg width="18" height="10" aria-hidden="true">
            <line x1="1" x2="17" y1="5" y2="5" stroke={it.color} strokeWidth="2.5" strokeDasharray={it.dashed ? "4 3" : undefined} strokeLinecap="round" />
          </svg>
          {it.label}
        </span>
      ))}
    </div>
  );
}
