/** Chart components: line with bands, scatter with intervals, grouped bars, histogram, composition strip. */

import { type ReactNode, useState } from "react";
import { Axes, fmt, linear, MARGIN, type Margin, Legend, niceDomain, Tooltip, type TooltipState, useWidth } from "./kit";

export interface LinePoint {
  x: number;
  y: number;
  lo?: number;
  hi?: number;
}

export interface LineSeries {
  name: string;
  color: string;
  points: LinePoint[];
  dashed?: boolean;
  band?: boolean;
}

interface LineChartProps {
  series: LineSeries[];
  height?: number;
  xLabel: string;
  yLabel: string;
  marker?: number;
  yDomain?: [number, number];
  xDomain?: [number, number];
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  ariaLabel: string;
  margin?: Margin;
}

/** Lines with optional shaded bands, a crosshair and a tooltip listing every series at the hovered x. */
export function LineChart({ series, height = 220, xLabel, yLabel, marker, yDomain, xDomain, yFormat, xFormat, ariaLabel, margin = MARGIN }: LineChartProps) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<TooltipState | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) => s.points.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]));
  const x = linear(xDomain ?? [Math.min(...xs), Math.max(...xs)], [margin.left, width - margin.right]);
  const y = linear(yDomain ?? niceDomain(ys, 0.05), [height - margin.bottom, margin.top]);
  const path = (pts: LinePoint[], key: "y" | "lo" | "hi") => pts.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(1)},${y(p[key] ?? p.y).toFixed(1)}`).join("");
  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const px = e.clientX - rect.left;
    const xv = x.invert(px);
    const ref0 = series[0]?.points ?? [];
    if (!ref0.length) return;
    let best = ref0[0].x;
    for (const p of ref0) if (Math.abs(p.x - xv) < Math.abs(best - xv)) best = p.x;
    setHoverX(best);
    setTip({
      x: x(best),
      y: e.clientY - rect.top,
      content: (
        <>
          <div className="tip-title">
            {xLabel}: {xFormat ? xFormat(best) : fmt(best)}
          </div>
          {series.map((s) => {
            const p = s.points.find((q) => q.x === best);
            if (!p) return null;
            return (
              <div key={s.name} className="tip-row">
                <span className="swatch" style={{ background: s.color }} />
                {s.name}: <b>{yFormat ? yFormat(p.y) : fmt(p.y)}</b>
                {p.lo !== undefined && p.hi !== undefined && (
                  <span className="muted">
                    {" "}
                    ({yFormat ? yFormat(p.lo) : fmt(p.lo)}–{yFormat ? yFormat(p.hi) : fmt(p.hi)})
                  </span>
                )}
              </div>
            );
          })}
        </>
      ),
    });
  };
  return (
    <div className="chart" ref={ref}>
      {series.length > 1 && <Legend items={series.map((s) => ({ label: s.name, color: s.color, dashed: s.dashed }))} />}
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        <Axes x={x} y={y} width={width} height={height} margin={margin} xLabel={xLabel} yLabel={yLabel} yFormat={yFormat} xFormat={xFormat} />
        {series.map((s) =>
          s.band && s.points.some((p) => p.lo !== undefined) ? (
            <path key={`${s.name}-band`} d={`${path(s.points, "hi")}L${[...s.points].reverse().map((p) => `${x(p.x).toFixed(1)},${y(p.lo ?? p.y).toFixed(1)}`).join("L")}Z`} fill={s.color} opacity={0.14} />
          ) : null,
        )}
        {marker !== undefined && <line x1={x(marker)} x2={x(marker)} y1={margin.top} y2={height - margin.bottom} className="marker" />}
        {series.map((s) => (
          <path key={s.name} d={path(s.points, "y")} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinejoin="round" />
        ))}
        {hoverX !== null && <line x1={x(hoverX)} x2={x(hoverX)} y1={margin.top} y2={height - margin.bottom} className="crosshair" />}
        {hoverX !== null &&
          series.map((s) => {
            const p = s.points.find((q) => q.x === hoverX);
            return p ? <circle key={s.name} cx={x(p.x)} cy={y(p.y)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} /> : null;
          })}
        <rect x={margin.left} y={margin.top} width={Math.max(width - margin.left - margin.right, 0)} height={Math.max(height - margin.top - margin.bottom, 0)} fill="transparent" onPointerMove={onMove} onPointerLeave={() => { setTip(null); setHoverX(null); }} />
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

export interface ScatterPoint {
  x: number;
  y: number;
  lo?: number;
  hi?: number;
  color?: string;
  tip: ReactNode;
}

/** Scatter with optional vertical intervals and a y = x reference line. */
export function ScatterChart({ points, height = 260, xLabel, yLabel, diagonal = false, ariaLabel, domain }: { points: ScatterPoint[]; height?: number; xLabel: string; yLabel: string; diagonal?: boolean; ariaLabel: string; domain?: [number, number] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<TooltipState | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const all = points.flatMap((p) => [p.x, p.y, p.lo ?? p.y, p.hi ?? p.y]);
  const d = domain ?? niceDomain(all, 0.04, true);
  const x = linear(diagonal ? d : niceDomain(points.map((p) => p.x), 0.04), [MARGIN.left, width - MARGIN.right]);
  const y = linear(diagonal ? d : niceDomain(points.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]), 0.04), [height - MARGIN.bottom, MARGIN.top]);
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        <Axes x={x} y={y} width={width} height={height} margin={MARGIN} xLabel={xLabel} yLabel={yLabel} />
        {diagonal && <line x1={x(d[0])} y1={y(d[0])} x2={x(d[1])} y2={y(d[1])} className="reference" />}
        {points.map((p, i) => (
          <g key={i} opacity={active === null || active === i ? 1 : 0.45}>
            {p.lo !== undefined && p.hi !== undefined && <line x1={x(p.x)} x2={x(p.x)} y1={y(p.lo)} y2={y(p.hi)} stroke={p.color ?? "var(--s1)"} strokeOpacity={0.35} strokeWidth={2} />}
            <circle cx={x(p.x)} cy={y(p.y)} r={active === i ? 5.5 : 4} fill={p.color ?? "var(--s1)"} stroke="var(--surface)" strokeWidth={1.5} />
            <circle
              cx={x(p.x)}
              cy={y(p.y)}
              r={10}
              fill="transparent"
              onPointerEnter={() => { setActive(i); setTip({ x: x(p.x), y: y(p.y), content: p.tip }); }}
              onPointerLeave={() => { setActive(null); setTip(null); }}
            />
          </g>
        ))}
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

export interface BarDatum {
  group: string;
  series: string;
  color: string;
  value: number;
  lo?: number;
  hi?: number;
  label?: string;
}

/** Grouped vertical bars with optional CI whiskers and value labels. */
export function BarChart({ data, height = 240, yLabel, ariaLabel, yFormat = (v: number) => fmt(v), yDomain }: { data: BarDatum[]; height?: number; yLabel: string; ariaLabel: string; yFormat?: (v: number) => string; yDomain?: [number, number] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<TooltipState | null>(null);
  const groups = [...new Set(data.map((d) => d.group))];
  const seriesNames = [...new Set(data.map((d) => d.series))];
  const margin = { ...MARGIN, bottom: 46 };
  const y = linear(yDomain ?? niceDomain(data.flatMap((d) => [d.value, d.hi ?? d.value, 0]), 0.08, true), [height - margin.bottom, margin.top]);
  const inner = width - margin.left - margin.right;
  const gw = inner / Math.max(groups.length, 1);
  const bw = Math.min((gw * 0.8) / seriesNames.length, 46);
  const x = linear([0, 1], [margin.left, width - margin.right]);
  return (
    <div className="chart" ref={ref}>
      {seriesNames.length > 1 && <Legend items={seriesNames.map((n) => ({ label: n, color: data.find((d) => d.series === n)!.color }))} />}
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        <Axes x={x} y={y} width={width} height={height} margin={margin} yLabel={yLabel} xTicks={[]} yFormat={yFormat} />
        {groups.map((g, gi) => {
          const cx = margin.left + gw * (gi + 0.5);
          return (
            <g key={g}>
              <text x={cx} y={height - margin.bottom + 16} textAnchor="middle" className="tick">
                {g}
              </text>
              {data
                .filter((d) => d.group === g)
                .map((d) => {
                  const si = seriesNames.indexOf(d.series);
                  const bx = cx - (seriesNames.length * bw) / 2 + si * bw + 1;
                  const top = y(Math.max(d.value, 0));
                  const base = y(0);
                  return (
                    <g key={d.series} onPointerEnter={() => setTip({ x: bx + bw / 2, y: top, content: <><div className="tip-title">{g}</div><div className="tip-row"><span className="swatch" style={{ background: d.color }} />{d.series}: <b>{yFormat(d.value)}</b>{d.lo !== undefined && <span className="muted"> ({yFormat(d.lo)}–{yFormat(d.hi ?? d.value)})</span>}</div></> })} onPointerLeave={() => setTip(null)}>
                      <path d={`M${bx},${base}V${top + 3}Q${bx},${top} ${bx + 3},${top}H${bx + bw - 5}Q${bx + bw - 2},${top} ${bx + bw - 2},${top + 3}V${base}Z`} fill={d.color} />
                      {d.lo !== undefined && d.hi !== undefined && (
                        <g stroke="var(--ink)" strokeWidth={1.2}>
                          <line x1={bx + (bw - 2) / 2} x2={bx + (bw - 2) / 2} y1={y(d.lo)} y2={y(d.hi)} />
                          <line x1={bx + (bw - 2) / 2 - 3} x2={bx + (bw - 2) / 2 + 3} y1={y(d.hi)} y2={y(d.hi)} />
                          <line x1={bx + (bw - 2) / 2 - 3} x2={bx + (bw - 2) / 2 + 3} y1={y(d.lo)} y2={y(d.lo)} />
                        </g>
                      )}
                      {d.label && seriesNames.length <= 3 && (
                        <text x={bx + (bw - 2) / 2} y={Math.min(y(d.hi ?? d.value), top) - 5} textAnchor="middle" className="bar-label">
                          {d.label}
                        </text>
                      )}
                      <rect x={bx} y={margin.top} width={bw} height={Math.max(base - margin.top, 0)} fill="transparent" />
                    </g>
                  );
                })}
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

export interface Distribution {
  name: string;
  color: string;
  values: number[];
}

/** Overlaid step histograms on shared bins (densities), with per-bin tooltip. */
export function Histogram({ dists, height = 220, xLabel, ariaLabel, bins = 30 }: { dists: Distribution[]; height?: number; xLabel: string; ariaLabel: string; bins?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<TooltipState | null>(null);
  const all = dists.flatMap((d) => d.values).filter(Number.isFinite);
  const [lo, hi] = all.length ? [Math.min(...all), Math.max(...all)] : [0, 1];
  const step = (hi - lo || 1) / bins;
  const counts = dists.map((d) => {
    const c = new Array<number>(bins).fill(0);
    const finite = d.values.filter(Number.isFinite);
    finite.forEach((v) => (c[Math.min(Math.floor((v - lo) / step), bins - 1)] += 1));
    return c.map((v) => v / Math.max(finite.length, 1));
  });
  const x = linear([lo, lo + step * bins], [MARGIN.left, width - MARGIN.right]);
  const y = linear([0, Math.max(...counts.flat(), 0.01) * 1.1], [height - MARGIN.bottom, MARGIN.top]);
  return (
    <div className="chart" ref={ref}>
      {dists.length > 1 && <Legend items={dists.map((d) => ({ label: d.name, color: d.color }))} />}
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        <Axes x={x} y={y} width={width} height={height} margin={MARGIN} xLabel={xLabel} yLabel="share of parts" yFormat={(v) => `${Math.round(v * 100)}%`} />
        {counts.map((c, di) => {
          let d = `M${x(lo)},${y(0)}`;
          c.forEach((v, b) => (d += `V${y(v)}H${x(lo + (b + 1) * step)}`));
          d += `V${y(0)}`;
          return (
            <g key={dists[di].name}>
              <path d={d + "Z"} fill={dists[di].color} opacity={0.12} />
              <path d={d} fill="none" stroke={dists[di].color} strokeWidth={2} strokeLinejoin="round" />
            </g>
          );
        })}
        {Array.from({ length: bins }, (_, b) => (
          <rect
            key={b}
            x={x(lo + b * step)}
            y={MARGIN.top}
            width={Math.max(x(lo + (b + 1) * step) - x(lo + b * step), 0)}
            height={Math.max(height - MARGIN.top - MARGIN.bottom, 0)}
            fill="transparent"
            onPointerEnter={() =>
              setTip({
                x: x(lo + (b + 0.5) * step),
                y: MARGIN.top + 10,
                content: (
                  <>
                    <div className="tip-title">
                      {fmt(lo + b * step)}–{fmt(lo + (b + 1) * step)} MPa
                    </div>
                    {dists.map((dd, di) => (
                      <div key={dd.name} className="tip-row">
                        <span className="swatch" style={{ background: dd.color }} />
                        {dd.name}: <b>{(counts[di][b] * 100).toFixed(1)}%</b>
                      </div>
                    ))}
                  </>
                ),
              })
            }
            onPointerLeave={() => setTip(null)}
          />
        ))}
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

/** Horizontal bars (e.g. relevance or variance shares). */
export function HBarChart({ rows, ariaLabel, format = (v: number) => fmt(v), color = "var(--s1)" }: { rows: { label: string; value: number; color?: string }[]; ariaLabel: string; format?: (v: number) => string; color?: string }) {
  const max = Math.max(...rows.map((r) => r.value), 1e-12);
  return (
    <div className="hbars" role="img" aria-label={ariaLabel}>
      {rows.map((r) => (
        <div key={r.label} className="hbar-row">
          <span className="hbar-label">{r.label}</span>
          <span className="hbar-track">
            <span className="hbar-fill" style={{ width: `${(100 * Math.max(r.value, 0)) / max}%`, background: r.color ?? color }} />
          </span>
          <span className="hbar-value">{format(r.value)}</span>
        </div>
      ))}
    </div>
  );
}
